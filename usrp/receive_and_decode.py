#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import socket
import struct
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import math
import tensorflow as tf
import time

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from usrp.pilot import PILOT_SIZE, SAMPLE_SIZE
from models.model import SemViT_Encoder_Only, SemViT_Decoder_Only
from config.usrp_config import RCV_ADDR, RCV_PORT, NORMALIZE_CONSTANT, USRP_HOST, USRP_PORT, TEMP_DIRECTORY
from config.train_config import BATCH_SIZE
from utils.image import imBatchtoImage
from utils.usrp_utils import get_lci_lcq_compensation, compensate_signal
from utils.networking import receive_constellation_udp

rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = (RCV_ADDR, RCV_PORT)
rcv_sock.bind(rcv_addr)
rcv_sock.settimeout(2)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (USRP_HOST, USRP_PORT)

SOCK_BUFF_SIZE = 16384 # EXPECTED_DATA_LENGTH * 4 + 4
EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - PILOT_SIZE * 2

ARCH = 'CCVVCC'
NUM_SYMBOLS = 512
CKPT_NAME = '../ckpt/CCVVCC_512_0dB_592'
HAS_GDN = False

encoder_network = SemViT_Encoder_Only(
	ARCH,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=HAS_GDN,
	num_symbols=NUM_SYMBOLS,
)

decoder_network = SemViT_Decoder_Only(
	ARCH,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=HAS_GDN,
	num_symbols=NUM_SYMBOLS,
)

encoder_network.load_weights(CKPT_NAME).expect_partial() #'./model_checkpoints/encoder-512.ckpt')
decoder_network.load_weights(CKPT_NAME).expect_partial() #'./model_checkpoints/decoder-512.ckpt')

#%%

LCI=0.0
LCQ=0.0

METADATA_BYTES = 4

if input("Conduct LCI/LCQ compensation? Y/N") == 'Y':
	LCI, LCQ = get_lci_lcq_compensation(rcv_sock, rcv_addr, send_sock, send_addr)
	print(f'Compensation constant set. LCI:{LCI:.4f}, LCQ:{LCQ:.4f}')

#%%
while True:
	data = receive_constellation_udp(rcv_sock)
	array_length = len(data) // 4
	rcv_iq, raw_i, raw_q = compensate_signal(data, LCI, LCQ)
	rcv_iq = tf.cast(tf.convert_to_tensor(rcv_iq), tf.float32)
	rcv_iq = tf.reshape(rcv_iq, (64, -1, 2))

	proposed_result = decoder_network(rcv_iq)
	tf.keras.utils.save_img(f'{TEMP_DIRECTORY}/proposed_usrp.png', imBatchtoImage(proposed_result))

	# for groudntruth computation
	images = pilimg.open(f'{TEMP_DIRECTORY}/source.png').convert('RGB')
	images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0
	h, w, c = images.shape
	images = tf.reshape(images, (1, h, w, c))

	images = tf.image.extract_patches(
		images,
		sizes=[1, 32, 32, 1],
		strides=[1, 32, 32, 1],
		rates=[1, 1, 1, 1],
		padding='VALID'
	)
	images = tf.reshape(images, (-1, 32, 32, c))

	encoding_delay = time.time()
	gt = encoder_network(images)
	encoding_delay = time.time() - encoding_delay

	gt_i = tf.reshape(gt[:, :, 0], shape=(-1,))
	gt_q = tf.reshape(gt[:, :, 1], shape=(-1,))
	decoding_delay = time.time()
	non_error_result = decoder_network(gt)
	decoding_delay = time.time() - decoding_delay

	tf.keras.utils.save_img(f'{TEMP_DIRECTORY}/proposed_gt.png', imBatchtoImage(non_error_result))

	# error computation
	rcv_iq = tf.reshape(rcv_iq, (-1, 2))
	ihat = rcv_iq[:, 0] / NORMALIZE_CONSTANT
	qhat = rcv_iq[:, 1] / NORMALIZE_CONSTANT

	err_i = gt_i/NORMALIZE_CONSTANT - ihat
	err_q = gt_q/NORMALIZE_CONSTANT - qhat

	fig, ax = plt.subplots(1, 3)
	fig.set_figheight(4)
	fig.set_figwidth(21)
	
	ax[0].set_title('Raw I/Q')
	ax[0].plot(raw_i)
	ax[0].plot(raw_q)

	ax[1].set_title('Error')
	ax[1].plot(err_i)
	ax[1].plot(err_q)

	ax[2].set_title('Decded I/Q')
	ax[2].plot(ihat)
	ax[2].plot(qhat)

	fig.tight_layout()
	plt.show()

	signal_power = np.mean(ihat ** 2 + qhat ** 2)
	noise_power = np.mean(err_i ** 2 + err_q ** 2)
	snr = signal_power / noise_power
	snrdB = 10 * math.log10(snr)

	fig, ax = plt.subplots(1, 3)
	fig.set_figheight(7)
	fig.set_figwidth(21)
	
	ax[0].set_title('Original')
	ax[0].imshow(imBatchtoImage(images).numpy())

	psnr = np.mean(tf.image.psnr(images, proposed_result, max_val=1.0))
	ssim = np.mean(tf.image.ssim(images, proposed_result, max_val=1.0))
	ax[1].set_title(f'Proposed\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
	ax[1].imshow(imBatchtoImage(proposed_result).numpy())
	
	image_jpeg = pilimg.open(f'{TEMP_DIRECTORY}/source_jpeg.jpg').convert('RGB')
	image_jpeg = tf.convert_to_tensor(image_jpeg, dtype=tf.float32) / 255.0
	h, w, c = image_jpeg.shape
	image_jpeg = tf.reshape(image_jpeg, (1, h, w, c))
	image_jpeg = tf.image.extract_patches(
		image_jpeg,
		sizes=[1, 32, 32, 1],
		strides=[1, 32, 32, 1],
		rates=[1, 1, 1, 1],
		padding='VALID'
	)
	image_jpeg = tf.reshape(image_jpeg, (-1, 32, 32, c))

	psnr = np.mean(tf.image.psnr(images, image_jpeg, max_val=1.0))
	ssim = np.mean(tf.image.ssim(images, image_jpeg, max_val=1.0))
	ax[2].set_title(f'Conventional (JPEG-based)\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
	ax[2].imshow(imBatchtoImage(image_jpeg).numpy())

	plt.show()

	print(f"Encoding time: {encoding_delay:.2f}s, Decoding time: {decoding_delay:.2f}s")
	print(f"Effective SNR: {snrdB:.2f}dB")


# %%
