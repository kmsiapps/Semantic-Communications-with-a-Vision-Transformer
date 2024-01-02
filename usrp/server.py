#%%
import socket
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import tensorflow as tf
import math
import struct

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.model import SemViT_Encoder_Only, SemViT_Decoder_Only
from utils.networking import receive_and_save_binary, send_binary
from utils.image import imBatchtoImage
from utils.usrp_utils import to_constellation_array
from config.train_config import BATCH_SIZE
from config.usrp_config import NORMALIZE_CONSTANT, TEMP_DIRECTORY
from usrp.pilot import PILOT_SIZE

ARCH = 'CCVVCC'
NUM_SYMBOLS = 512
CKPT_NAME = '../bkup_ckpt/best/awgn/CCVVCC_512_10dB_599'
TARGET_JPEG_RATE = 2048

encoder_network = SemViT_Encoder_Only(
	ARCH,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=False,
	num_symbols=NUM_SYMBOLS,
)

decoder_network = SemViT_Decoder_Only(
	ARCH,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=False,
	num_symbols=NUM_SYMBOLS,
)
encoder_network.load_weights(CKPT_NAME).expect_partial()
decoder_network.load_weights(CKPT_NAME).expect_partial()

if not os.path.exists(TEMP_DIRECTORY):
  os.makedirs(TEMP_DIRECTORY)

# Server thing
serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serverSock.bind(('0.0.0.0', 8080))
print('Listening')
serverSock.listen(1)

while True:
  print('Waiting')
  clientSock, addr = serverSock.accept()

  try:
    print(f'Connection: {str(addr)}')

    while True:
        # Whether to receive image or not
        use_cache = bool(int.from_bytes(clientSock.recv(4), byteorder='big', signed=False))
        print('Cache: ', use_cache)

        if use_cache:
          constellations = np.load(f'cifar_constellations.npz')['constellations']
          data = constellations[PILOT_SIZE:-PILOT_SIZE].byteswap()
          i = np.right_shift(np.left_shift(data, 16), 16).astype('>f4')
          q = np.right_shift(data, 16).astype('>f4')
        else:
          # Receive image
          receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/cam_received.png')
          images = pilimg.open(f'{TEMP_DIRECTORY}/cam_received.png').convert('RGB')
          plt.imshow(images)
          plt.show()

          # Encode image
          images = tf.convert_to_tensor(np.array(images), dtype=tf.float32) / 255.0
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

          data = encoder_network(images)
          i = data[:,:,0].numpy().flatten()
          q = data[:,:,1].numpy().flatten()
          i = np.round(np.clip(i / NORMALIZE_CONSTANT * 32767, -32767, 32767))
          q = np.round(np.clip(q / NORMALIZE_CONSTANT * 32767, -32767, 32767))         
          constellations = to_constellation_array(i, q, i_pilot=True, q_pilot=True)

          # Send constellations
          np.savez_compressed(f'{TEMP_DIRECTORY}/constellations.npz', constellations=constellations)
          send_binary(clientSock, f'{TEMP_DIRECTORY}/constellations.npz')

        # Receive rcv_iq.npz file and decode
        receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/rcv_iq.npz')
        rcv_iq = np.load(f'{TEMP_DIRECTORY}/rcv_iq.npz')['rcv_iq']
        rcv_i = np.right_shift(np.left_shift(rcv_iq, 16), 16).astype('>f4') / 32767 * NORMALIZE_CONSTANT
        rcv_q = np.right_shift(rcv_iq, 16).astype('>f4') / 32767 * NORMALIZE_CONSTANT
        
        rcv_iq = np.zeros((len(rcv_i), 2), dtype=np.float32)
        rcv_iq[:, 0] = rcv_i
        rcv_iq[:, 1] = rcv_q

        # Decode constellations
        rcv_iq = tf.cast(tf.convert_to_tensor(rcv_iq), tf.float32)
        rcv_iq = tf.reshape(rcv_iq, (BATCH_SIZE, -1, 2))
        proposed_result = decoder_network(rcv_iq)
        tf.keras.utils.save_img(f'{TEMP_DIRECTORY}/decoded_image.png', imBatchtoImage(proposed_result))

        # Send image
        send_binary(clientSock, f'{TEMP_DIRECTORY}/decoded_image.png')

        # Send effective SNR
        i = i / 32767 * NORMALIZE_CONSTANT
        q = q / 32767 * NORMALIZE_CONSTANT

        noise_power = (rcv_i - i) ** 2 + (rcv_q - q) ** 2
        signal_power = i ** 2 + q ** 2
        effective_snr = 10 * math.log10(tf.reduce_mean(signal_power / (0.001 + noise_power)))
        clientSock.send(struct.pack('!f', effective_snr))

  except Exception as e:
    print('Error:', e)
    clientSock.close()


# %%
