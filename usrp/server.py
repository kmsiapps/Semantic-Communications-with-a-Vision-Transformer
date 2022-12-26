#%%
import socket
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import tensorflow as tf

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

ARCH = 'CCVVCC'
NUM_SYMBOLS = 512
CKPT_NAME = '../ckpt/CCVVCC_512_15dB_585'
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
serverSock.bind(('', 8080))
print('Listening')
serverSock.listen(1)

while True:
  print('Waiting')
  clientSock, addr = serverSock.accept()
  print(f'Connection: {str(addr)}')

  # Receive image
  receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/cam_received.png')
  images = pilimg.open(f'{TEMP_DIRECTORY}/cam_received.png').convert('RGB')
  plt.imshow(images)
  plt.show()

  # Encode image
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

  data = encoder_network(images)
  i = data[:,:,0].numpy().flatten()
  q = data[:,:,1].numpy().flatten()
  i = np.clip(i / NORMALIZE_CONSTANT * 32767, -32767, 32767)
  q = np.clip(q / NORMALIZE_CONSTANT * 32767, -32767, 32767)
  constellations = to_constellation_array(i, q, i_pilot=True, q_pilot=True)

  # Send constellations
  np.savez_compressed(f'{TEMP_DIRECTORY}/constellations.npz', constellations=constellations)
  send_binary(clientSock, f'{TEMP_DIRECTORY}/constellations.npz')

  # Receive rcv_iq.npz file and decode
  receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/rcv_iq.npz')
  rcv_iq = np.load(f'{TEMP_DIRECTORY}/rcv_iq.npz')['rcv_iq']

  # Decode constellations
  rcv_iq = tf.cast(tf.convert_to_tensor(rcv_iq), tf.float32)
  rcv_iq = tf.reshape(rcv_iq, (BATCH_SIZE, -1, 2))
  proposed_result = decoder_network(rcv_iq)
  tf.keras.utils.save_img(f'{TEMP_DIRECTORY}/decoded_image.png', imBatchtoImage(proposed_result))

  # Send image
  send_binary(clientSock, f'{TEMP_DIRECTORY}/decoded_image.png')

  # TODO: Send effective SNR


# %%
