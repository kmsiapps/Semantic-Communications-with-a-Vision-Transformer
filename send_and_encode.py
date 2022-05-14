#%%

import tensorflow as tf
import numpy as np
import socket

import random
import PIL.Image as pilimg
import matplotlib.pyplot as plt

from config import FILTERS, NUM_BLOCKS, DIM_PER_HEAD, DATA_SIZE, BATCH_SIZE
from models.model import VitCommNet_Encoder_Only, VitCommNet_Decoder_Only
from utils.datasets import dataset_generator

from pilot import p_start, p_end, PILOT_SIZE, SAMPLE_SIZE

# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image

################## CONFIG ####################
best_model = './model_checkpoints/encoder-512.ckpt'
##############################################
encoder_network = VitCommNet_Encoder_Only(
FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  512,
  snrdB=25,
  channel='Rayleigh'
)
encoder_network.load_weights(best_model)

#%%
images = pilimg.open('/dataset/test.png').convert('RGB')
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

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = ("192.168.1.121", 60000)

data = encoder_network(images)

i = tf.reshape(data[0], shape=(-1,))
q = tf.reshape(data[1], shape=(-1,))

plt.title('In-phase')
plt.plot(i)
plt.show()

plt.title('Q-phase')
plt.plot(q)
plt.show()

NORMALIZE_CONSTANT = 80

i = i / NORMALIZE_CONSTANT
q = q / NORMALIZE_CONSTANT

max_i = np.max(i)
max_q = np.max(q)

pwr = np.mean(i ** 2 + q ** 2)

print(f"max_i: {max_i:.2f}, max_q: {max_q:.2f}")
print(f"pwr: {pwr:.2f}")

plt.title('Constellations')
plt.scatter(i, q, s=0.01)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()

i *=  32787
q *=  32787

tf.clip_by_value(i, -32787, 32787)
tf.clip_by_value(q, -32787, 32787)

p_i = np.concatenate([p_start, i, p_end]).astype(np.int16)
p_q = np.concatenate([p_start, q, p_end]).astype(np.int32)

i_ = p_i
q_ = np.left_shift(p_q, 16)
data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)
send_data = data.tobytes()

SEND_SOCK_BUFF_SIZE = 256

for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)
print(f'ORIGIN SEND DONE. len: {len(send_data)}, #data: {len(data)}')

tf.keras.utils.save_img(f'./results/source.png', imBatchtoImage(images))

send_sock.close()

# %%
