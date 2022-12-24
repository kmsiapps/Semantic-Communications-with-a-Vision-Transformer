#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
import numpy as np
import socket

import PIL.Image as pilimg
import matplotlib.pyplot as plt
import os

from models.model import SemViT_Encoder_Only
from utils.datasets import dataset_generator
from config.train_config import USRP_HOST, USRP_PORT, NORMALIZE_CONSTANT

from usrp.pilot import p_start_i, p_end_i, p_start_q, p_end_q, PILOT_SIZE, SAMPLE_SIZE

################## CONFIG ####################
ARCH = 'CCVVCC'
NUM_SYMBOLS = 512
CKPT_NAME = './ckpt/CCVVCC_512_0dB_592'
HAS_GDN = False

TARGET_JPEG_RATE = 2048
# Our encoder produces 512 constellations per 32 x 32 patch
# so for 256 * 256 image,
# 512 * 8 * 8 = 32768

# For comparison, we assume JPEG, QPSK, 1/4 coding rates
# (LTE MCS selection, -1.9 ~ 1.8 dB SNR)
# so desired JPEG-encoded file size is
# 32768 / 4 / 4 (=4 QPSK symbols/bytes) ~ 2048 Bytes

'''
LTE MCS scheme (Turbo code-based)
===========================================
SNR         Mod.    Rate    Fair JPEG Bytes
-5 ~ -1.9   QPSK    1/8     1024
-1.9 ~ 1.8  QPSK    1/4     2048
1.8 ~ 3.8   QPSK    1/2     4096
3.8 ~ 7.1   QPSK    2/3     5461.33
7.1 ~ 9.3   16QAM   1/2     8192
9.3 ~ 11.3  16QAM   2/3     10922
11.3 ~ 14.5 64QAM   1/2     12288
14.5 ~ 17.2 64QAM   2/3     16384
17.2 ~ 19.5 64QAM   0.81    19906.56
19.5 ~      64QAM   7/8     21504
'''
##############################################

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

encoder_network = SemViT_Encoder_Only(
	ARCH,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=HAS_GDN,
	num_symbols=NUM_SYMBOLS,
)
encoder_network.load_weights(CKPT_NAME).expect_partial()

test_ds = dataset_generator('/dataset/CIFAR100/test')
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#%%
cam_mode = int(input('Image mode: CIFAR(0) vs CAM(1)?:'))

if cam_mode:
    # for cam image
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
else:
    # for CIFAR-100
    images = next(iter(test_ds))[0]


send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (USRP_HOST, USRP_PORT)

data = encoder_network(images)

i = tf.reshape(data[:,:,0], shape=(-1,))
q = tf.reshape(data[:,:,1], shape=(-1,))

# plt.title('In-phase')
# plt.plot(i)
# plt.show()

# plt.title('Q-phase')
# plt.plot(q)
# plt.show()

i = i / NORMALIZE_CONSTANT
q = q / NORMALIZE_CONSTANT

max_i = np.max(i)
max_q = np.max(q)

pwr = np.mean(i ** 2 + q ** 2)

plt.title('Constellations')
plt.scatter(i, q, s=0.1)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
plt.savefig('./results/sent_constellations.png', dpi=100)

i *=  32767
q *=  32767

i = tf.clip_by_value(i, -32767, 32767)
q = tf.clip_by_value(q, -32767, 32767)


p_i = np.concatenate([p_start_i, i, p_end_i]).astype(np.int16)
p_q = np.concatenate([p_start_q, q, p_end_q]).astype(np.int32)

i_ = p_i
q_ = np.left_shift(p_q, 16)
data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)

send_data = data.tobytes()

SEND_SOCK_BUFF_SIZE = 256

tf.keras.utils.save_img(f'./results/source.png', imBatchtoImage(images))

for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)

print(f'SEND DONE. # Constellations: {len(data)}, ' \
      f'max_i: {max_i:.2f}, max_q: {max_q:.2f}, pwr: {pwr:.2f}')

pil_image = pilimg.fromarray((imBatchtoImage(images).numpy() * 255).astype(np.uint8))

# JPEG rate-match algorithm
quality_max = 100
quality = 50
quality_min = 0

while True:
    pil_image.save('./results/source_jpeg.jpg', quality=quality)
    bytes = os.path.getsize('./results/source_jpeg.jpg')
    if quality == 0 or quality == quality_min or quality == quality_max:
        break
    elif bytes > TARGET_JPEG_RATE and quality_min != quality - 1:
        quality_max = quality
        quality -= (quality - quality_min) // 2
    elif bytes > TARGET_JPEG_RATE and quality_min == quality - 1:
        quality_max = quality
        quality -= 1
    elif bytes < TARGET_JPEG_RATE and quality_max > quality:
        quality_min = quality
        quality += (quality_max - quality) // 2
    else:
        break

print(f'JPEG file size: {bytes} Bytes, Quality: {quality}')

send_sock.close()


# %%
