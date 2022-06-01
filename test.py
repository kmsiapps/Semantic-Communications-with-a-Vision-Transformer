import PIL.Image as pilimg
import tensorflow as tf
import numpy as np

from models.model import VitCommNet
from config import FILTERS, NUM_BLOCKS, DIM_PER_HEAD, DATA_SIZE, TRAIN_SNRDB, TRAIN_CHANNEL

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

model = VitCommNet(
  FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  DATA_SIZE,
  snrdB=TRAIN_SNRDB,
  channel=TRAIN_CHANNEL
)
model.build(input_shape=(1, 32, 32, 3))
model.load_weights('./epoch_174.ckpt')

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

output = model(images)
tf.keras.utils.save_img(f'/dataset/test_decoded.png', imBatchtoImage(output))
