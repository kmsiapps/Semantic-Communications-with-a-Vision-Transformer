# %%

import tensorflow as tf
import time
import datetime

from config import *
from models.model import VitCommNet, VitCommNet_Encoder_Only, CnnCommNet
from utils.datasets import dataset_generator
# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

# test_ds = dataset_generator('/dataset/CIFAR10/test/').cache().prefetch(tf.data.experimental.AUTOTUNE)
# train_ds = dataset_generator('/dataset/CIFAR10/train/').cache().prefetch(tf.data.experimental.AUTOTUNE)
test_ds = dataset_generator('/dataset/KODAK_PATCH', mode='validation').cache().prefetch(tf.data.experimental.AUTOTUNE)
train_ds = dataset_generator('/dataset/KODAK_PATCH', mode='training').cache().prefetch(tf.data.experimental.AUTOTUNE)

loss_object = tf.keras.losses.MeanSquaredError() # MeanAbsoluteError() # MeanSquaredError()

first_decay_steps = 5000
initial_learning_rate = 1e-4
'''
lr = (
  tf.keras.optimizers.schedules.CosineDecayRestarts(
      initial_learning_rate,
      first_decay_steps,
      alpha = 0.1))
'''
lr = 1e-4 # 1e-4 # 6*1e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
augment_layer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)
  ])

def normalize_and_augment(image, training):
  image = augment_layer(image, training=training)
  return image

train_ds = train_ds.shuffle(50000, reshuffle_each_iteration=True)
train_ds = train_ds.map(lambda x, y: (normalize_and_augment(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalize(x), y))

# %%

'''
model = CnnCommNet(
  FILTERS,
  NUM_BLOCKS,
  DATA_SIZE,
  snrdB=TRAIN_SNRDB,
  channel=TRAIN_CHANNEL
)
model.load_weights('./model_checkpoints/data-512.ckpt')
'''
'''
model_trained = VitCommNet(
  FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  512,
  snrdB=TRAIN_SNRDB,
  channel=TRAIN_CHANNEL
)
model_trained.load_weights('./model_checkpoints/data-512.ckpt')
model_trained.build(input_shape=(1, 32, 32, 3))
'''
model = VitCommNet(
  FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  DATA_SIZE,
  snrdB=TRAIN_SNRDB,
  channel=TRAIN_CHANNEL
)
model.build(input_shape=(1, 32, 32, 3))
model.load_weights('./model_checkpoints/data-512.ckpt')

'''
model.layers[0].res0.set_weights(model_trained.layers[0].res0.get_weights())
model.layers[0].res1.set_weights(model_trained.layers[0].res1.get_weights())
model.layers[0].vit2.set_weights(model_trained.layers[0].vit2.get_weights())
model.layers[0].vit3.set_weights(model_trained.layers[0].vit3.get_weights())

model.layers[2].vit4.layers[1].set_weights(model_trained.layers[2].vit4.layers[1].get_weights())
model.layers[2].vit4.layers[2].set_weights(model_trained.layers[2].vit4.layers[2].get_weights())

model.layers[2].res5.set_weights(model_trained.layers[2].res5.get_weights())
model.layers[2].res6.set_weights(model_trained.layers[2].res6.get_weights())
model.layers[2].to_image.set_weights(model_trained.layers[2].to_image.get_weights())
'''

'''
model_encoder = VitCommNet_Encoder_Only(
  FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  DATA_SIZE,
  snrdB=TRAIN_SNRDB,
  channel=TRAIN_CHANNEL
)

ckpts = './PWR_0.006532.ckpt'
model.load_weights(ckpts)
model_encoder.load_weights(ckpts)
'''

@tf.function
def train_step(images):
  # constellations = model_encoder(images)
  # i = constellations[0]
  # q = constellations[1]
  # # papr = tf.math.divide_no_nan(tf.reduce_max(i ** 2 + q ** 2), tf.reduce_mean(i ** 2 + q ** 2))
  # pwr = tf.reduce_mean(i ** 2 + q ** 2)

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(images, predictions) #+ PWR_LAMBDA * tf.math.log(pwr)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)

@tf.function

def test_step(images):
  # constellations = model_encoder(images)
  # i = constellations[0]
  # q = constellations[1]
  # # papr = tf.math.divide_no_nan(tf.reduce_max(i ** 2 + q ** 2), tf.reduce_mean(i ** 2 + q ** 2))
  # pwr = tf.reduce_mean(i ** 2 + q ** 2)

  predictions = model(images, training=False)
  t_loss = loss_object(images, predictions) #+ PWR_LAMBDA * tf.math.log(pwr)

  test_loss(t_loss)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

lowest_loss = 100

for epoch in range(1, EPOCHS+1):
  start_time = time.time()
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  test_loss.reset_states()

  one_test_step_time = time.time()
  i = 0
  TIME_ESTIMATION_IDX = len(train_ds) // 10
  for images, labels in train_ds:
      train_step(images)
      # if i == TIME_ESTIMATION_IDX:
      #     print(f'Estimated train epoch time: {len(train_ds) * (time.time() - one_test_step_time) / TIME_ESTIMATION_IDX / 60:.2f} minutes')
      i += 1

  one_test_step_time = time.time()
  i = 0
  for test_images, test_labels in test_ds:
      test_step(test_images)
      # if i == TIME_ESTIMATION_IDX:
      #     print(f'Estimated test epoch time: {len(test_ds) * (time.time() - one_test_step_time) / TIME_ESTIMATION_IDX / 60:.2f} minutes')
      i += 1

  print(
    f'Epoch {epoch}, '
    f'Loss: {train_loss.result():.6f}, '
    f'Test Loss: {test_loss.result():.6f}, '
    f'Training time: {(time.time() - start_time)/60:.2f}m, '
    f'Learning rate: {lr}'
  )

  with train_summary_writer.as_default():
    tf.summary.image("Training data input", images[:10], max_outputs=10, step=0)
    tf.summary.image("Training data output", model(images[:10]), max_outputs=10, step=0)
    tf.summary.scalar('loss', train_loss.result(), step=epoch)

  with test_summary_writer.as_default():
    tf.summary.image("Test data input", test_images[:10], max_outputs=10, step=0)
    tf.summary.image("Test data output", model(test_images[:10]), max_outputs=10, step=0)
    tf.summary.scalar('loss', test_loss.result(), step=epoch)

  # best model save
  if test_loss.result() < lowest_loss:
      lowest_loss = float(test_loss.result())
      model.save_weights(f'epoch_{epoch}.ckpt')
