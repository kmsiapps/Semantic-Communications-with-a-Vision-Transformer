# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import argparse

from models.model import SemVit
from models.deepjscc import DeepJSCC
from utils.datasets import dataset_generator

def main(args):
  # Load CIFAR-10 dataset
  train_ds, test_ds = prepare_dataset()

  EXPERIMENT_NAME = args.experiment_name
  print(f'Running {EXPERIMENT_NAME}')

  # TODO: PAPR loss **********************************************
  # strategy = tf.distribute.MultiWorkerMirroredStrategy()
  # with strategy.scope():

  '''
  model = SemVit(
    args.block_types,
    args.filters,
    args.repetitions,
    args.dim_per_head,
    args.data_size,
    snrdB=args.train_snrdB,
    channel=args.channel_types,
    papr_reduction=args.papr_reduction
  )
  '''
  model = DeepJSCC(snrdB=args.train_snrdB, channel=args.channel_types)

  def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)
  
  model.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=1e-4
      ),
      metrics=[
          psnr
      ]
  )
  
  if args.ckpt is not None:
    model.load_weights(args.ckpt)

  save_ckpt = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=f"./ckpt/{EXPERIMENT_NAME}_" + "{epoch}",
          save_best_only=True,
          monitor="val_loss",
          save_weights_only=True,
          options=tf.train.CheckpointOptions(
              experimental_io_device=None, experimental_enable_async_checkpoint=True
          )
      )
  ]

  tensorboard = TensorBoard(log_dir=f'logs/{EXPERIMENT_NAME}')
  history = model.fit(
      train_ds,
      initial_epoch=args.initial_epoch,
      epochs=args.epochs,
      callbacks=[tensorboard, save_ckpt],
      validation_data=test_ds,
  )

  model.save_weights(f"{EXPERIMENT_NAME}_" + f"{args.epochs}")


def prepare_dataset():
  AUTO = tf.data.experimental.AUTOTUNE
  test_ds = dataset_generator('/dataset/CIFAR10/test/')
  train_ds = dataset_generator('/dataset/CIFAR10/train/').cache()

  normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
  augment_layer = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    ])

  def normalize_and_augment(image, training):
    image = augment_layer(image, training=training)
    return image

  train_ds = (
    train_ds.shuffle(50000, reshuffle_each_iteration=True)
            .map(lambda x, y: (normalize_and_augment(x, training=True), y), num_parallel_calls=AUTO)
            .map(lambda x, _: (x, x))
            .prefetch(AUTO)
  )
  test_ds = (
    test_ds.map(lambda x, y: (normalize(x), y))
           .map(lambda x, _: (x, x))
           .cache()
           .prefetch(AUTO)
  )

  return train_ds, test_ds


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_size', type=int, help='total number of complex symbols sent'),
  parser.add_argument('channel_types', type=str, help='channel types. Rayleigh, AWGN, or None'),
  parser.add_argument('train_snrdB', type=int, help='train snr (in dB)'),
  parser.add_argument('block_types', type=str, help='Type of the each block (in a String, total 7). C for conv, V for ViT e.g,. CCVVVCC')
  parser.add_argument('experiment_name',
                      type=str,
                      help='experiment name (used for ckpt & logs)')
  parser.add_argument('epochs', type=int, help='total epochs')

  parser.add_argument('--filters', nargs='+', help='number of output dimensions for each block', required=True, type=int)
  parser.add_argument('--repetitions', nargs='+', help='repetitions of each block', required=True, type=int)

  parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
  parser.add_argument('--dim_per_head', type=int, default=32, help='dimensions per head in ViT blocks')
  parser.add_argument('--papr_reduction', type=str, default=None, help='papr reduction method ("clip" or None)')
  parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path (optional)')

  args = parser.parse_args()
  main(args)
