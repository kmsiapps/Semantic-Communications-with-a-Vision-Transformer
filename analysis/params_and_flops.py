# %%
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.model import SemViT
from utils.datasets import dataset_generator

# NOTE: to calculate FLOPs with ViTs, disable positional embeddings
#       - currently tensorflow profiler cannot return FLOPs with them

def main(args):
  if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

  EXPERIMENT_NAME = args.experiment_name
  print(f'Running {EXPERIMENT_NAME}')

  model = SemViT(
    args.block_types,
    args.filters,
    args.repetitions,
    has_gdn=args.gdn,
    num_symbols=args.data_size,
    snrdB=args.train_snrdB,
    channel=args.channel_types
  )

  model.build(input_shape=(1, 32, 32, 3))
  model.summary()

  print(f'FLOPS: {get_flops(model):.2f}GFLOPs')


def get_flops(model) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    source: https://github.com/wandb/wandb/blob/latest/wandb/integration/keras/keras.py#L1025-L1073
    """
    if not isinstance(
        model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = tf.random.normal((1, 32, 32, 3))

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2


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
  parser.add_argument('--gdn', type=lambda x: True if x.lower() == 'true' else False, default=True, help='(true/false) use gdn/igdn')

  parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
  parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path (optional)')

  parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')

  args = parser.parse_args()
  main(args)

# python3 ./params_and_flops.py 512 AWGN 10 CCCCCC CCCCCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0