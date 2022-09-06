# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import argparse

from models.model import SemViT
from models.deepjscc import DeepJSCC
from utils.datasets import dataset_generator

def main(args):
  if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

  EXPERIMENT_NAME = args.experiment_name
  print(f'Running {EXPERIMENT_NAME}')

  # model = DeepJSCC()

  model = SemViT(
    args.block_types,
    args.filters,
    args.repetitions,
    args.dim_per_head,
    args.data_size,
    snrdB=args.train_snrdB,
    channel=args.channel_types
  )

  model.build(input_shape=(None, 32, 32, 3))
  model.summary()
  


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

  parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')

  args = parser.parse_args()
  main(args)

# python3 ./summary.py 512 AWGN 10 CCCCCC DeepJSCC_test_10dB 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0