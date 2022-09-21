# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from tensorflow.python.keras import backend as K

import argparse
import csv

from models.model_debug import SemViT_Debug
from utils.datasets import dataset_generator

ckpt_name = './bkup_ckpt/best/CCVVCC_512_10dB_599'

def main(args):
	if args.gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	
	# Load CIFAR-10 dataset
	test_ds = prepare_dataset()
	
	def psnr(y_true, y_pred):
		return tf.image.psnr(y_true, y_pred, max_val=1)

	print(f'Running {ckpt_name}: ')
	arch, num_symbols, snr_trained, *_ = ckpt_name.split('_')
	has_gdn = (arch == 'CCCCCC') or '_GDN_' in ckpt_name

	model = SemViT_Debug(
		arch,
		[256, 256, 256, 256, 256, 256],
		[1, 1, 3, 3, 1, 1],
		has_gdn=has_gdn,
		num_symbols=int(num_symbols),
		snrdB=int(snr_trained[:-2]),
		channel='AWGN'
	)
	model.load_weights(ckpt_name).expect_partial()

	# CALC ATTMAPS, POSEMB
	attmaps = tf.zeros((2, 3, 64, 64))

	i = 0
	for image, _ in test_ds:
		# UGLY SOLUTION: MAKE CUSTOM MODEL THAT RETURNS EVERY LAYER OUTPUT
		pred, m, att, pe = model(image)
		attmaps = attmaps + att
		
		i += 1

	# Note: PE does not change over images; just use last one
	pe = tf.convert_to_tensor(pe)

	# PSNR to validate if models are well trained
	test_psnr = tf.reduce_mean(psnr(image, pred))
	print(f'PSNR:{float(test_psnr):.2f}', )
	
	# TODO: save pe, attmaps

	# PE shape: (2, 3, 15, 15)
	# Attmaps shape: (2, 3, 64, 64)
	# note: shape is (ENC/DEC, RepetitionInStage, H, W)


def prepare_dataset():
	AUTO = tf.data.experimental.AUTOTUNE
	test_ds = dataset_generator('/dataset/CIFAR10/test/')

	normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

	test_ds = (
	test_ds.map(lambda x, y: (normalize(x), y))
			 .map(lambda x, _: (x, x))
			 .cache()
			 .prefetch(AUTO)
	)

	return test_ds


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')

	args = parser.parse_args()
	main(args)
