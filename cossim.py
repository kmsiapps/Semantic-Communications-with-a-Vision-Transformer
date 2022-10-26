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

# ckpt_dir = './ckpt_timeseries'
# test_ckpts = list(set([name.split('.')[0] for name in os.listdir(ckpt_dir)]))
ckpt_dir = './'
test_ckpts = ['CCCCCC_512_10dB_597']

# get checkpoint name from given directory (without extensions)

test_ckpts.sort()

def get_avg_cossim(x):
	'''
	Get average cosine similarity along spatial domain, except for self-similarity
	x: tensor with shape (B, H, W, C)
	'''
	b, h, w, c = tf.shape(x)
	assert h == w, 'h should be equal to w'
	
	x1 = tf.reshape(x, (-1, h*w, c))
	x2 = tf.reshape(x, (-1, h*w, c))

	cossim = tf.einsum('bic,bjc->bij', x1, x2)
	normalizer = tf.norm(x1, axis=-1, keepdims=True) * tf.reshape(tf.norm(x2, axis=-1), (-1, 1, h*w))
	cossim = cossim / normalizer

	# remove diagonal elements
	cossim = tf.linalg.set_diag(cossim, tf.zeros(cossim.shape[0:-1]))
	avg_cossim = tf.reduce_sum(cossim) / tf.cast(b * (h*w*h*w - h*w), dtype=tf.float32)

	return avg_cossim


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


# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')

# 	args = parser.parse_args()
# 	main(args)

# if args.gpu:
	# 	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	
# Load CIFAR-10 dataset
test_ds = prepare_dataset()

def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=1)

f = open('cossim.csv', 'w', newline='')
csv_writer = csv.writer(f)

csv_writer.writerow(['name', 'arch', '# of symbols', 'train SNR(dB)',
										 'l0', 'l1', 'l2-0', 'l2-1', 'l2-2',
										 'channel', 'l3-0', 'l3-1', 'l3-2', 'l4', 'l5'])

for idx, ckpt_name in enumerate(test_ckpts):
	print(f'Running {ckpt_name} ({idx+1:>03}/{len(test_ckpts):>03}): ')
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
	model.load_weights(f'{ckpt_dir}/{ckpt_name}').expect_partial()

	# CALC COSSIM
	i = 0
	avg_cossim = [0 for _ in range(11)]
	cossims = tf.zeros((6,))
	for image, _ in test_ds:
		# UGLY SOLUTION: MAKE CUSTOM MODEL THAT RETURNS EVERY LAYER OUTPUT
		pred, m, _, _, cs = model(image)
		cossims += tf.convert_to_tensor(cs)

		# m: [enc_input, l0, l1, l2, enc_proj, dec_input, l3, resize, l4, resize, l5, proj]

		# m: [enc_input, l0, l1, l2-0, l2-1, l2-2, enc_proj, dec_input, l3-0, l3-1, l3-2, resize, l4, resize, l5, proj]
		# remove resizing/projection layers
		intermediate_outputs = m[1:7] + m[8:11] + [m[12]] + [m[13]]

		for idx, x in enumerate(intermediate_outputs):
			avg_cossim[idx] += float(get_avg_cossim(x))
		i += 1
	
	cossims /= i
	print(cossims)
	
	def psnr(y_true, y_pred):
		return tf.image.psnr(y_true, y_pred, max_val=1)

	# PSNR to validate if models are well trained
	test_psnr = tf.reduce_mean(psnr(image, pred))
	avg_cossim = [j / i for j in avg_cossim]

	print(f'PSNR:{float(test_psnr):.2f}', avg_cossim)
	csv_writer.writerow([ckpt_name, arch, num_symbols, int(snr_trained[:-2])] + avg_cossim)

f.close()
# %%
