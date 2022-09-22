# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
import csv

from models.model_debug import SemViT_Debug
from utils.datasets import dataset_generator

ckpt_dir = './'
ckpt_name = 'CCVVCC_512_10dB_599'


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
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')
	# args = parser.parse_args()
	# if args and args.gpu:
	# 	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	
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
	model.load_weights(f'{ckpt_dir}/{ckpt_name}').expect_partial()

	# CALC ATTMAPS, POSEMB
	attmaps = tf.zeros((6, 1, 64, 64))
	cossims = tf.zeros((6,))

	i = 0
	for image, _ in test_ds:
		# UGLY SOLUTION: MAKE CUSTOM MODEL THAT RETURNS EVERY LAYER OUTPUT
		pred, m, att, pe, cs = model(image)
		attmaps = attmaps + tf.convert_to_tensor(att)
		cossims = cossims + tf.convert_to_tensor(cs)
		
		i += 1

	# Note: PE does not change over images; just use last one
	pe = tf.convert_to_tensor(pe)
	cossims = cossims / i

	# PSNR to validate if models are well trained
	test_psnr = tf.reduce_mean(psnr(image, pred))
	print(f'PSNR:{float(test_psnr):.2f}', )
	
	# TODO: save pe, attmaps

	# PE shape: (6, 1, 225)
	# Attmaps shape: (6, 1, 64, 64)
	# note: shape is (layers, dummy, H, W)

#%%

print(cossims)

# %%
for i in range(6):
	plt.imshow(attmaps[i, 0, :, :])
	plt.clim(0, 0.15)
	plt.colorbar()
	plt.show()

# %%
# encoder/decoder avg att map & pe

# Encoder-part avg
encoder_att_map = tf.reduce_mean(attmaps[:3, 0, :, :], axis=0)
plt.imshow(encoder_att_map)
plt.title('Encoder avg attmap')
plt.colorbar()
plt.clim(0, 0.12)
plt.show()

# Decoder-part avg
decoder_att_map = tf.reduce_mean(attmaps[3:, 0, :, :], axis=0)
plt.imshow(decoder_att_map)
plt.title('Decoder avg attmap')
plt.colorbar()
plt.clim(0, 0.12)
plt.show()

encoder_pe = tf.reshape(tf.reduce_mean(pe[:3, 0, :], axis=0), (15, 15))
plt.imshow(encoder_pe)
plt.title('Encoder avg pos. emb.')
plt.colorbar()
plt.clim(-1, 2)
plt.show()

decoder_pe = tf.reshape(tf.reduce_mean(pe[3:, 0, :], axis=0), (15, 15))
plt.imshow(decoder_pe)
plt.title('Decoder avg pos. emb.')
plt.colorbar()
plt.clim(-1, 2)
plt.show()

target_idx = 36

plt.imshow(tf.reshape(encoder_att_map[target_idx, :], (8, 8)))
plt.title(f'encoder att map: at index ({target_idx//8},{target_idx%8})')
plt.colorbar()
plt.clim(0.09, -0.02)
plt.show()

plt.imshow(tf.reshape(decoder_att_map[target_idx, :], (8, 8)))
plt.title(f'decoder att map: at index ({target_idx//8},{target_idx%8})')
plt.colorbar()
plt.clim(0.09, -0.02)
plt.show()

# %%

# layer-wise att map
target_idx = 36

for target_layer in range(6):
	plt.title(f'l{target_layer//3+2}-{target_layer%3} att map: at index ({target_idx//8},{target_idx%8})')
	plt.imshow(tf.reshape(attmaps[target_layer, 0, target_idx, :], (8, 8)))
	plt.colorbar()
	plt.clim(0, 0.1)
	plt.show()

# %%
