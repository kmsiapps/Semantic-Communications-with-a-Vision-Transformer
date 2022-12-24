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

ckpt_dir = './ckpt'
# test_ckpts = list(set([name.split('.')[0] for name in os.listdir(ckpt_dir)]))

test_ckpts = [
	'CCVVCC_512_10dB_599'
	# 'CCCCCC_512_10dB_597'
]

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
image = next(iter(test_ds))[0]

# import PIL.Image as pilimg

# image = pilimg.open('/dataset/test.png').convert('RGB')
# image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
# h, w, c = image.shape
# image = tf.reshape(image, (1, h, w, c))

# image = tf.image.extract_patches(
#     image,
#     sizes=[1, 32, 32, 1],
#     strides=[1, 32, 32, 1],
#     rates=[1, 1, 1, 1],
#     padding='VALID'
# )
# image = tf.reshape(image, (-1, 32, 32, c))

def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=1)

#%%
ckpt_name = test_ckpts[0]
arch, num_symbols, snr_trained, *_ = ckpt_name.split('_')
has_gdn = (arch == 'CCCCCC') or '_GDN_' in ckpt_name

model = SemViT_Debug(
  arch,
  [256, 256, 256, 256, 256, 256],
  [1, 1, 3, 3, 1, 1],
  has_gdn=has_gdn,
  num_symbols=int(num_symbols),
  snrdB=int(snr_trained[:-2]),
  channel=None
)
model.load_weights(f'{ckpt_dir}/{ckpt_name}').expect_partial()

# UGLY SOLUTION: MAKE CUSTOM MODEL THAT RETURNS EVERY LAYER OUTPUT
pred, m, _, _, _ = model(image)

# m: [enc_input, l0, l1, l2-0, l2-1, l2-2, enc_proj, dec_input, l3-0, l3-1, l3-2, resize, l4, resize, l5, proj]
# remove resizing/projection layers
l2 = m[5]
symbols = m[6]

def psnr(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1)

# PSNR to validate if models are well trained
test_psnr = tf.reduce_mean(psnr(image, pred))
print(f'PSNR:{float(test_psnr):.2f}')

# %%
@tf.function
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

# %%
import matplotlib.pyplot as plt
plt.subplot(1, 3, 1)
plt.imshow(imBatchtoImage(image))
plt.colorbar()
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(imBatchtoImage(tf.reduce_mean(l2, axis=-1, keepdims=True)), cmap='gray')
plt.colorbar()
plt.title('Mean')

plt.subplot(1, 3, 3)
plt.imshow(imBatchtoImage(tf.math.reduce_std(l2, axis=-1, keepdims=True)), cmap='gray')
plt.colorbar()
plt.title('std')

plt.tight_layout()
plt.savefig('ViT_structure.png')
# %%
