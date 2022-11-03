# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
from tensorflow.python.keras import backend as K
from config import BATCH_SIZE

import argparse
import csv

from models.model_debug import SemViT_Debug
from utils.datasets import dataset_generator

# ckpt_dir = './ckpt_timeseries'
# test_ckpts = list(set([name.split('.')[0] for name in os.listdir(ckpt_dir)]))
ckpt_dir = './ckpt'
test_ckpt = 'CCCCCC_512_10dB_597'

# get checkpoint name from given directory (without extensions)

def get_fourier_diagonal_amp(x):
	'''
	returns amplitude of diagonal elements of 2d-ffted x
	x: tensor with shape (B, H, W)
	'''
	b, h, w = tf.shape(x)
	assert h == w, 'h should be equal to w'
	x = tf.complex(x, 0.0)

	fft_amp = tf.math.abs(tf.signal.fft2d(x))

	# fft: b, h, w
	fourier_diagonal = tf.reduce_sum(fft_amp * tf.expand_dims(tf.eye(h), axis=0), axis=-1)
	return tf.reduce_mean(fourier_diagonal, axis=0)


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


# Load CIFAR-10 dataset
test_ds = prepare_dataset()

def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=1)

print(f'Running {test_ckpt}: ')
arch, num_symbols, snr_trained, *_ = test_ckpt.split('_')
has_gdn = (arch == 'CCCCCC') or '_GDN_' in test_ckpt

model = SemViT_Debug(
	arch,
	[256, 256, 256, 256, 256, 256],
	[1, 1, 3, 3, 1, 1],
	has_gdn=has_gdn,
	num_symbols=int(num_symbols),
	snrdB=int(snr_trained[:-2]),
	channel='AWGN'
)
model.load_weights(f'{ckpt_dir}/{test_ckpt}').expect_partial()

# CALC COSSIM
i = 0
avg_fourier_diagonal = [0 for _ in range(11)]
average_image_fourier_diag = tf.zeros((32, 32))
for image, _ in test_ds:
	# UGLY SOLUTION: MAKE CUSTOM MODEL THAT RETURNS EVERY LAYER OUTPUT
	pred, m, *_ = model(image)
	# m: [enc_input, l0, l1, l2-0, l2-1, l2-2, enc_proj, dec_input, l3-0, l3-1, l3-2, resize, l4, resize, l5, proj]
	# remove resizing/projection layers
	intermediate_outputs = m[1:7] + m[8:11] + [m[12]] + [m[13]]

	for idx, x in enumerate(intermediate_outputs):
		avg_fourier_diagonal[idx] += get_fourier_diagonal_amp(tf.reduce_mean(x, axis=-1))
	
	average_image_fourier_diag += get_fourier_diagonal_amp(tf.reduce_mean(image, axis=-1))
	i += 1

for idx, x in enumerate(avg_fourier_diagonal):
	avg_fourier_diagonal[idx] = avg_fourier_diagonal[idx] / i

average_image_fourier_diag /= i

def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=1)

# PSNR to validate if models are well trained
test_psnr = tf.reduce_mean(psnr(image, pred))
print(f'PSNR:{float(test_psnr):.2f}')

f.close()

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc
	
# %%

# target: [l1, l2-0, l2-1, l2-2]
# m: [l0, l1, l2-0, l2-1, l2-2, enc_proj, l3-0, l3-1, l3-2, l4, l5]
target = avg_fourier_diagonal[1:5]
yint = np.arange(0, len(target), 1)
xs = [tf.linspace(0, 1, len(x)//2+1) for x in target]
ys = [tf.math.log(x[:len(x)//2+1]) - tf.math.log(x[0]) for x in target]
fig, ax = plt.subplots()
lc = multiline(xs, ys, yint, cmap='summer', lw=2)
axcb = fig.colorbar(lc)
plt.title('L2 - frequency')
plt.show()

# %%

# target: [enc_proj, l3-0, l3-1, l3-2]
# m: [l0, l1, l2-0, l2-1, l2-2, enc_proj, l3-0, l3-1, l3-2, l4, l5]
target = avg_fourier_diagonal[5:9]
yint = np.arange(0, len(target), 1)
xs = [tf.linspace(0, 1, len(x)//2+1) for x in target]
ys = [tf.math.log(x[:len(x)//2+1]) - tf.math.log(x[0]) for x in target]
fig, ax = plt.subplots()
lc = multiline(xs, ys, yint, cmap='summer', lw=2)
axcb = fig.colorbar(lc)
plt.title('L3 - frequency')
plt.show()

# %%

# m: [l0, l1, l2-0, l2-1, l2-2, enc_proj, l3-0, l3-1, l3-2, l4, l5]
target = [get_fourier_diagonal_amp(tf.expand_dims(average_image_fourier_diag, axis=0))] + avg_fourier_diagonal[:2] + [avg_fourier_diagonal[4]] + avg_fourier_diagonal[8:]
yint = np.arange(0, len(target), 1)
xs = [tf.linspace(0, 1, len(x)//2+1) for x in target]
ys = [tf.math.log(x[:len(x)//2+1]) - tf.math.log(x[0]) for x in target]
fig, ax = plt.subplots()
lc = multiline(xs, ys, yint, cmap='summer', lw=2)
axcb = fig.colorbar(lc)
plt.title('Overall - frequency')
plt.show()

# %%
