# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
import argparse
import csv

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.model import SemViT
from utils.datasets import dataset_generator

ckpt_dir = './bkup_ckpt/best/awgn/'
test_SNRs = [0, 2, 5, 7, 10, 12, 15]
test_ckpts = list(set([name.split('.')[0] for name in os.listdir(ckpt_dir)]))
# get checkpoint name from given directory (without extensions)

test_ckpts.sort()

def main(args):
	if args.gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	# Load CIFAR-10 dataset
	test_ds = prepare_dataset()
	
	def ssim(y_true, y_pred):
		return tf.image.ssim(y_true, y_pred, max_val=1)

	f = open('ssim.csv', 'w', newline='')
	csv_writer = csv.writer(f)

	csv_writer.writerow(['name', 'arch', '# of symbols', 'train SNR(dB)'] + test_SNRs)
	
	for idx, ckpt_name in enumerate(test_ckpts):
		print(f'Running {ckpt_name} ({idx+1:>03}/{len(test_ckpts):>03}): ')
		arch, num_symbols, snr_trained, *_ = ckpt_name.split('_')
		has_gdn = (arch == 'CCCCCC') or '_GDN_' in ckpt_name

		ssims = []
		
		for test_snrdB in test_SNRs:
			print(f'{test_snrdB}dB...', end='')
			model = SemViT(
				arch,
				[256, 256, 256, 256, 256, 256],
				[1, 1, 3, 3, 1, 1],
				has_gdn=has_gdn,
				num_symbols=int(num_symbols),
				snrdB=test_snrdB,
				channel='Rayleigh'
			)
			model.load_weights(f'{ckpt_dir}/{ckpt_name}').expect_partial()

			i = 0
			avg_ssim = 0
			for image, _ in test_ds:
				decoded_image = model(image)
				avg_ssim += tf.reduce_mean(ssim(image, decoded_image))
				i += 1
			avg_ssim = avg_ssim / i
			ssims.append(float(avg_ssim))

		print()
		csv_writer.writerow([ckpt_name, arch, num_symbols, int(snr_trained[:-2])] + ssims)

	f.close()

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
