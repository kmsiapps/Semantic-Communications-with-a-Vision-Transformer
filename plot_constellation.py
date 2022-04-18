# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models.qam_model import QAMModulator

import tensorflow as tf
import os
import csv

from config import FILTERS, NUM_BLOCKS, DIM_PER_HEAD, DATA_SIZE, BATCH_SIZE
from models.model import VitCommNet, VitCommNet_Encoder_Only
from models.qam_model import QAMModem
from utils.datasets import dataset_generator

test_ds = dataset_generator('/dataset/CIFAR100/test/')
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

################## CONFIG ####################
best_model = './MAE_0.047384.ckpt'
##############################################

# Layer-wise image
images, _ = next(iter(test_ds))

model = VitCommNet_Encoder_Only(
    FILTERS,
    NUM_BLOCKS,
    DIM_PER_HEAD,
    DATA_SIZE,
    snrdB=10,
    channel='Rayleigh'
    )
model.load_weights(best_model)

constellation = model(images)
constellation = tf.reshape(constellation, (2, -1)).numpy()

'''
QAM_ORDER = 256
data = tf.random.uniform([4096], 0, QAM_ORDER, dtype=tf.int32)
qammod = QAMModulator(order=QAM_ORDER)

constellation = qammod(data).numpy()
print(constellation.shape)
'''

i = constellation[0]
q = constellation[1]

power = np.sqrt(np.sum(i ** 2 + q ** 2) / len(constellation[0]))
print(power)

i = i / power
q = q / power

papr = np.max(i ** 2 + q ** 2) / np.mean(i ** 2 + q ** 2)
print(papr)

plt.scatter(i[:], q[:], s=0.1)
plt.show()

# %%
