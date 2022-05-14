#%%

import socket
import struct
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import math
import tensorflow as tf

from pilot import p_start, p_end, PILOT_SIZE, SAMPLE_SIZE
from models.model import VitCommNet_Encoder_Only, VitCommNet_Decoder_Only
from config import FILTERS, NUM_BLOCKS, DIM_PER_HEAD, DATA_SIZE, BATCH_SIZE

rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = ("0.0.0.0", 50000)
rcv_sock.bind(rcv_addr)

rcv_sock.settimeout(2)

EXPECTED_DATA_LENGTH = 4096 * 2
SOCK_BUFF_SIZE = 16384 # EXPECTED_DATA_LENGTH * 4 + 4

EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - PILOT_SIZE * 2

images = pilimg.open('./results/source.png').convert('RGB')
images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0
h, w, c = images.shape
images = tf.reshape(images, (1, h, w, c))

images = tf.image.extract_patches(
    images,
    sizes=[1, 32, 32, 1],
    strides=[1, 32, 32, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)
images = tf.reshape(images, (-1, 32, 32, c))

encoder_network = VitCommNet_Encoder_Only(
FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  512,
  snrdB=25,
  channel='Rayleigh'
)

decoder_network = VitCommNet_Decoder_Only(
    FILTERS,
    NUM_BLOCKS,
    DIM_PER_HEAD,
    512,
    snrdB=25,
    channel='Rayleigh'
)

# best_model = './model_checkpoints/data-512.ckpt' #'./model_checkpoints/data-512.ckpt'
encoder_network.load_weights('./model_checkpoints/encoder-512.ckpt')
decoder_network.load_weights('./model_checkpoints/decoder-512.ckpt')

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

gt = encoder_network(images)
gt_i = tf.reshape(gt[0], shape=(-1,))
gt_q = tf.reshape(gt[1], shape=(-1,))

# while True:
#     try:
#         _, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
#     except socket.timeout:
#         break

#%%

rcv_data = bytes()
_data = bytes()

read_header = False
while True:
    rcv = True
    try:
        _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
    except socket.timeout:
        print('.', end='')
        rcv = False
    if rcv:
        # print()
        if not read_header:
            array_length = int.from_bytes(_data[:4], byteorder='big')
            ARRAY_END = array_length * 4 + 4
            read_header = True
        rcv_data += _data
        # print(f"RCV: {len(rcv_data)}/{ARRAY_END} (+{len(_data)})", end='')
        
        if len(rcv_data) >= ARRAY_END:
            data = rcv_data[4:ARRAY_END]
            d_iq = struct.unpack('!' + 'f' * array_length, data)

            raw_i = np.array(d_iq[:array_length // 2])
            raw_q = np.array(d_iq[array_length // 2:])

            # Leakage compensation
            LC = 1 # Leakage ratio constant
            i_compensated = (raw_i - LC * raw_q) / (1+LC**2)
            q_compensated = (raw_q + LC * raw_i) / (1+LC**2) / 3.5 # magic num

            print()
            print(f"{len(rcv_data)} => ", end='')
            rcv_data = rcv_data[ARRAY_END:]
            print(f"{len(rcv_data)}")
            read_header = False

            pilot_mask = np.concatenate([p_start, np.zeros(EXPECTED_SAMPLE_SIZE), p_end])
            start_idx = np.argmax(np.abs(np.correlate(i_compensated, pilot_mask))) + PILOT_SIZE

            # get noise & zero-mean normalize
            noises = np.concatenate(
                [i_compensated[:start_idx-PILOT_SIZE],
                i_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
            )
            n = np.mean(noises)
            i_compensated -= n
            noise_power = np.sum(noises ** 2) / len(noises)

            # get average h
            p_start_rx = i_compensated[(start_idx - PILOT_SIZE):start_idx]
            p_end_rx = i_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
            p_rx = np.concatenate([p_start_rx, p_end_rx])
            p = np.concatenate([p_start, p_end]) / 32767
            nonzero_idx = np.where(p != 0)
            hi = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])
            hi *= 1.15 # magic num

            i_compensated /= hi

            # get data
            ihat = i_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]
            signal_power = np.sum((ihat * hi) ** 2) / len(ihat)

            plt.title('Decoded In-phase signal\n(h & n compensated)')
            plt.plot(ihat)
            plt.show()

            # Quadrature phase detection =================
            pilot_mask = np.concatenate([p_start, np.zeros(EXPECTED_SAMPLE_SIZE), p_end])
            start_idx = np.argmax(np.abs(np.correlate(q_compensated, pilot_mask))) + PILOT_SIZE

            # get noise & zero-mean normalize
            noises = np.concatenate(
                [q_compensated[:start_idx-PILOT_SIZE],
                q_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
            )
            n = np.mean(noises)
            q_compensated -= n
            noise_power = np.sum(noises ** 2) / len(noises)

            # get average h
            p_start_rx = q_compensated[(start_idx - PILOT_SIZE):start_idx]
            p_end_rx = q_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
            p_rx = np.concatenate([p_start_rx, p_end_rx])
            p = np.concatenate([p_start, p_end]) / 32767
            nonzero_idx = np.where(p != 0)
            hq = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])
            hq *= 1.15 # magic num

            q_compensated /= hq

            # get data
            qhat = q_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]

            plt.title('Decoded Quadrature-phase signal\n(h & n compensated)')
            plt.plot(qhat)
            plt.show()

            plt.title('Raw I/Q')
            plt.plot(raw_i)# [start_idx-PILOT_SIZE*2:start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE*2])
            plt.plot(raw_q)# [start_idx-PILOT_SIZE*2:start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE*2])
            plt.show()

            max_i = 80 # float(input('max_i?'))
            max_q = 80 # float(input('max_q?'))

            plt.title('Received Constallations')
            plt.scatter(ihat, qhat, s=0.01)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.show()

            rcv_iq = np.zeros(shape=(2, len(ihat)))
            rcv_iq[0, :] = ihat * max_i
            rcv_iq[1, :] = qhat * max_q
            
            rcv_iq = tf.cast(tf.convert_to_tensor(rcv_iq), tf.float32)
            rcv_iq = tf.reshape(rcv_iq, (2, 64, -1))

            proposed_result = decoder_network(rcv_iq)
            tf.keras.utils.save_img(f'./results/proposed_usrp.png', imBatchtoImage(proposed_result))

            non_error_result = decoder_network(gt)
            tf.keras.utils.save_img(f'./results/proposed_gt.png', imBatchtoImage(non_error_result))

            err_i = gt_i/max_i - ihat
            err_q = gt_q/max_q - qhat

            plt.title('Error')
            plt.plot(err_i)
            plt.plot(err_q)
            plt.show()

            signal_power = np.mean(ihat ** 2 + qhat ** 2)
            noise_power = np.mean(err_i ** 2 + err_q ** 2)
            snr = signal_power / noise_power
            snrdB = 10 * math.log10(snr)

            print(f"Effective SNR: {snrdB:.2f}dB")

            psnr = np.mean(tf.image.psnr(images, proposed_result, max_val=1.0))
            print(f"Image PSNR (averaged over patches): {psnr:.2f} dB")



# i = decoded_data[0::2]
# q = decoded_data[1::2]

plt.plot(range(len(decoded_data)), decoded_data)
plt.show()

# %%
