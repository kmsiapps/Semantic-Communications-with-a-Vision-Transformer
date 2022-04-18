import tensorflow as tf
import numpy as np
import socket

import random

from config import BATCH_SIZE
from models.model_large_todo import E2E_Decoder_Network, E2E_Encoder
from utils.datasets import dataset_generator

# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

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


test_ds = dataset_generator('/dataset/CIFAR10/test/')

loss_object = tf.keras.losses.MeanSquaredError()
test_loss = tf.keras.metrics.Mean(name='test_loss')

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

################## CONFIG ####################
best_model = './model_checkpoints/ae_non_selfattention_mae_25dB.ckpt'
##############################################

encoder_network = E2E_Encoder(filters=[32, 64, 128])
decoder_network = E2E_Decoder_Network(filters=[32, 64, 128])

encoder_network.load_weights(best_model)
decoder_network.load_weights(best_model)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = ("192.168.0.10", 60000)

rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = ("0.0.0.0", 50000)
rcv_sock.bind(rcv_addr)

rcv_sock.settimeout(0.1)

SOCK_BUFF_SIZE = 1024
ARTIFICIAL_DATA_LOSS_PROB = 0

for images, _ in test_ds:
    # send/receive original image

    send_data = tf.cast(images * 255, tf.uint8).numpy().tobytes()
    for i in range(0, len(send_data), SOCK_BUFF_SIZE):
        _data = send_data[i:min(len(send_data), i + SOCK_BUFF_SIZE)]
        send_sock.sendto(_data, send_addr)
    print(f'ORIGIN SEND DONE. len: {len(send_data)}')
    rcv_data = bytes()
    _data = bytes(SOCK_BUFF_SIZE)
    is_lossy = False
    while True:
        try:
            _data_bkup = _data
            _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
            if random.random() < ARTIFICIAL_DATA_LOSS_PROB:
                _data = _data_bkup
                print('Loss (SIM)', end='')
        except socket.timeout:
            print('Loss! ', end='')
            is_lossy = True
            # use last data instead
        rcv_data += _data
        print(f'rcv {len(rcv_data)}... ', end='')
        if len(rcv_data) >= len(send_data):
            break
    rcv_data = tf.cast(tf.convert_to_tensor(np.frombuffer(rcv_data, np.uint8)), tf.float32) / 255
    original_result = tf.reshape(rcv_data, images.shape)
    print(f'[{is_lossy}] ORIGIN RCV DONE')

    # send/receive proposed image
    encoded_image = encoder_network(images)
    send_data = tf.cast(encoded_image * 255, tf.uint8).numpy().tobytes()
    for i in range(0, len(send_data), SOCK_BUFF_SIZE):
        _data = send_data[i:min(len(send_data), i + SOCK_BUFF_SIZE)]
        send_sock.sendto(_data, send_addr)
    print(f'PROPOSED SEND DONE. len: {len(send_data)}')
    rcv_data = bytes()
    _data = bytes(SOCK_BUFF_SIZE)
    is_lossy = False
    while True:
        try:
            _data_bkup = _data
            _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
            if random.random() < ARTIFICIAL_DATA_LOSS_PROB:
                _data = _data_bkup
                print('Loss (SIM)', end='')
        except socket.timeout:
            print('Loss! ', end='')
            is_lossy = True
            # use last data instead
        rcv_data += _data
        print(f'rcv {len(rcv_data)}... ', end='')
        if len(rcv_data) >= len(send_data):
            break
    rcv_data = tf.cast(tf.convert_to_tensor(np.frombuffer(rcv_data, np.uint8)), tf.float32) / 255
    proposed_result = decoder_network(tf.reshape(rcv_data, images.shape))
    print(f'[{is_lossy}] PROPOSED RCV DONE')

    # Compare SSIM
    origin_ssim = tf.reduce_mean(tf.image.ssim(images, original_result, max_val=1.0))
    prop_ssim = tf.reduce_mean(tf.image.ssim(images, proposed_result, max_val=1.0))
    
    # Compare PSNR
    origin_psnr = tf.reduce_mean(tf.image.psnr(images, original_result, max_val=1.0))
    prop_psnr = tf.reduce_mean(tf.image.psnr(images, proposed_result, max_val=1.0))

    print(f'ORIGINAL: (PSNR) {origin_psnr} (SSIM) {origin_ssim:.4f}\n')
    print(f'PROPOSED: (PSNR) {prop_psnr} (SSIM) {prop_ssim:.4f}\n')

    tf.keras.utils.save_img(f'./usrp_source.png', imBatchtoImage(images))
    tf.keras.utils.save_img(f'./usrp_original.png', imBatchtoImage(original_result))
    tf.keras.utils.save_img(f'./usrp_proposed.png', imBatchtoImage(proposed_result))
    
    input("Press ENTER key to continue")

send_sock.close()
