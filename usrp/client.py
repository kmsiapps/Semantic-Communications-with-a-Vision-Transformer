#%%
import socket
import cv2
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import multiprocessing
import time
import struct

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.networking import receive_and_save_binary, send_binary, send_constellation_udp
from utils.usrp_utils import get_lci_lcq_compensation, compensate_signal, rcv_worker
from config.usrp_config import USRP_HOST, USRP_PORT, RCV_ADDR, RCV_PORT, TEMP_DIRECTORY

TARGET_JPEG_RATE = 2048
# Our encoder produces 512 constellations per 32 x 32 patch
# so for 256 * 256 image,
# 512 * 8 * 8 = 32768

# For comparison, we assume JPEG, QPSK, 1/4 coding rates
# (LTE MCS selection, -1.9 ~ 1.8 dB SNR)
# so desired JPEG-encoded file size is
# 32768 / 4 / 4 (=4 QPSK symbols/bytes) ~ 2048 Bytes

if __name__ == '__main__':
    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    BUFF_SIZE = 4096
    clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSock.connect(('1.233.219.33', 8080))
    print('Connected')

    # Construct sockets
    rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rcv_addr = (RCV_ADDR, RCV_PORT)
    rcv_sock.bind(rcv_addr)
    # rcv_sock.settimeout(2)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_addr = (USRP_HOST, USRP_PORT)

    demo_type = int(input("Select image type: CIFAR (0) vs CELEB (1) vs CAM (2)?: "))
    use_cache = False

    if demo_type == 0:
        TARGET_IMAGE = 'cifar.png'
        use_cache = (int(input("Use cached constellation (0/1)? ")) == 1)
    elif demo_type == 1:
        TARGET_IMAGE = 'celeb.png'
    else:
    # Webcam capture
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        while True:
            check, frame = cam.read()
            cv2.imshow('video', frame)
            key = cv2.waitKey(1)
            if key == 27:
                image = frame[:, 100:-100, :]
                image = cv2.resize(image, (256, 256))
                cv2.imwrite('cam.png', image)
                break

        cam.release()
        cv2.destroyAllWindows()
        TARGET_IMAGE = 'cam.png'

    print('Image type:', demo_type, 'cache:', use_cache)

    # IQ compensations
    LCI, LCQ = get_lci_lcq_compensation(rcv_sock, rcv_addr, send_sock, send_addr)

    # Tell the server whether we will use cached one
    clientSock.send(int(use_cache).to_bytes(length=4, byteorder='big', signed=False))

    if use_cache:
        constellations = np.load(f'cifar_constellations.npz')['constellations']
    else:
        # Send image
        send_binary(clientSock, TARGET_IMAGE)

        # Receive constellations (from HOST)
        receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/constellations.npz')
        constellations = np.load(f'{TEMP_DIRECTORY}/constellations.npz')['constellations']

    # Send/receive constellations CONCURRENTLY (to USRP)
    manager = multiprocessing.Manager()
    ret_dict = manager.dict()
    p = multiprocessing.Process(target=rcv_worker, args=(rcv_sock, ret_dict))
    p.start()
    time.sleep(1)

    send_constellation_udp(constellations.tobytes(), send_sock, send_addr)

    p.join()
    data = ret_dict['return']
    rcv_iq, raw_i, raw_q = compensate_signal(data, LCI, LCQ)

    # Send channel corrupted (I/Q compensated) constellations (=rcv_iq)
    np.savez_compressed(f'{TEMP_DIRECTORY}/snd_iq.npz', rcv_iq=rcv_iq)
    send_binary(clientSock, f'{TEMP_DIRECTORY}/snd_iq.npz')

    # Receive decoded image
    receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/decoded.png')

    # receive effective SNR
    SNRdB = struct.unpack('!f', clientSock.recv(4))[0]
    print(f'Effective SNR: {SNRdB:.2f} dB')

    # TODO: plot
    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(7)
    fig.set_figwidth(21)

    ax[0].set_title('Original')
    original = pilimg.open(TARGET_IMAGE).convert('RGB')
    ax[0].imshow(original)

    psnr = 0 # np.mean(tf.image.psnr(images, proposed_result, max_val=1.0))
    ssim = 0 # np.mean(tf.image.ssim(images, proposed_result, max_val=1.0))
    ax[1].set_title(f'Proposed\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
    proposed = pilimg.open(f'{TEMP_DIRECTORY}/decoded.png').convert('RGB')
    ax[1].imshow(proposed)

    # JPEG rate-match algorithm
    quality_max = 100
    quality = 50
    quality_min = 0

    while True:
        original.save(f'{TEMP_DIRECTORY}/source_jpeg.jpg', quality=quality)
        bytes = os.path.getsize(f'{TEMP_DIRECTORY}/source_jpeg.jpg')
        if quality == 0 or quality == quality_min or quality == quality_max:
            break
        elif bytes > TARGET_JPEG_RATE and quality_min != quality - 1:
            quality_max = quality
            quality -= (quality - quality_min) // 2
        elif bytes > TARGET_JPEG_RATE and quality_min == quality - 1:
            quality_max = quality
            quality -= 1
        elif bytes < TARGET_JPEG_RATE and quality_max > quality:
            quality_min = quality
            quality += (quality_max - quality) // 2
        else:
            break
    print(f'JPEG file size: {bytes} Bytes, Quality: {quality}')

    image_jpeg = pilimg.open(f'{TEMP_DIRECTORY}/source_jpeg.jpg').convert('RGB')
    # image_jpeg = tf.convert_to_tensor(image_jpeg, dtype=tf.float32) / 255.0
    # h, w, c = image_jpeg.shape
    # image_jpeg = tf.reshape(image_jpeg, (1, h, w, c))
    # image_jpeg = tf.image.extract_patches(
    #     image_jpeg,
    #     sizes=[1, 32, 32, 1],
    #     strides=[1, 32, 32, 1],
    #     rates=[1, 1, 1, 1],
    #     padding='VALID'
    # )
    # image_jpeg = tf.reshape(image_jpeg, (-1, 32, 32, c))

    psnr = 0 # np.mean(tf.image.psnr(images, image_jpeg, max_val=1.0))
    ssim = 0 # np.mean(tf.image.ssim(images, image_jpeg, max_val=1.0))
    ax[2].set_title(f'Conventional (JPEG-based)\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
    ax[2].imshow(image_jpeg)

    plt.show()

# %%
