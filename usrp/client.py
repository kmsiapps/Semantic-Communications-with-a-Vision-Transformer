#%%
import socket
import cv2
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import numpy as np
import struct
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.networking import receive_and_save_binary, send_binary
from utils.usrp_utils import get_lci_lcq_compensation, compensate_signal, receive_constellation_tcp
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
    clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSock.connect(('1.233.219.33', 8080))
    print('Connected to server')
    
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSock.bind((RCV_ADDR, RCV_PORT))
    serverSock.listen(1)
    print('Waiting')
    usrpSock, addr = serverSock.accept()
    print('Connected to USRP:', addr)

    LCI, LCQ = 0, 0
    BUFF_SIZE = 4096
    while True:
        demo_type = int(input("Select image type: CIFAR (0) vs CELEB (1) vs CAM (2)?: "))
        use_cache = False

        if demo_type == 0:
            TARGET_IMAGE = 'cifar.png'
            use_cache = (int(input("Use cached constellation (0/1)? ")) == 1)
        elif demo_type == 1:
            TARGET_IMAGE = 'celeb.png'
        else:
        # Webcam capture
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        LCI, LCQ = get_lci_lcq_compensation(usrpSock)

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

        # Send/receive constellations (to USRP)
        usrpSock.send(constellations.tobytes())
        data = receive_constellation_tcp(usrpSock)

        rcv_iq, raw_i, raw_q = compensate_signal(data, LCI, LCQ)

        # Send channel corrupted (I/Q compensated) constellations (=rcv_iq)
        np.savez_compressed(f'{TEMP_DIRECTORY}/snd_iq.npz', rcv_iq=rcv_iq)
        send_binary(clientSock, f'{TEMP_DIRECTORY}/snd_iq.npz')

        # Receive decoded image
        receive_and_save_binary(clientSock, f'{TEMP_DIRECTORY}/decoded.png')

        # receive effective SNR
        SNRdB = struct.unpack('!f', clientSock.recv(4))[0]

        # TODO: plot
        fig, ax = plt.subplots(1, 3)
        fig.set_figheight(7)
        fig.set_figwidth(21)

        ax[0].set_title('Original')
        original_pil = pilimg.open(TARGET_IMAGE).convert('RGB')
        original = np.array(original_pil)
        ax[0].imshow(original)

        proposed = np.array(pilimg.open(f'{TEMP_DIRECTORY}/decoded.png').convert('RGB'))
        psnr = peak_signal_noise_ratio(original, proposed)
        ssim = structural_similarity(original, proposed, channel_axis=2)
        ax[1].set_title(f'Proposed\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
        ax[1].imshow(proposed)

        # JPEG rate-match algorithm
        quality_max = 100
        quality = 50
        quality_min = 0

        while True:
            original_pil.save(f'{TEMP_DIRECTORY}/source_jpeg.jpg', quality=quality)
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

        image_jpeg = np.array(pilimg.open(f'{TEMP_DIRECTORY}/source_jpeg.jpg').convert('RGB'))
        psnr = peak_signal_noise_ratio(original, image_jpeg)
        ssim = structural_similarity(original, image_jpeg, channel_axis=2)
        ax[2].set_title(f'Conventional (JPEG-based)\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.2f}')
        ax[2].imshow(image_jpeg)

        plt.show()
        print(f'Effective SNR: {SNRdB:.2f} dB')
        time.sleep(0.1)


# %%
