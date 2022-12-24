import struct
from tqdm import tqdm
from socket import *
import tensorflow as tf

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from usrp.pilot import PILOT_SIZE, SAMPLE_SIZE

BUFF_SIZE = 4096
SEND_SOCK_BUFF_SIZE = 256
SOCK_BUFF_SIZE = 16384 # EXPECTED_DATA_LENGTH * 4 + 4
METADATA_BYTES = 4

def receive_and_save_binary(sock, filename):
  data = b''
  partial_data = sock.recv(BUFF_SIZE)
  size, *_ = struct.unpack(
    'I',
    partial_data[:4]
  )
  data += partial_data[4:]

  with tqdm(initial=len(data), total=size) as progress:
    while len(data) < size:
      partial_data = sock.recv(BUFF_SIZE)
      data += partial_data
      progress.update(len(partial_data))

  with open(filename, 'wb') as f:
    f.write(data)


def send_binary(sock, filename):
  with open(filename, "rb") as f:
    payload = f.read()
    header = struct.pack('I', len(payload))
    data = header + payload
    sock.send(data)


def receive_constellation_udp(rcv_sock):
  rcv_data = bytes()
  _data = bytes()
  read_header = False
  with tqdm(initial=0, total=100) as progress:
    while True:
        rcv = True
        try:
          _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
        except socket.timeout:
          rcv = False
        if rcv:
          # print()
          if not read_header:
            array_length = int.from_bytes(_data[:METADATA_BYTES], byteorder='big')
            ARRAY_END = array_length * 4 + METADATA_BYTES
            read_header = True
          rcv_data += _data
          progress.update(len(_data) / ARRAY_END * 100)
          
          if len(rcv_data) >= ARRAY_END:
            data = rcv_data[METADATA_BYTES:ARRAY_END]
            return data


def send_constellation_udp(send_data, send_sock, send_addr):
  for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)



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
