import struct
from tqdm import tqdm
import socket
import time

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
          if not read_header:
            array_length = int.from_bytes(_data[:METADATA_BYTES], byteorder='big')
            ARRAY_END = array_length * 4 + METADATA_BYTES
            read_header = True
            progress.reset(total=ARRAY_END)
            progress.update(len(_data))
          rcv_data += _data
          progress.update(len(_data))
          
          if len(rcv_data) >= ARRAY_END:
            data = rcv_data[METADATA_BYTES:ARRAY_END]
            return data


def receive_udp(rcv_sock, total_bytes):
  rcv_data = bytes()
  _data = bytes()
  ARRAY_END = total_bytes
  with tqdm(initial=0, total=total_bytes) as progress:
    while True:
        rcv = True
        try:
          _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
        except socket.timeout:
          rcv = False
        if rcv:
          rcv_data += _data
          progress.update(len(_data))
          if len(rcv_data) >= ARRAY_END:
            return rcv_data


def send_constellation_udp(send_data, send_sock, send_addr):
  for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)
    time.sleep(0.001)
