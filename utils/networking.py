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
  partial_data = sock.recv(4)
  size, *_ = struct.unpack(
    'I',
    partial_data[:4]
  )
  data += partial_data[4:]

  with tqdm(initial=len(data), total=size) as progress:
    while len(data) < size:
      partial_data = sock.recv(min(BUFF_SIZE, size - len(data)))
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


def receive_constellation_tcp(rcv_sock, total_bytes=None):
  rcv_data = bytes()
  _data = bytes()
  if not total_bytes:
    ARRAY_END = 2 * SOCK_BUFF_SIZE # some initial value
    read_header = False
  else:
    ARRAY_END = total_bytes
    read_header = True
  with tqdm(initial=0, total=ARRAY_END) as progress:
    while True:
      rcv = True
      try:
        _data = rcv_sock.recv(min(SOCK_BUFF_SIZE, ARRAY_END-len(rcv_data)))
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
          break
  
  if total_bytes:
    # No size information in the front
    data = rcv_data
  else:
    data = rcv_data[METADATA_BYTES:ARRAY_END]
  return data

