#%%

# Mockup USRP
# USRP connects to client (via TCP) to conduct wireless transmissions of given I/Q symbols.

import socket
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.usrp_config import CLIENT_ADDR, CLIENT_PORT
from usrp.pilot import SAMPLE_SIZE
from utils.networking import receive_constellation_tcp, METADATA_BYTES
from time import sleep

clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSock.connect((CLIENT_ADDR, CLIENT_PORT))
print('Connected to client')

ADDITIONAL_LENGTH = 1024

while True:
  send_data = receive_constellation_tcp(clientSock)
  print('Receive done')
  
  # mock USRP return value
  data = np.frombuffer(send_data, dtype=np.int32).byteswap()
  i_mask = 0x00001111 * np.ones(len(data), dtype=np.uint32)
  i = np.right_shift(np.left_shift(data, 16), 16).astype('>f4') / 32767
  q = np.right_shift(data, 16).astype('>f4') / 32767
  
  noise = np.random.normal(loc=0, scale=0.1, size=ADDITIONAL_LENGTH)
  i = np.concatenate((i, noise))
  q = np.concatenate((q, noise))

  iq_concat = np.concatenate((i, q), axis=-1).astype('>f4')
  send_data = len(iq_concat).to_bytes(length=METADATA_BYTES, byteorder='big') + iq_concat.tobytes()
  clientSock.send(send_data)
  print('Send done')


# %%
