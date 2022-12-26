#%%

# Mockup USRP
# just relay constellations from receive UDP socket to send UDP socket
import socket
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config.usrp_config import USRP_HOST, USRP_PORT, RCV_ADDR, RCV_PORT
from usrp.pilot import SAMPLE_SIZE
from utils.networking import send_constellation_udp, receive_udp, METADATA_BYTES
from time import sleep

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (RCV_ADDR, RCV_PORT)
rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = (USRP_HOST, USRP_PORT)
rcv_sock.bind(rcv_addr)
rcv_sock.settimeout(2)

ADDITIONAL_LENGTH = 1024

while True:
  send_data = receive_udp(rcv_sock, SAMPLE_SIZE * 4)
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
  send_constellation_udp(send_data, send_sock, send_addr)
  print('Send done', send_addr)
  sleep(0.1)

# 그대로 보내면 안 되고
# USRP처럼 저거 바꿔서 보내야됨

# %%
