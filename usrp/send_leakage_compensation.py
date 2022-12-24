#%%

import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
from usrp.pilot import p_start_i, p_end_i, p_start_q, p_end_q, PILOT_SIZE, SAMPLE_SIZE
from config.train_config import RCV_ADDR, RCV_PORT, USRP_HOST, USRP_PORT
from time import sleep

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (USRP_HOST, USRP_PORT)
SEND_SOCK_BUFF_SIZE = 256

#%%
# get ki (set i=0) =========================================
x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
i = 0 * np.cos(x) * 32767
q = 1 * np.sin(x) * 32767

zero_pilot = np.zeros(len(p_start_i))
p_i = np.concatenate([zero_pilot, i, zero_pilot]).astype(np.int16)
p_q = np.concatenate([p_start_q, q, p_end_q]).astype(np.int32)

# Q is in the higher bits!
i_ = p_i
q_ = np.left_shift(p_q, 16)

data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)
send_data = data.tobytes()

for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)
print(f'LCI SEND DONE. len: {len(send_data)}')

# %%

input('Press Enter to send LCQ')


x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
i = 1 * np.cos(x) * 32767
q = 0 * np.sin(x) * 32767

zero_pilot = np.zeros(len(p_start_i))
p_i = np.concatenate([p_start_i, i, p_end_i]).astype(np.int16)
p_q = np.concatenate([zero_pilot, q, zero_pilot]).astype(np.int32)

# Q is in the higher bits!
i_ = p_i
q_ = np.left_shift(p_q, 16)

data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)
send_data = data.tobytes()

for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)
print(f'LCQ SEND DONE. len: {len(send_data)}')

#%%
send_sock.close()

# %%
