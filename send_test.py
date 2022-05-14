#%%

import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
from pilot import p_start, p_end, PILOT_SIZE, SAMPLE_SIZE

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = ("192.168.1.121", 60000)
SEND_SOCK_BUFF_SIZE = 256

# while True:
#     try:
#         _, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
#     except socket.timeout:
#         break

rcv_data = bytes()
_data = bytes()

x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
i = 1 * np.ones(x.size) * 32767 # np.cos(x) * 32767
q = 1 * np.sin(x) * 32767 # np.ones(x.size) * 32767 # np.sin(x) * 32767

zero_pilot = np.zeros(len(p_start))

p_i = np.concatenate([p_start, i, p_end]).astype(np.int16)
# p_i = np.concatenate([zero_pilot, i, zero_pilot]).astype(np.int16)
p_q = np.concatenate([p_start, q, p_end]).astype(np.int32)

# Q is in the higher bits!
i_ = p_i
q_ = np.left_shift(p_q, 16)

data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)
send_data = data.tobytes()

for j in range(0, len(send_data), SEND_SOCK_BUFF_SIZE):
    _data = send_data[j:min(len(send_data), j + SEND_SOCK_BUFF_SIZE)]
    send_sock.sendto(_data, send_addr)
print(f'ORIGIN SEND DONE. len: {len(send_data)}')

send_sock.close()

k = np.bitwise_or(q_, i_.view(dtype=np.uint16))
qhat = np.right_shift(k, 16)
ihat = np.right_shift(np.left_shift(k, 16), 16)

plt.title('Raw IQ signal')
plt.plot(ihat / 32767)
plt.plot(qhat / 32767)
plt.show()

EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - 2*PILOT_SIZE
pilot_mask = np.concatenate([p_start, np.zeros(EXPECTED_SAMPLE_SIZE), p_end])

start_idx = np.argmax(np.correlate(ihat, pilot_mask)) + PILOT_SIZE
ihat_detected = ihat[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]

plt.title('Inphase payload')
plt.plot(ihat_detected / 32767)
plt.show()

# %%
