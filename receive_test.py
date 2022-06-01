#%%

import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
import math

from pilot import p_start, p_end, PILOT_SIZE, SAMPLE_SIZE

rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = ("0.0.0.0", 50000)
rcv_sock.bind(rcv_addr)

rcv_sock.settimeout(2)

SOCK_BUFF_SIZE = 16384
EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - PILOT_SIZE * 2

# while True:
#     try:
#         _, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
#     except socket.timeout:
#         break

rcv_data = bytes()
_data = bytes()

read_header = False
while True:
    rcv = True
    try:
        _data, _ = rcv_sock.recvfrom(SOCK_BUFF_SIZE)
    except socket.timeout:
        print('.', end='')
        rcv = False
    if rcv:
        # print()
        if not read_header:
            array_length = int.from_bytes(_data[:4], byteorder='big')
            ARRAY_END = array_length * 4 + 4
            read_header = True
        rcv_data += _data
        print(f"RCV: {len(rcv_data)}/{ARRAY_END} (+{len(_data)})", end='')
        
        if len(rcv_data) >= ARRAY_END:
            data = rcv_data[4:ARRAY_END]
            d_iq = struct.unpack('!' + 'f' * array_length, data)

            raw_i = np.array(d_iq[:array_length // 2])
            raw_q = np.array(d_iq[array_length // 2:])

            # Leakage compensation
            LCI = 2.24 # Leakage ratio constant
            LCQ = 2.5
            i_compensated = (raw_i - LCI * raw_q) / (1+LCI**2)
            q_compensated = (raw_q + LCQ * raw_i) / (1+LCQ**2) / 3.5 # magic num

            print()
            print(f"{len(rcv_data)} => ", end='')
            rcv_data = rcv_data[ARRAY_END:]
            print(f"{len(rcv_data)}")
            read_header = False

            pilot_mask = np.concatenate([p_start, np.zeros(EXPECTED_SAMPLE_SIZE), p_end])
            # start_idx = np.argmax(np.abs(np.correlate(i_compensated, pilot_mask))) + PILOT_SIZE
            start_idx = PILOT_SIZE

            # get noise & zero-mean normalize
            noises = np.concatenate(
                [i_compensated[:start_idx-PILOT_SIZE],
                i_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
            )
            n = np.mean(noises)
            i_compensated -= n
            noise_power = np.sum(noises ** 2) / len(noises)

            # get average h
            p_start_rx = i_compensated[(start_idx - PILOT_SIZE):start_idx]
            p_end_rx = i_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
            p_rx = np.concatenate([p_start_rx, p_end_rx])
            p = np.concatenate([p_start, p_end]) / 32767
            nonzero_idx = np.where(p != 0)
            hi = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])
            hi *= 3.0 # 1.15 # magic num

            i_compensated /= hi

            # get data
            ihat = i_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]
            signal_power = np.sum((ihat * hi) ** 2) / len(ihat)

            plt.title('Decoded In-phase signal\n(h & n compensated)')
            plt.plot(ihat)
            plt.show()

            # Quadrature phase detection =================
            pilot_mask = np.concatenate([p_start, np.zeros(EXPECTED_SAMPLE_SIZE), p_end])
            # start_idx = np.argmax(np.abs(np.correlate(q_compensated, pilot_mask))) + PILOT_SIZE
            start_idx = PILOT_SIZE

            # get noise & zero-mean normalize
            noises = np.concatenate(
                [q_compensated[:start_idx-PILOT_SIZE],
                q_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
            )
            n = np.mean(noises)
            q_compensated -= n
            noise_power = np.sum(noises ** 2) / len(noises)

            # get average h
            p_start_rx = q_compensated[(start_idx - PILOT_SIZE):start_idx]
            p_end_rx = q_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
            p_rx = np.concatenate([p_start_rx, p_end_rx])
            p = np.concatenate([p_start, p_end]) / 32767
            nonzero_idx = np.where(p != 0)
            hq = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])
            hq *= 2.5 # 1.15 # magic num

            q_compensated /= hq

            # get data
            qhat = q_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]

            plt.title('Decoded Quadrature-phase signal\n(h & n compensated)')
            plt.plot(qhat)
            plt.show()

            plt.title('Raw I')
            plt.plot(raw_i) #[start_idx-PILOT_SIZE*2:start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE*2])
            plt.show()

            plt.title('Raw Q')
            plt.plot(raw_q) #[start_idx-PILOT_SIZE*2:start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE*2])
            plt.show()

            signal_power = np.sum((qhat * hq) ** 2) / len(qhat)
            snr = signal_power / noise_power
            snrdB = 10 * math.log10(snr)
            print(f"SNR: {snrdB:.2f}dB")
            print(f"leakage ratio: {np.max(raw_i[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]) / np.max(raw_q[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]):.2f}")
            # print(f"leakage ratio: {np.median((raw_i[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]) / (raw_q[start_idx:start_idx+EXPECTED_SAMPLE_SIZE])):.2f}")

            rcv_data = bytes()

# i = decoded_data[0::2]
# q = decoded_data[1::2]

plt.plot(range(len(decoded_data)), decoded_data)
plt.show()

# %%
