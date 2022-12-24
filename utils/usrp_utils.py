import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from usrp.pilot import p_start_i, p_end_i, p_start_q, p_end_q, PILOT_SIZE, SAMPLE_SIZE
from utils.networking import *
from config.usrp_config import NORMALIZE_CONSTANT

EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - PILOT_SIZE * 2

def to_constellation_array(i, q, i_pilot=True, q_pilot=True):
  i_start_pilot = p_start_i if i_pilot else np.zeros(len(p_start_i))
  i_end_pilot = p_end_i if i_pilot else np.zeros(len(p_end_i))
  q_start_pilot = p_start_q if q_pilot else np.zeros(len(p_start_q))
  q_end_pilot = p_end_q if q_pilot else np.zeros(len(p_end_q))
  
  p_i = np.concatenate([i_start_pilot, i, i_end_pilot]).astype(np.int16)
  p_q = np.concatenate([q_start_pilot, q, q_end_pilot]).astype(np.int32)

  # Q is in the higher bits!
  i_ = p_i
  q_ = np.left_shift(p_q, 16)

  data = np.bitwise_or(q_, i_.view(dtype=np.uint16)).byteswap(inplace=True)
  return data


def get_lci_lcq_compensation(rcv_sock, rcv_addr, send_sock, send_addr):
  # Send LCI message
  x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
  i = 0 * np.cos(x) * 32767
  q = 1 * np.sin(x) * 32767

  data = to_constellation_array(i, q, i_pilot=False, q_pilot=True)
  send_data = data.tobytes()
  send_constellation_udp(send_data, send_sock, send_addr)

  # Receive LCI message
  data = receive_constellation_udp(rcv_sock)
  array_length = len(data) // 4
  d_iq = struct.unpack('!' + 'f' * array_length, data)

  raw_i = np.array(d_iq[:array_length // 2])
  raw_q = np.array(d_iq[array_length // 2:])

  pilot_mask_q = np.concatenate([p_start_q, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_q])
  start_idx = np.argmax(np.abs(np.correlate(raw_q, pilot_mask_q))) + PILOT_SIZE

  # get noise & zero-mean normalize
  LCI = np.mean(raw_i[start_idx:start_idx+EXPECTED_SAMPLE_SIZE] / (raw_q[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]+0.0001))

  # Send LCQ message
  x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
  i = 1 * np.cos(x) * 32767
  q = 0 * np.sin(x) * 32767

  data = to_constellation_array(i, q, i_pilot=True, q_pilot=False)
  send_data = data.tobytes()
  send_constellation_udp(send_data, send_sock, send_addr)

  # Receive LCQ message
  data = receive_constellation_udp(rcv_sock)
  array_length = len(data) // 4
  d_iq = struct.unpack('!' + 'f' * array_length, data)

  raw_i = np.array(d_iq[:array_length // 2])
  raw_q = np.array(d_iq[array_length // 2:])

  pilot_mask_i = np.concatenate([p_start_i, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_i])
  start_idx = np.argmax(np.abs(np.correlate(raw_i, pilot_mask_i))) + PILOT_SIZE

  # get noise & zero-mean normalize
  LCQ = np.mean(raw_q[start_idx:start_idx+EXPECTED_SAMPLE_SIZE] / (raw_i[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]+0.0001))
  return LCI, LCQ


def compensate_signal(data, LCI, LCQ):
  array_length = len(data) // 4
  d_iq = struct.unpack('!' + 'f' * array_length, data)

  raw_i = np.array(d_iq[:array_length // 2])
  raw_q = np.array(d_iq[array_length // 2:])

  raw_i -= np.mean(raw_i)
  raw_q -= np.mean(raw_q)

  # Leakage compensation
  i_compensated = (LCI * raw_q - raw_i) / (LCI*LCQ-1)
  q_compensated = (LCQ * raw_i - raw_q) / (LCQ*LCI-1) # / 3.5 # magic num

  pilot_mask_i = np.concatenate([p_start_i, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_i])
  start_idx = np.argmax(np.abs(np.correlate(i_compensated, pilot_mask_i))) + PILOT_SIZE

  # get noise & zero-mean normalize
  noises = np.concatenate(
    [i_compensated[:start_idx-PILOT_SIZE],
    i_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
  )
  n = np.mean(noises)
  i_compensated -= n
  noise_power_i = np.sum(noises ** 2) / len(noises)

  # get average h
  p_start_rx = i_compensated[(start_idx - PILOT_SIZE):start_idx]
  p_end_rx = i_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
  p_rx = np.concatenate([p_start_rx, p_end_rx])
  p = np.concatenate([p_start_i, p_end_i]) / 32767
  nonzero_idx = np.where(p != 0)
  hi = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])

  i_compensated /= hi

  # get data
  ihat = i_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]
  signal_power = np.sum((ihat * hi) ** 2) / len(ihat)

  # plt.title('Decoded In-phase signal\n(h & n compensated)')
  # plt.plot(ihat)
  # plt.show()

  # Quadrature phase detection =================
  pilot_mask = np.concatenate([p_start_q, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_q])
  start_idx = np.argmax(np.abs(np.correlate(q_compensated, pilot_mask))) + PILOT_SIZE

  # get noise & zero-mean normalize
  noises = np.concatenate(
    [q_compensated[:start_idx-PILOT_SIZE],
    q_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
  )
  n = np.mean(noises)
  q_compensated -= n
  noise_power_q = np.sum(noises ** 2) / len(noises)

  # get average h
  p_start_rx = q_compensated[(start_idx - PILOT_SIZE):start_idx]
  p_end_rx = q_compensated[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
  p_rx = np.concatenate([p_start_rx, p_end_rx])
  p = np.concatenate([p_start_q, p_end_q]) / 32767
  nonzero_idx = np.where(p != 0)
  hq = np.sum(np.divide(p_rx[nonzero_idx], p[nonzero_idx])) / len(p[nonzero_idx])

  q_compensated /= hq

  # get data
  qhat = q_compensated[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]

  # plt.title('Decoded Quadrature-phase signal\n(h & n compensated)')
  # plt.plot(qhat)
  # plt.show()

  max_i = NORMALIZE_CONSTANT
  max_q = NORMALIZE_CONSTANT

  rcv_iq = np.zeros(shape=(len(ihat), 2))
  rcv_iq[:, 0] = ihat * max_i
  rcv_iq[:, 1] = qhat * max_q

  return rcv_iq

