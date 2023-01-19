import numpy as np
import struct

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from usrp.pilot import p_start_i, p_end_i, p_start_q, p_end_q, PILOT_SIZE, SAMPLE_SIZE
from utils.networking import *

EXPECTED_SAMPLE_SIZE = SAMPLE_SIZE - PILOT_SIZE * 2

def to_constellation_array(i, q, i_pilot=True, q_pilot=True):
  # data: numpy array
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


def get_lci_lcq_compensation(clientSock):
	# Get leakage compensation constant
	# Leakage model: 
	# ihat = LCI * q + i
	# qhat = LCQ * i + q

  # Send LCI message
  x = np.linspace(0, 4 * 2 * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
  i = 0 * np.cos(x) * 32767
  q = 1 * np.sin(x) * 32767

  data = to_constellation_array(i, q, i_pilot=False, q_pilot=True)
  send_data = data.tobytes()
  
  clientSock.send(send_data)
  data = receive_constellation_tcp(clientSock)
  print('LCI set.')

  array_length = len(data) // 4
  d_iq = struct.unpack('!' + 'f' * array_length, data)

  raw_i = np.array(d_iq[:array_length // 2])
  raw_q = np.array(d_iq[array_length // 2:])

  pilot_mask_q = np.concatenate([p_start_q, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_q])
  start_idx = np.argmax(np.abs(np.correlate(raw_q, pilot_mask_q))) + PILOT_SIZE
  # get noise & zero-mean normalize
  LCI = np.mean(raw_i[start_idx:start_idx+EXPECTED_SAMPLE_SIZE] / (raw_q[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]+0.0001))

  # Send LCQ message
  x = np.linspace(0, (4 * 2) * np.pi, SAMPLE_SIZE - 2*PILOT_SIZE)
  i = 1 * np.cos(x) * 32767
  q = 0 * np.sin(x) * 32767

  data = to_constellation_array(i, q, i_pilot=True, q_pilot=False)
  send_data = data.tobytes()

  clientSock.send(send_data)
  data = receive_constellation_tcp(clientSock)
  print('LCQ set.')

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

  # Leakage compensation
  i_compensated = (LCI * raw_q - raw_i) / (LCI*LCQ-1)
  q_compensated = (LCQ * raw_i - raw_q) / (LCQ*LCI-1) # / 3.5 # magic num

  pilot_mask_i = np.concatenate([p_start_i, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_i])
  pilot_mask_q = np.concatenate([p_start_q, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_q])
  start_idx = np.argmax(np.correlate(i_compensated, pilot_mask_i) + np.abs(np.correlate(q_compensated, pilot_mask_q))) + PILOT_SIZE

  # get noise & zero-mean normalize
  noises = np.concatenate(
    [i_compensated[:start_idx-PILOT_SIZE],
    i_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
  )
  n = np.mean(noises)
  i_compensated -= n

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

  # get noise & zero-mean normalize
  noises = np.concatenate(
    [q_compensated[:start_idx-PILOT_SIZE],
    q_compensated[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
  )
  n = np.mean(noises)
  q_compensated -= n

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

  ihat = np.clip((ihat * 32767).astype(np.int32), -32767, 32767)
  qhat = np.clip((qhat * 32767).astype(np.int32), -32767, 32767)

  rcv_iq = to_constellation_array(ihat, qhat, i_pilot=False, q_pilot=False)[PILOT_SIZE:-PILOT_SIZE].byteswap()

  return rcv_iq, raw_i, raw_q

