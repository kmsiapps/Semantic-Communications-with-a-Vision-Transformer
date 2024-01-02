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


def compensate_signal(data):
  array_length = len(data) // 4
  d_iq = struct.unpack('!' + 'f' * array_length, data)

  raw_i = np.array(d_iq[:array_length // 2])
  raw_q = np.array(d_iq[array_length // 2:])

  pilot_mask_i = np.concatenate([p_start_i, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_i])
  pilot_mask_q = np.concatenate([p_start_q, np.zeros(EXPECTED_SAMPLE_SIZE), p_end_q])
  start_idx = np.argmax(np.abs(np.correlate(raw_i, pilot_mask_i)) + np.abs(np.correlate(raw_q, pilot_mask_q))) + PILOT_SIZE

  iq = raw_i + 1j*raw_q

  # get noise & zero-mean normalize
  noises = np.concatenate(
  [iq[:start_idx-PILOT_SIZE],
  iq[start_idx+EXPECTED_SAMPLE_SIZE+PILOT_SIZE:]]
  )
  n = np.mean(noises)
  iq -= n

  # channel compensation
  p_start_rx = iq[(start_idx - PILOT_SIZE):start_idx]
  p_end_rx = iq[(start_idx + EXPECTED_SAMPLE_SIZE):(start_idx + EXPECTED_SAMPLE_SIZE + PILOT_SIZE)]
  p_rx = np.concatenate([p_start_rx, p_end_rx])

  p_i = np.concatenate([p_start_i, p_end_i]) / 32767
  p_q = np.concatenate([p_start_q, p_end_q]) / 32767
  p_gt = p_i + 1j*p_q

  # get average h_amp
  p_amp_rx = np.sqrt(np.real(p_rx) ** 2 + np.imag(p_rx) ** 2)
  p_amp = np.sqrt(np.real(p_gt) ** 2 + np.imag(p_gt) ** 2)
  nonzero_idx = np.where(np.real(p_amp) != 0)
  h_amp = np.mean(np.divide(p_amp_rx[nonzero_idx], p_amp[nonzero_idx])) / 0.5 * 0.6 # magic num

  # get average h_phase
  nonzero_idx = np.where(np.real(p_gt) != 0)
  p_phase_rx = np.angle(p_rx[nonzero_idx])
  p_phase_gt = np.angle(p_gt[nonzero_idx])
  p_phase_diff = p_phase_rx - p_phase_gt
  p_phase_diff = np.mean(np.arctan2(np.sin(p_phase_diff), np.cos(p_phase_diff))) # normalize to (-pi, pi)

  # compensate signal    
  iq /= h_amp  
  iq *= np.exp(-1j*p_phase_diff)

  ihat = np.real(iq)[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]
  qhat = np.imag(iq)[start_idx:start_idx+EXPECTED_SAMPLE_SIZE]

  ihat = np.clip((ihat * 32767).astype(np.int32), -32767, 32767)
  qhat = np.clip((qhat * 32767).astype(np.int32), -32767, 32767)

  rcv_iq = to_constellation_array(ihat, qhat, i_pilot=False, q_pilot=False)[PILOT_SIZE:-PILOT_SIZE].byteswap()

  return rcv_iq, raw_i, raw_q

