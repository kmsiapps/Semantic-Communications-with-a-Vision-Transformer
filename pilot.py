import numpy as np

# p_start = np.array([-1, 1, 1, -1,
#                     -1, -1, 1, 1,
#                     1, 1, -1, 1,
#                     1, 1, -1, 1]) * 32767

# p_end = np.array([1, 1, -1, 1,
#                   -1, 1, -1, 1,
#                   1, -1, -1, 1,
#                   -1, 1, -1, -1]) * 32767

PILOT_SIZE = 256
SAMPLE_SIZE = 512 * 8 * 8 + 2 * PILOT_SIZE

np.random.seed(20220511)
'''
p_start = (2 * np.random.randint(0, 2, size=PILOT_SIZE) - 1) * 32767
p_end = (2 * np.random.randint(0, 2, size=PILOT_SIZE) - 1) * 32767
'''

x = np.linspace(0, 2 * 2 * np.pi, PILOT_SIZE)

p_start_i = (np.random.randint(0, 2, size=PILOT_SIZE)) * 32767
p_end_i =  (-1 * np.random.randint(0, 2, size=PILOT_SIZE)) * 32767

p_start_q = (np.random.randint(0, 2, size=PILOT_SIZE)) * 32767
p_end_q =  (-1 * np.random.randint(0, 2, size=PILOT_SIZE)) * 32767
