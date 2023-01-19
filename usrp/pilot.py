#%%

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

PILOT_SIZE = 256
SAMPLE_SIZE = 512 * 8 * 8 + 2 * PILOT_SIZE

np.random.seed(20220511)
x = np.linspace(0, 2 * 2 * np.pi, PILOT_SIZE)

p_start_i = (np.random.randint(0, 2, size=PILOT_SIZE)) * 32767
p_end_i = (-1 * np.random.randint(0, 2, size=PILOT_SIZE)) * 32767

p_start_q = (np.random.randint(0, 2, size=PILOT_SIZE)) * 32767
p_end_q = (-1 * np.random.randint(0, 2, size=PILOT_SIZE)) * 32767

# %%
