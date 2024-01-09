#%%
import socket
import numpy as np
import time
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.usrp_utils import compensate_signal, receive_constellation_tcp, to_constellation_array
from config.usrp_config import CLIENT_ADDR, CLIENT_PORT, TEMP_DIRECTORY

if __name__ == '__main__':
    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)
    
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSock.bind((CLIENT_ADDR, CLIENT_PORT))
    serverSock.listen(1)
    print('Waiting')
    usrpSock, addr = serverSock.accept()
    print('Connected to USRP:', addr)

    BUFF_SIZE = 4096
    
    while True:
        amp = float(input("Signal amplitude? (0~1): "))
        use_cache = False

        i = amp * np.sin(np.linspace(0, 4 * np.pi, 32768)) # random float. range: [-1, 1]
        q = amp * np.cos(np.linspace(0, 4 * np.pi, 32768)) 
        # i, q: [-1, 1] ranged. shape: (32768,)
        
        i = np.round(np.clip(i * 32767, -32767, 32767))
        q = np.round(np.clip(q * 32767, -32767, 32767))         
        constellations = to_constellation_array(i, q, i_pilot=True, q_pilot=True)
        
        i = i.astype(np.float32) / 32767.0
        q = q.astype(np.float32) / 32767.0

        # Send/receive constellations (to USRP)
        usrpSock.send(constellations.tobytes())
        data = receive_constellation_tcp(usrpSock)

        rcv_iq, raw_i, raw_q = compensate_signal(data)
        
        rcv_i = np.right_shift(np.left_shift(rcv_iq, 16), 16).astype('>f4') / 32767
        rcv_q = np.right_shift(rcv_iq, 16).astype('>f4') / 32767
        
        err_q = rcv_q - q
        err_i = rcv_i - i
        
        err = np.mean((err_q)**2 + (err_i)**2)
        print(f'MSE: {err:.6f}')
        print(f'SNR: {10*np.log(np.mean(i**2+q**2)/err)/np.log(10):.2f} dB')
        
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        plt.plot(i)
        plt.subplot(1, 3, 2)
        plt.plot(rcv_i)
        plt.subplot(1, 3, 3)
        plt.plot(err_i)
        plt.show()
        
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        plt.plot(q)
        plt.subplot(1, 3, 2)
        plt.plot(rcv_q)
        plt.subplot(1, 3, 3)
        plt.plot(err_q)
        plt.show()
        
        time.sleep(0.1)


# %%