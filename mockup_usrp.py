# Mockup USRP
# just relay constellations from receive UDP socket to send UDP socket
from socket import *
from config import USRP_HOST, USRP_PORT, RCV_ADDR, RCV_PORT
from utils import receive_constellation_udp, send_constellation_udp
from time import sleep

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_addr = (RCV_ADDR, RCV_PORT)
rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rcv_addr = (USRP_HOST, USRP_PORT)
rcv_sock.bind(rcv_addr)

while True:
  send_data = receive_constellation_udp(rcv_sock)
  send_constellation_udp(send_data, send_sock, send_addr)
  sleep(0.1)
