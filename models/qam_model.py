import tensorflow as tf

from utils.qam_modem_tf import QAMModulator, QAMDemodulator
from models.channellayer import RayleighChannel, AWGNChannel

class QAMModem(tf.keras.Model):
    def __init__(self, snrdB=25, order=256, channel='Rayleigh'):
        super().__init__()
        self.encoder = QAMModulator(order=order)
        self.decoder = QAMDemodulator(order=order)
        if channel.lower() == 'rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        else:
            self.channel = AWGNChannel(snrdB=snrdB)

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.channel(x)
        x = self.decoder(x)
        return x
