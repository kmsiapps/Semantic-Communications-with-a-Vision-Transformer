import tensorflow as tf
import random

class RayleighChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, clip_snrdB=5):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.clip_snr = 10 ** (clip_snrdB / 10)
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading, where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        
        # batch-wise slow fading
        h = tf.random.normal(
            (1, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        yhat = h * x + n

        return yhat
    

    def get_config(self):
        config = super().get_config()
        return config


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        y = x + n
        return y
    

    def get_config(self):
        config = super().get_config()
        return config


class RicianChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, k=2):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.k = k
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading (for NLOS part), where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        
        # batch-wise slow fading
        h = tf.random.normal(
            (1, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        k = self.k

        yhat = tf.math.sqrt(1 / (1+k)) * h * x + tf.math.sqrt(k / (1+k)) * x + n

        return yhat