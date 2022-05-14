import tensorflow as tf
import random

class RayleighChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, clip_snrdB=5):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.clip_snr = 10 ** (clip_snrdB / 10)
    

    def call(self, x):
        '''
        x: inputs with shape (2, Batch, Any)
           where first dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading, where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[0] == 2, "input shape should be (2, Batch, Any), where first dimension denotes i and q, respectively"
        assert len(x.shape) >= 3, "input shape should be (2, Batch, Any)"
        
        i = x[0]
        q = x[1]

        b = x.shape[1]
        h_shape = [1 for _ in range(len(x.shape))]
        h_shape[1] = b
        h_shape = tuple(h_shape)

        # power normalization
        normalizer = tf.math.sqrt(
            tf.math.reduce_mean(i ** 2 + q ** 2)
        )
        x = x / normalizer
        
        h = tf.random.normal(
            h_shape,
            mean=0,
            stddev=(1/2)**0.5
        )

        h_sign = h / tf.abs(h)

        h = tf.abs(h)
        h = tf.clip_by_value(
            h,
            clip_value_min = 1/self.clip_snr,
            clip_value_max = self.clip_snr
        )

        h = h * h_sign

        '''
        # for random SNR value, use below:
        snrdB = random.randint(0, 40)
        snr = 10 ** (snrdB / 10)
        '''
        snr = self.snr

        n = tf.random.normal(
            x.shape,
            mean=0,
            stddev=tf.math.sqrt(1/(2*snr))
        )

        nhat = tf.math.divide_no_nan(n, h)
        yhat = x + nhat
        yhat = yhat * normalizer

        return yhat


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    

    def call(self, x):
        '''
        x: inputs with shape (2, Any)
           where first dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[0] == 2, "input shape should be (2, Any), which denotes i and q, respectively"
        
        i = x[0]
        q = x[1]

        # power normalization
        normalizer = tf.math.sqrt(
            tf.math.reduce_mean(i ** 2 + q ** 2)
        )
        x = x / normalizer

        '''
        # for random SNR value, use below:
        snrdB = random.randint(0, 40)
        snr = 10 ** (snrdB / 10)
        '''
        snr = self.snr

        n = tf.random.normal(
            x.shape,
            mean=0,
            stddev=tf.math.sqrt(1/(2*snr))
        )

        y = x + n

        yhat = y * normalizer
        return yhat
