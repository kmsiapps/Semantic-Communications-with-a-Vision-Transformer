import tensorflow as tf
import random

class RayleighChannel(tf.keras.layers.Layer):
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

        # Quantization
        i = tf.cast(i, dtype=tf.float16)
        q = tf.cast(q, dtype=tf.float16)
        i = tf.cast(i, dtype=tf.float32)
        q = tf.cast(q, dtype=tf.float32)

        # power normalization
        normalizer = tf.math.sqrt(
            tf.math.reduce_mean(i ** 2 + q ** 2)
        )
        x = x / normalizer
        
        h = tf.random.normal(
            x.shape,
            mean=0,
            stddev=1.0
        )

        '''
        # for random SNR value, use below:
        snrdB = random.randint(0, 40)
        snr = 10 ** (snrdB / 10)
        '''
        snr = self.snr

        n = tf.random.normal(
            x.shape,
            mean=0,
            stddev=tf.math.sqrt(1/snr)
        )

        y = h * x + n

        yhat = y * normalizer
        yhat = tf.math.divide_no_nan(yhat, h)

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
            stddev=tf.math.sqrt(1/snr)
        )

        y = x + n

        yhat = y * normalizer
        return yhat
