import tensorflow as tf
import math

MAX_ORDER = 256

class QAMModulator(tf.keras.layers.Layer):
    def __init__(self, order=256):
        '''
        Apply M-QAM modulation to given tensor.
        input shape: (Any)
        input dtype: int
        output shape: (2, Any), where first dimension denotes in-phase and quadrature-phase elements, respectively.
        output dtype: float
        '''
        super().__init__()
        assert order <= MAX_ORDER, f"Order should not be greater than {MAX_ORDER}"
        self.order = order
        self.n_bit = int(math.log2(self.order))

    def call(self, inputs):
        assert inputs.dtype in [tf.int8, tf.int16, tf.int32, tf.int64], \
               "input tensor dtype should be int"
        avg_power = math.sqrt((self.order-1)/3*2)

        # bit split
        lower_half_bit_mask = 2 ** (self.n_bit//2) - 1
        upper_half_bit_mask = lower_half_bit_mask << (self.n_bit//2)

        lower_bit = tf.bitwise.bitwise_and(inputs, lower_half_bit_mask)
        upper_bit = tf.bitwise.right_shift(tf.bitwise.bitwise_and(inputs, upper_half_bit_mask), self.n_bit//2)

        output = tf.stack([upper_bit, lower_bit])

        # to gray code
        output = tf.bitwise.bitwise_xor(output, tf.bitwise.right_shift(output, 1)) + 1

        # center to zero and power normalization
        output = tf.cast(output, tf.float32)
        output = (2 * output - 1 - self.order**0.5) / avg_power

        return output


class QAMDemodulator(tf.keras.layers.Layer):
    def __init__(self, order=256):
        '''
        Demodulate given tensor using M-QAM modulation.
        input shape: (2, Any), where in-phase and quadrature-phase elements are divided in the first dimension.
        input dtype: float
        output shape: (Any)
        output dtype: int
        '''
        super().__init__()
        assert(order < MAX_ORDER, f"Order should not be greater than {MAX_ORDER}")
        self.order = order
        self.n_bit = int(math.log2(self.order))

    def call(self, inputs):       
        assert(inputs.shape[0] == 2, "first dimension size of the given tensor should be 2 (for in-phase and quadrature-phase, respectively)")

        avg_power = tf.math.sqrt((self.order-1)/3*2)

        # QAM detection
        yhat = tf.cast(tf.math.floor(inputs * avg_power / 2) * 2 + 1, tf.int32)
        max_val = 2 * 2 ** (self.n_bit // 2) - 1 - int(self.order**0.5)
        min_val = 1 - int(self.order**0.5)
        yhat = tf.math.minimum(max_val, tf.math.maximum(yhat, min_val))

        # undo zero-centering
        yhat = (yhat + int(self.order**0.5) + 1) // 2 - 1

        yhat = tf.cast(yhat, tf.int16)

        # grey code to binary
        # assume modulation order <= 256
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 128))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 64))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 32))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 16))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 8))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 4))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 2))
        yhat = tf.bitwise.bitwise_xor(yhat, tf.bitwise.right_shift(yhat, 1))

        # bit concatenation
        output = tf.bitwise.bitwise_or(tf.bitwise.left_shift(yhat[0], (self.n_bit//2)), yhat[1])

        return output


if __name__ == '__main__':
    import time
    import numpy as np
    
    start = time.time()

    m = 256
    n_bit = int(math.log2(m))
    snrdB = 15
    num_repeat = int(1e+6)

    mod = QAMModulator(order=m)
    demod = QAMDemodulator(order=m)

    snr = 10 ** (snrdB / 10) # in dB
    sigma = 1 / math.sqrt(snr*2)
    biterror = 0

    source = np.random.uniform(low=0, high=255, size=num_repeat).astype(np.int16)
    x = mod(source)

    noise = np.random.normal(loc=0, scale=sigma, size=(2, num_repeat))
    y = tf.cast(x, tf.float32) + noise
    shat = demod(y)

    end = time.time() - start
    print(f'N: {num_repeat}, Elapsed: {end:.4f}s')

    power = tf.reduce_mean(x ** 2)
    print(f'SNR: {1/(2 * sigma**2):.4f} / Target: {snr:.4f}')
    print(f'AVG Power: {power:.4f}')
    print(f'Eb/N0: {(power)/(2 * sigma**2):.4f}')

    source_output_xor = source ^ shat
    for i in source_output_xor:
        biterror += bin(i).count("1")
    print(f'BER: {biterror / num_repeat / n_bit}')
