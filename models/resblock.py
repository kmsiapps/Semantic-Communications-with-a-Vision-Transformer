import tensorflow as tf

# Simple ResBlock for CIFAR-10
# See original paper's section 4.2 (https://arxiv.org/pdf/1512.03385.pdf)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, stride=1, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.proj = tf.keras.layers.Conv2D(filters=filter_size,
                                           kernel_size=1,
                                           strides=stride,
                                           padding="same")
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_size,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters=filter_size,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()


    def call(self, x, training=False, **kwargs):
        x_residual = self.proj(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x + x_residual
        x = tf.nn.relu(x)

        return x
