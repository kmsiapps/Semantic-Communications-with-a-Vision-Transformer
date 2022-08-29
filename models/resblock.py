import tensorflow as tf

# Reference: https://github.com/calmisential/TensorFlow2.0_ResNet/

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, stride=1, kernel_size=5, is_bottleneck=True):
        super().__init__()
        self.bottleneck = is_bottleneck
        self.stride = stride
        henorm = tf.keras.initializers.HeNormal()

        self.conv1 = tf.keras.layers.Conv2D(filters = filter_size,
                                            kernel_size = (1, 1),
                                            strides = 1,
                                            padding="same",
                                            kernel_initializer=henorm)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters = filter_size,
                                            kernel_size = (kernel_size, kernel_size),
                                            strides = stride,
                                            padding="same",
                                            kernel_initializer=henorm)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if self.bottleneck:
            self.relu = lambda x: tf.nn.relu(x)
            self.conv3 = tf.keras.layers.Conv2D(filters = filter_size * 4,
                                                kernel_size = (1, 1),
                                                strides = 1,
                                                padding="same",
                                                kernel_initializer=henorm)
            self.bn3 = tf.keras.layers.BatchNormalization()
        else:
            self.relu = lambda x: x
            self.conv3 = lambda x: x
            self.bn3 = lambda x, **kwargs: x

        self.projection = tf.keras.Sequential()
        ds_filter_size = filter_size * 4 if self.bottleneck else filter_size
        self.projection.add(tf.keras.layers.Conv2D(filters = ds_filter_size,
                                                    kernel_size = (1, 1),
                                                    strides = stride,
                                                    padding="same",
                                                    kernel_initializer=henorm))
        self.projection.add(tf.keras.layers.BatchNormalization())


    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Note: for non-bottleneck resblock,
        # next three functions literally does nothing
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        shortcut = self.projection(inputs)
        x = tf.keras.layers.add([x, shortcut])
        output = tf.nn.relu(x)

        return output
