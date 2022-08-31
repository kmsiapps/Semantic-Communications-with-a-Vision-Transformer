import tensorflow as tf
import tensorflow_compression as tfc
from models.channellayer import RayleighChannel, AWGNChannel

# Source: https://github.com/kurka/deepJSCC-feedback/blob/master/jscc.py

class DeepJSCC(tf.keras.Model):
    def __init__(self, snrdB=25, channel='AWGN'):
        super().__init__()
        self.encoder = DeepJSCC_Encoder(conv_depth=16)

        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        else:
            self.channel = tf.identity

        self.decoder = DeepJSCC_Decoder(n_channels=3)
    
    def call(self, x):
        x = self.encoder(x)

        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        x = self.channel(x)

        # (2, b, c) to (b, c, 2)
        x = tf.transpose(x, (1, 2, 0))
        x = self.decoder(x)

        return x


class DeepJSCC_Encoder(tf.keras.layers.Layer):
    def __init__(self, conv_depth, name="encoder", **kwargs):
        super().__init__()
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="gdn_0"),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="gdn_1"),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="gdn_2"),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="gdn_3"),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                conv_depth,
                (5, 5),
                name="layer_out",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                # activation=None,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        
        b, h, w, c = x.shape
        x = tf.reshape(x, (-1, h*w*c//2, 2))
        return x


class DeepJSCC_Decoder(tf.keras.layers.Layer):
    def __init__(self, n_channels, name="decoder", **kwargs):
        super().__init__()
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_out",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="igdn_out", inverse=True),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                # activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            tf.nn.relu,
            # tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                n_channels,
                (9, 9),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
            ),
        ]

    def call(self, x):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c*2//64))

        for sublayer in self.sublayers:
            x = sublayer(x)
        return x
