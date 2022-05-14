import tensorflow as tf
from abc import *

from models.resblock import ResBlock
from models.vitblock import VitBlock
from models.channellayer import RayleighChannel, AWGNChannel
from utils.qam_modem_tf import QAMModulator, QAMDemodulator


class VitCommNet(tf.keras.Model):
    def __init__(self, filters, num_blocks, dim_per_head,
                 data_size, snrdB=25, channel='Rayleigh'):
        '''
        filters: output features for 4 consecutive blocks
                (ResNet0, ResNet1, Vit2, Vit3, ViT4, ResNet5, ResNet6)
                e.g., [64, 96, 192, 384, 96, 64, 32]
        num_blocks: num of repetition for each block
                e.g., [2, 2, 3, 2, 5, 2, 2]
        dim_per_head: used for MHSA in Vit blocks (e.g., 32)
        data_size: # of symbol sent
        snrdB: SNR (in dB) value for channel simulation
        channel: 'Rayleigh' or 'AWGN'
        '''
        assert len(filters) == 7, 'len(filters) != 7'
        assert len(num_blocks) == 7, 'len(num_blocks) != 7'
        super().__init__()

        self.encoder = VitCommNet_Encoder(
            filters[:4], num_blocks[:4], dim_per_head, data_size)

        self.channel = RayleighChannel(snrdB) if channel=='Rayleigh' \
                       else AWGNChannel(snrdB)

        self.decoder = VitCommNet_Decoder(
            filters[4:], num_blocks[4:], dim_per_head)

        
    def call(self, x):
        x = self.encoder(x)
        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        x = self.channel(x)

        # (2, b, c) to (b, c, 2)
        x = tf.transpose(x, (1, 2, 0))
        x = self.decoder(x)

        return x


class VitCommNet_Encoder_Only(tf.keras.Model):
    def __init__(self, filters, num_blocks, dim_per_head,
                 data_size, snrdB=25, channel='Rayleigh'):
        super().__init__()

        self.encoder = VitCommNet_Encoder(
            filters[:4], num_blocks[:4], dim_per_head, data_size)
        
    def call(self, x):
        x = self.encoder(x)
        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        return x


class VitCommNet_Decoder_Only(tf.keras.Model):
    def __init__(self, filters, num_blocks, dim_per_head,
                 data_size, snrdB=25, channel='Rayleigh'):
        super().__init__()
        self.decoder = VitCommNet_Decoder(
            filters[4:], num_blocks[4:], dim_per_head)

    def call(self, x):
        x = tf.transpose(x, (1, 2, 0))
        x = self.decoder(x)

        return x


class VitCommNet_Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, num_blocks, dim_per_head,
                 data_size):
        super().__init__()
        # assume 32 x 32 input
        self.res0 = build_resblocks(
            repetition=num_blocks[0],
            filter_size=filters[0],
            downsample=False)
        # no downsampling here
    
        self.res1 = build_resblocks(
            repetition=num_blocks[1],
            filter_size=filters[1])
        # downsampled to 16 x 16

        self.vit2 = build_vitblocks(
            repetition=num_blocks[2],
            num_heads=filters[2]//dim_per_head,
            head_size=dim_per_head,
            spatial_size=16)
        # downsampled to 8 x 8

        self.vit3 = build_vitblocks(
            repetition=num_blocks[3],
            num_heads=filters[3]//dim_per_head,
            head_size=dim_per_head,
            spatial_size=8)
        # downsampled to 4 x 4

        self.to_constellation = tf.keras.layers.Conv2D(
            filters=data_size // 16 * 2,
            # current spatial dimension is 4 x 4
            # and 2 for iq dimension
            kernel_size=1
        )
    
    def call(self, x):
        b, h, w, c = x.shape

        x = self.res0(x)
        x = self.res1(x)
        x = self.vit2(x)
        x = self.vit3(x)
        x = self.to_constellation(x)
        x = tf.reshape(x, (b, -1, 2))
        # x = self.dense(x)

        return x


class VitCommNet_Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, num_blocks, dim_per_head):
        super().__init__()

        # assume 8 x 8 input (up-sampled from 4x4, 96, 2)
        self.vit4 = build_vitblocks(
            repetition=num_blocks[0],
            num_heads=filters[0]//dim_per_head,
            head_size=dim_per_head,
            spatial_size=8,
            downsample=False)
        
        # upsampled to 16 x 16
        self.res5 = build_resblocks(
            repetition=num_blocks[1],
            filter_size=filters[1],
            downsample=False)
        
        '''
        # assume 8 x 8 input (up-sampled from 4x4, 96, 2)
        self.res5 = build_resblocks(
            repetition=num_blocks[0],
            filter_size=filters[0],
            downsample=False)
        
        # upsampled to 16 x 16
        self.vit4 = build_vitblocks(
            repetition=num_blocks[1],
            num_heads=filters[1]//dim_per_head,
            head_size=dim_per_head,
            spatial_size=16,
            downsample=False)
        '''
        
        # upsampled to 32 x 32
        self.res6 = build_resblocks(
            repetition=num_blocks[2],
            filter_size=filters[2],
            downsample=False)

        self.to_image = tf.keras.layers.Conv2D(
            kernel_size=1,
            filters=3,
            padding='same',
            activation='sigmoid'
        )
    
    def call(self, x):
        # x = self.dense(x)
        
        b = x.shape[0]
        x = tf.reshape(x, (b, 4, 4, -1))
        
        x = tf.image.resize(x, (8, 8))
        # x = self.res5(x)
        x = self.vit4(x)

        x = tf.image.resize(x, (16, 16)) # bilinear upsampling
        # x = self.vit4(x)
        x = self.res5(x)

        x = tf.image.resize(x, (32, 32))
        x = self.res6(x)

        x = self.to_image(x)
        return x


class CnnCommNet(tf.keras.Model):
    def __init__(self, filters, num_blocks,
                 data_size, snrdB=25, channel='Rayleigh'):
        '''
        filters: output features for 4 consecutive blocks
                (ResNet0, ResNet1, Vit2, Vit3, ViT4, ResNet5, ResNet6)
                e.g., [64, 96, 192, 384, 96, 64, 32]
        num_blocks: num of repetition for each block
                e.g., [2, 2, 3, 2, 5, 2, 2]
        dim_per_head: used for MHSA in Vit blocks (e.g., 32)
        data_size: # of symbol sent
        snrdB: SNR (in dB) value for channel simulation
        channel: 'Rayleigh' or 'AWGN'
        '''
        assert len(filters) == 7, 'len(filters) != 7'
        assert len(num_blocks) == 7, 'len(num_blocks) != 7'
        super().__init__()

        self.encoder = CnnCommNet_Encoder(
            filters[:4], num_blocks[:4], data_size)

        self.channel = RayleighChannel(snrdB) if channel=='Rayleigh' \
                       else AWGNChannel(snrdB)

        self.decoder = CnnCommNet_Decoder(
            filters[4:], num_blocks[4:])
        
    def call(self, x):
        x = self.encoder(x)
        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        x = self.channel(x)

        # (2, b, c) to (b, c, 2)
        x = tf.transpose(x, (1, 2, 0))
        x = self.decoder(x)

        return x


class CnnCommNet_Encoder_Only(tf.keras.Model):
    def __init__(self, filters, num_blocks,
                 data_size, snrdB=25, channel='Rayleigh'):
        super().__init__()

        self.encoder = CnnCommNet_Encoder(
            filters[:4], num_blocks[:4], data_size)
        
    def call(self, x):
        x = self.encoder(x)
        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        return x


class CnnCommNet_Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, num_blocks, data_size):
        super().__init__()
        # assume 32 x 32 input
        self.res0 = build_resblocks(
            repetition=num_blocks[0],
            filter_size=filters[0],
            downsample=False)
        # no downsampling here
    
        self.res1 = build_resblocks(
            repetition=num_blocks[1],
            filter_size=filters[1])
        # downsampled to 16 x 16

        self.res2 = build_resblocks(
            repetition=num_blocks[2],
            filter_size=filters[2])
        # downsampled to 8 x 8

        self.res3 = build_resblocks(
            repetition=num_blocks[3],
            filter_size=filters[3])
        # downsampled to 4 x 4

        self.to_constellation = tf.keras.layers.Conv2D(
            filters=data_size // 16 * 2,
            # current spatial dimension is 4 x 4
            # and 2 for iq dimension
            kernel_size=1
        )
    
    def call(self, x):
        b, h, w, c = x.shape

        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.to_constellation(x)
        x = tf.reshape(x, (b, -1, 2))
        # x = self.dense(x)

        return x


class CnnCommNet_Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, num_blocks):
        super().__init__()

        # assume 8 x 8 input (up-sampled from 4x4, 96, 2)
        self.res4 = build_resblocks(
            repetition=num_blocks[0],
            filter_size=filters[0],
            downsample=False)
        
        # upsampled to 16 x 16
        self.res5 = build_resblocks(
            repetition=num_blocks[1],
            filter_size=filters[1],
            downsample=False)
        
        # upsampled to 32 x 32
        self.res6 = build_resblocks(
            repetition=num_blocks[2],
            filter_size=filters[2],
            downsample=False)

        self.to_image = tf.keras.layers.Conv2D(
            kernel_size=1,
            filters=3,
            padding='same',
            activation='sigmoid'
        )
    
    def call(self, x):
        # x = self.dense(x)
        
        b = x.shape[0]
        x = tf.reshape(x, (b, 4, 4, -1))
        
        x = tf.image.resize(x, (8, 8))
        x = self.res4(x)

        x = tf.image.resize(x, (16, 16)) # bilinear upsampling
        x = self.res5(x)

        x = tf.image.resize(x, (32, 32))
        x = self.res6(x)

        x = self.to_image(x)
        return x


def build_resblocks(repetition, filter_size,
                    downsample=True):
    init_stride = 2 if downsample else 1
    x = tf.keras.Sequential()
    x.add(ResBlock(filter_size, stride=init_stride, is_bottleneck=False))
    for _ in range(repetition-1):
        x.add(ResBlock(filter_size, is_bottleneck=False))
    return x


def build_vitblocks(repetition,
                    num_heads, head_size,
                    spatial_size, downsample=True):
    x = tf.keras.Sequential()
    x.add(VitBlock(num_heads, head_size,
                    spatial_size, downsample))
    
    if downsample:
        spatial_size //= 2
    for _ in range(repetition-1):
        x.add(VitBlock(num_heads, head_size, spatial_size))
    return x
