import tensorflow as tf
from abc import *

from models.resblock import ResBlock
from models.vitblock import VitBlock
from models.channellayer import RayleighChannel, AWGNChannel


class SemVit(tf.keras.Model):
    def __init__(self, block_types, filters, num_blocks, dim_per_head,
                 num_symbols, papr_reduction='clip', snrdB=25, channel='Rayleigh'):
        '''
        block_types: types of each building blocks
            'V' for ViT block, 'C' for Conv (ResNet) block
            e.g., ['C', 'C', 'V', 'V', 'V', 'C', 'C']
        filters: output dimensions for each block
            e.g., [64, 96, 192, 384, 96, 64, 32]
        num_blocks: # of repetition for each block
            e.g., [2, 2, 3, 2, 5, 2, 2]
        num_symbols: # of total complex symbols sent
        papr: type of papr-reduction methods. 'clip' for clipping, None for nothing
        snrdB: channel snr (in dB)
        channel: channel type ('Rayleigh' or 'AWGN')
        '''
        assert len(filters) == len(num_blocks) == len(block_types) == 7, \
               "block_types, filters and num_blocks must have same size (7)"

        super().__init__()

        self.encoder = SemVit_Encoder(
            block_types[:4], filters[:4], num_blocks[:4], dim_per_head, num_symbols, papr_reduction=papr_reduction)

        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        else:
            self.channel = tf.identity

        self.decoder = SemVit_Decoder(
            block_types[4:], filters[4:], num_blocks[4:], dim_per_head)

    
    def call(self, x):
        x = self.encoder(x)
        # (b, c, iq(=2)) to (2, b, c) 
        x = tf.transpose(x, (2, 0, 1))
        x = self.channel(x)

        # (2, b, c) to (b, c, 2)
        x = tf.transpose(x, (1, 2, 0))
        x = self.decoder(x)

        return x


class SemVit_Encoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks, dim_per_head, num_symbols, papr_reduction):
        super().__init__()
        # assume 32 x 32 input
        self.l0 = build_blocks(0, block_types, num_blocks, filters, dim_per_head, 32, downsample=False)
        # no downsampling here
        
        self.l1 = build_blocks(1, block_types, num_blocks, filters, dim_per_head, 32)
        # downsampled to 16 x 16

        self.l2 = build_blocks(2, block_types, num_blocks, filters, dim_per_head, 16)
        # downsampled to 8 x 8

        self.l3 = build_blocks(3, block_types, num_blocks, filters, dim_per_head, 8)
        # downsampled to 4 x 4

        self.to_constellation = tf.keras.layers.Conv2D(
            filters=num_symbols // 16 * 2,
            # current spatial dimension is 4 x 4
            # and 2 for iq dimension
            kernel_size=1
        )

        self.papr_reduction = papr_reduction
    
    def call(self, x):
        b, h, w, c = x.shape

        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.to_constellation(x)
        x = tf.reshape(x, (b, -1, 2))
        # x = self.dense(x)

        # TODO: polar coordinate로 바꿔서 잘 clipping? 지금은 사각형이라 꼭짓점 쪽이 파워가 더 큼
        if self.papr_reduction == 'clip':
            i = x[:, :, 0]
            q = x[:, :, 1]
            avg_power = tf.reduce_mean(i ** 2 + q ** 2)
            PAPR_CONST = 1.2
            max_amp = tf.sqrt(avg_power * PAPR_CONST)
            x = tf.clip_by_value(x, -1 * max_amp, max_amp)

        # # power normalization
        # i = x[:, :, 0]
        # q = x[:, :, 1]
        # avg_power = tf.reduce_mean(i ** 2 + q ** 2)
        # x = x / tf.sqrt(avg_power)

        return x


class SemVit_Decoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks, dim_per_head):
        super().__init__()

        # assume 8 x 8 input (up-sampled from 4x4, 96, 2)
        self.l4 = build_blocks(0, block_types, num_blocks, filters, dim_per_head, 8, downsample=False)
        
        # upsampled to 16 x 16
        self.l5 = build_blocks(1, block_types, num_blocks, filters, dim_per_head, 16, downsample=False)
        
        # upsampled to 32 x 32
        self.l6 = build_blocks(2, block_types, num_blocks, filters, dim_per_head, 32, downsample=False)

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
        x = self.l4(x)

        x = tf.image.resize(x, (16, 16)) # bilinear upsampling
        # x = self.vit4(x)
        x = self.l5(x)

        x = tf.image.resize(x, (32, 32))
        x = self.l6(x)

        x = self.to_image(x)
        return x


def build_blocks(layer_idx, block_types, num_blocks, filters, dim_per_head, spatial_size, downsample=True):
    if block_types[layer_idx] == 'C':
        return build_resblocks(
            repetition=num_blocks[layer_idx],
            filter_size=filters[layer_idx],
            kernel_size=5,
            downsample=downsample)
    else:
        return build_vitblocks(
            repetition=num_blocks[layer_idx],
            num_heads=filters[layer_idx]//dim_per_head,
            head_size=dim_per_head,
            spatial_size=spatial_size,
            downsample=downsample)


def build_resblocks(repetition, filter_size, kernel_size=5,
                    downsample=True):
    init_stride = 2 if downsample else 1
    x = tf.keras.Sequential()
    x.add(ResBlock(filter_size, stride=init_stride, kernel_size=kernel_size, is_bottleneck=False))
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
