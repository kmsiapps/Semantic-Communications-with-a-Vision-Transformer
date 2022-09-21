import tensorflow as tf
import tensorflow_compression as tfc
from models.channellayer import RayleighChannel, AWGNChannel

# Modified from DeepJSCC codes (https://github.com/kurka/deepJSCC-feedback/blob/master/jscc.py)

class SemViT_Debug(tf.keras.Model):
    def __init__(self, block_types, filters, num_blocks, has_gdn=True,
                 num_symbols=512, snrdB=25, channel='AWGN'):
        '''
        block_types: (list) types of each building blocks
            'V' for ViT block, 'C' for Conv (ResNet) block
            e.g., ['C', 'C', 'V', 'V', 'C', 'C']
        filters: (list) output dimensions for each block
            e.g., [256, 256, 256, 256, 256, 256]
        num_blocks: (list) # of repetition for each block
            e.g., [1, 1, 3, 3, 1, 1]
        has_gdn: (bool) include GDN/IGDN?
        num_symbols: (int) # of total complex symbols sent
            e.g., 512 for 1/6 bandwidth ratio (512 / 32*32*3)
        snrdB: (int) channel snr (in dB)
        channel: (str) channel type ('Rayleigh', 'AWGN', or None)
        '''
        super().__init__()
        if has_gdn:
            gdn_func=tfc.layers.GDN()
            igdn_func=tfc.layers.GDN(inverse=True)
        else:
            gdn_func=tf.keras.layers.Lambda(lambda x: x)
            igdn_func=tf.keras.layers.Lambda(lambda x: x)

        assert len(block_types) == len(filters) == len(num_blocks) == 6, \
               "length of block_types, filters, num_blocks should be 6"
        self.encoder = SemViT_Encoder(
            block_types[:3],
            filters[:3],
            num_blocks[:3],
            num_symbols,
            gdn_func=gdn_func
        )

        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        else:
            self.channel = tf.identity

        self.decoder = SemViT_Decoder(
            block_types[3:],
            filters[3:],
            num_blocks[3:],
            gdn_func=igdn_func
        )
    
    def call(self, x):
        x, enc_out, enc_att, enc_pos = self.encoder(x)
        x = self.channel(x)
        x, dec_out, dec_att, dec_pos = self.decoder(x)

        return x, enc_out + dec_out, enc_att + dec_att, enc_pos + dec_pos


class SemViT_Encoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks,
                 num_symbols, gdn_func=None, **kwargs):
        super().__init__()
        self.layers = [
            # 32 x 32 input
            build_blocks(0, block_types, num_blocks, filters, 32, kernel_size=9, stride=2, gdn_func=gdn_func),
            # downsampled to 16 x 16
            build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, stride=2, gdn_func=gdn_func),
            # downsampled to 8 x 8
            build_blocks(2, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
            # to constellation
            tf.keras.layers.Conv2D(
                filters=num_symbols // 8 // 8 * 2,
                # current spatial dimension is 8 x 8
                # and 2 for iq dimension
                kernel_size=1
            )
        ]

    def call(self, x):
        layer_outputs = []
        att_maps = []
        pos_embs = []
        layer_outputs.append(x)

        for sublayer in self.layers:
            x = sublayer(x)
            if isinstance(x, tuple) and len(x) == 3:
                x, att_map, pos_emb = x
                att_maps.append(att_map)
                pos_embs.append(pos_emb)
            layer_outputs.append(x)
        
        b, h, w, c = x.shape
        x = tf.reshape(x, (-1, h*w*c//2, 2))
        return x, layer_outputs, att_maps, pos_embs



class SemViT_Decoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks, gdn_func=None, **kwargs):
        super().__init__()
        self.layers = [
            # 8 x 8 input
            build_blocks(0, block_types, num_blocks, filters, 8, kernel_size=5, gdn_func=gdn_func),
            # upsampled to 16 x 16
            tf.keras.layers.Resizing(16, 16),
            build_blocks(1, block_types, num_blocks, filters, 16, kernel_size=5, gdn_func=gdn_func),
            # upsampled to 32 x 32
            tf.keras.layers.Resizing(32, 32),
            build_blocks(2, block_types, num_blocks, filters, 32, kernel_size=9, gdn_func=gdn_func),
            # to image
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=1,
                activation='sigmoid'
            )
        ]


    def call(self, x):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c*2//64))

        layer_outputs = []
        att_maps = []
        pos_embs = []
        layer_outputs.append(x)

        for sublayer in self.layers:
            x = sublayer(x)
            if isinstance(x, tuple) and len(x) == 3:
                x, att_map, pos_emb = x
                att_maps.append(att_map)
                pos_embs.append(pos_emb)
            layer_outputs.append(x)

        return x, layer_outputs, att_maps, pos_embs


def build_blocks(layer_idx, block_types, num_blocks, filters, spatial_size, kernel_size=5, stride=1, gdn_func=None):
    assert block_types[layer_idx] in ('C', 'V'), "layer type should be either C or V"

    if block_types[layer_idx] == 'C':
        return build_conv(
            repetition=num_blocks[layer_idx],
            filter_size=filters[layer_idx],
            kernel_size=kernel_size,
            stride=stride,
            gdn_func=gdn_func)
    else:
        return build_vitblocks(
            repetition=num_blocks[layer_idx],
            num_heads=filters[layer_idx]//32,
            head_size=32,
            spatial_size=spatial_size,
            stride=stride,
            gdn_func=gdn_func)


def build_conv(repetition, filter_size, kernel_size=5, stride=1, gdn_func=None):
    x = tf.keras.Sequential()
    for i in range(repetition):
        s = stride if i == 0 else 1
        x.add(tfc.SignalConv2D(
                filter_size,
                kernel_size,
                corr=True,
                strides_down=s,
                padding="same_zeros",
                use_bias=True,
        ))
        if gdn_func:
            x.add(gdn_func)
        x.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    return x


def build_vitblocks(repetition, num_heads, head_size, spatial_size, stride=1, gdn_func=None):
    x = tf.keras.Sequential()
    for i in range(repetition):
        s = stride if i == 0 else 1
        x.add(VitBlock(num_heads, head_size, spatial_size, stride=s))
        if gdn_func:
            x.add(gdn_func)
    return x


import tensorflow as tf

class MLP(tf.keras.layers.Layer):
    def __init__(self, out_features, expansion_coeff=4):
        super().__init__()

        self.fc1 = tf.keras.layers.Dense(
            out_features * expansion_coeff
        )
        self.gelu = tf.nn.gelu
        self.fc2 = tf.keras.layers.Dense(
            out_features
        )
    
    def call(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class RelativeMHSA(tf.keras.layers.Layer):
    '''
    Implements multihead attention 
    with Swin-like learnable 2d relative positional encoding
    '''
    def __init__(self, num_heads, dim_head, spatial_size):
        '''
        num_heads: the number of heads
        dim_head: channel dimensions per head
        spatial_size: height/width of the input
        query/key/value shape: (b, h, w, c) where h == w 
        '''
        super().__init__()

        assert num_heads != 0, "num_heads should be nonzero"

        self.dim_head = dim_head
        self.num_heads = num_heads

        self.qkv = tf.keras.layers.Conv2D(
            filters=dim_head * 3,
            kernel_size=1
        )

        self.head_transform = tf.keras.layers.Conv2D(
            filters=dim_head*num_heads,
            kernel_size=1
        )

        # build rel. pos parameter and bias index here
        h = spatial_size
        pos_emb_idx_horizontal = tf.tile(tf.constant(
            [range(i, i+h) for i in range(0, -h, -1)]),
            multiples=[h, h]
        )

        pos_emb_idx_vertical = tf.repeat(
            tf.repeat(
                tf.constant([range(i, i+h)
                             for i in range(0, -h, -1)]),
                repeats=h,
                axis=0
            ),
            repeats=h,
            axis=-1
        )

        pos_emb_idx = (2*h-1) * (pos_emb_idx_vertical + h - 1) + \
                      (pos_emb_idx_horizontal + h - 1)

        self.pos_emb_idx = pos_emb_idx

        initializer = tf.keras.initializers.GlorotNormal()
        self.learned_pos_emb = tf.Variable(
            initializer(shape=((2*h-1)**2,))
        )


    def call(self, x):
        b, h, w, c = x.shape
        m = self.num_heads

        assert c % m == 0, "channel dimension should be divisible " \
               f"with number of heads, but c={c} and m={m} found"
        d_h = c//m

        # [b, h, w, c] to [b, m, h, w, c//m]
        x = tf.reshape(x, (-1, h, w, m, d_h))
        x = tf.transpose(x, (0, 3, 1, 2, 4))

        x = self.qkv(x)
        x = tf.reshape(x, (-1, h*w, self.dim_head, 3))
        q = x[:, :, :, 0]
        k = x[:, :, :, 1]
        v = x[:, :, :, 2]

        # normalize with sqrt(d)
        q = q / tf.sqrt(tf.constant(self.dim_head, tf.float32))

        # attention map computation; q, k: (b*m, h*w, d_h)
        att_map = tf.einsum('bic,bjc->bij', q, k)

        # add rel. pos. encoding to attention map
        pos_emb = tf.gather(self.learned_pos_emb, self.pos_emb_idx)

        att_map = att_map + pos_emb
        att_map = tf.nn.softmax(att_map)
        
        v = tf.reshape(v, (-1, h*w, self.dim_head))
        v = tf.einsum('bij,bjc->bic', att_map, v)

        # [b, m, h, w, c//m] to [b, h, w, c]
        v = tf.reshape(v, (-1, m, h, w, c//m))
        v = tf.transpose(v, (0, 2, 3, 1, 4))
        v = tf.reshape(v, (-1, h, w, c))

        v = self.head_transform(v)
        return v, att_map, pos_emb


class VitBlock(tf.keras.layers.Layer):
    # for debug
    def __init__(self, num_heads, head_size,
                 spatial_size, stride=1):
        '''
        num_heads: the number of heads
        head_size: channel dimensions per head
        spatial_size: height/width of the input
                      (before downsampling)
        patchmerge: (boolean) 1/2 downsampling before MHSA
        '''
        super().__init__()

        d_out = num_heads * head_size
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.patchmerge = tf.keras.layers.Conv2D(
            filters=d_out,
            kernel_size=stride,
            strides=stride,
        )
        spatial_size //= stride

        self.mhsa = RelativeMHSA(
            num_heads=num_heads,
            dim_head=head_size,
            spatial_size=spatial_size
        )

        self.ln2 = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(d_out)

    def call(self, x):
        att_map = []
        pos_emb = []
        if isinstance(x, tuple) and len(x) == 3:
            x, att_map, pos_emb = x

        x = self.patchmerge(x)
        x = self.ln1(x)
        x_residual = x

        x, att, pos = self.mhsa(x)
        att_map.append(att)
        pos_emb.append(pos)

        x = tf.add(x, x_residual)
        
        x_residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = tf.add(x, x_residual)

        return x, att_map, pos_emb
