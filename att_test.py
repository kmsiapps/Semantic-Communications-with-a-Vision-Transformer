import tensorflow as tf

h = 4
x = tf.random.normal(shape=(2, h*h, 1024))
b, d, c = x.shape

learned_pos_emb = tf.range(0, (2*h-1)**2)

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
pos_emb = tf.gather(learned_pos_emb, pos_emb_idx)

q = x
k = x

# normalize with sqrt(d)
q = q / tf.sqrt(tf.constant(c, tf.float32))

# attention map computation; q, k: (b*m, h*w, d_h)
att_map = tf.einsum('bic,bjc->bij', q, k)
att_map = tf.nn.softmax(att_map)

print(att_map.shape)
print(pos_emb_idx.shape)
