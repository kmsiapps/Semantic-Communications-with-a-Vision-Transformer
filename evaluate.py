import tensorflow as tf
import os
import csv

from config import FILTERS, NUM_BLOCKS, DIM_PER_HEAD, DATA_SIZE, BATCH_SIZE
from models.model import VitCommNet, VitCommNet_Encoder_Only, CnnCommNet
from models.qam_model import QAMModem
from utils.datasets import dataset_generator

# Reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

@tf.function
def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image


test_ds = dataset_generator('/dataset/CIFAR100/test/')

loss_object = tf.keras.losses.MeanSquaredError()
test_loss = tf.keras.metrics.Mean(name='test_loss')

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
'''
model = CnnCommNet(
  FILTERS,
  NUM_BLOCKS,
  DATA_SIZE,
  snrdB=25,
  channel='Rayleigh'
)
'''
model = VitCommNet(
  FILTERS,
  NUM_BLOCKS,
  DIM_PER_HEAD,
  DATA_SIZE,
  snrdB=25,
  channel='Rayleigh'
)

model.build(input_shape=(1,32,32,3))
model.summary()

################## CONFIG ####################
best_model = './model_checkpoints/data-256.ckpt'
QAM_ORDER = 256
##############################################

for channelname in ['Rayleigh', 'AWGN']:
    f = open(f'./results/{channelname.lower()}_results.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['SNR', 'PropSSIM', 'PropMSE', 'PropPSNR', 'QAMSSIM', 'QAMMSE', 'QAMPSNR'])

    if not os.path.isdir('./results'):
        os.mkdir('./results')
    
    if not os.path.isdir(f'./results/{channelname}/'):
        os.mkdir(f'./results/{channelname}/')
    
    for EVAL_SNRDB in [0, 2.69, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        qam_modem = QAMModem(snrdB=EVAL_SNRDB, order=QAM_ORDER, channel=channelname)
        '''
        model = CnnCommNet(
            FILTERS,
            NUM_BLOCKS,
            DATA_SIZE,
            snrdB=EVAL_SNRDB,
            channel=channelname
            )
        '''
        model = VitCommNet(
            FILTERS,
            NUM_BLOCKS,
            DIM_PER_HEAD,
            DATA_SIZE,
            snrdB=EVAL_SNRDB,
            channel=channelname
            )

        model.load_weights(best_model)

        i = 0
        ssim_props = 0; ssim_qams = 0
        mse_props = 0; mse_qams = 0
        psnr_props = 0; psnr_qams = 0
        for images, _ in test_ds:
            int16_images = tf.cast(images * 255, tf.int16)
            qam_results = tf.cast(qam_modem(int16_images), tf.float32) / 255
            prop_results = model(images, training=False)
            
            ssim_props += tf.reduce_sum(tf.image.ssim(images, prop_results, max_val=1.0))
            ssim_qams += tf.reduce_sum(tf.image.ssim(images, qam_results, max_val=1.0))

            mse_props += tf.reduce_mean((images - prop_results) ** 2) * BATCH_SIZE
            mse_qams += tf.reduce_mean((images - qam_results) ** 2) * BATCH_SIZE

            psnr_props += tf.reduce_sum(tf.image.psnr(images, prop_results, max_val=1.0))
            psnr_qams += tf.reduce_sum(tf.image.psnr(images, qam_results, max_val=1.0))
            
            if i == 0:
                tf.keras.utils.save_img(f'./results/{channelname}/original_SNR{EVAL_SNRDB}.png', imBatchtoImage(images))
                tf.keras.utils.save_img(f'./results/{channelname}/proposed_SNR{EVAL_SNRDB}.png', imBatchtoImage(prop_results))
                tf.keras.utils.save_img(f'./results/{channelname}/256qam_SNR{EVAL_SNRDB}.png', imBatchtoImage(qam_results))
            
            if i == 9:
                break

            i += 1


        total_images = i * BATCH_SIZE
        ssim_props /= total_images
        ssim_qams /= total_images
        mse_props /= total_images
        mse_qams /= total_images
        psnr_props /= total_images
        psnr_qams /= total_images
        
        print(f'Channel: {channelname} / SNR: {EVAL_SNRDB}dB =======================================')
        print(f'SSIM: (Proposed){ssim_props:.6f} vs. (QAM){ssim_qams:.6f}')
        print(f'MSE:  (Proposed){mse_props:.6f} vs. (QAM){mse_qams:.6f}')
        print(f'PSNR:  (Proposed){psnr_props:.6f} vs. (QAM){psnr_qams:.6f}')

        writer.writerow([EVAL_SNRDB, float(ssim_props), float(mse_props), float(psnr_props), float(ssim_qams), float(mse_qams), float(psnr_qams)])
    '''
    # Layer-wise image
    images, _ = next(iter(test_ds))

    for model_subclass in [VitCommNet, VitCommNet_Encoder_Only]:
        model = model_subclass(
            FILTERS,
            NUM_BLOCKS,
            DIM_PER_HEAD,
            DATA_SIZE,
            snrdB=10,
            channel=channelname
            )
        model.load_weights(best_model)
        result = model(images[:1, :, :, :])
        output = result
        
        if model_subclass.__name__ != 'VitCommNet':
            # For intermediate layers, flatten channels and normalize to grayscale images
            output = output / tf.reduce_max(output, axis=-1, keepdims=True)
            output = tf.transpose(output, (3, 0, 1, 2))
            output = tf.reshape(output, (-1, 32, 1))
        else:
            output = tf.reshape(output, (-1, 32, 3))
        tf.keras.utils.save_img(f'./results/{channelname}/{model_subclass.__name__}_SNR10.png', output)

        if model_subclass.__name__ == 'VitCommNet_Encoder_Only':
            result.numpy().tofile(f'./results/constellation.bin')

    tf.keras.utils.save_img(f'./results/{channelname}/E2E_before_SNR10.png',
                            images[0, :, :, :])
    '''
