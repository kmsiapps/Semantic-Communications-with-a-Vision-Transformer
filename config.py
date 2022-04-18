BATCH_SIZE = 64
EPOCHS = 600

# ViTComm parameters
# each for ResNet0, ResNet1, Vit2, Vit3, (Channel), ViT4, ResNet5, ResNet6
FILTERS = [64, 96, 192, 384, 128, 64, 32]
NUM_BLOCKS = [2, 2, 3, 2, 7, 4, 4]
DIM_PER_HEAD = 32
SPATIAL_SIZE = 32
DATA_SIZE = 2048

TRAIN_SNRDB = 25

# 32 * 32 * 3 // 5 * (256 // 64) // 16 // DIM_PER_HEAD * DIM_PER_HEAD * 16
# 32 x 32 x 3 image, 0~255 pixel level, 1/5 compression ratio, 64-QAM target
# and make it evenly divisible to DIM_PER_HEAD and 16 (4 x 4 image)
# ~ 2048 (1/6 compression ratio)

PAPR_LAMBDA = 1e-5
PWR_LAMBDA = 1e-4