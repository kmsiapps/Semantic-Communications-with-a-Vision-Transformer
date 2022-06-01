from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds

from config import BATCH_SIZE

def dataset_generator(dir, mode=None, shuffle=True):
    if mode:
        dataset = image_dataset_from_directory(
            directory=dir,
            label_mode='int',
            labels='inferred',
            color_mode='rgb',
            batch_size=BATCH_SIZE,
            image_size=(32, 32),
            shuffle=shuffle,
            interpolation='bilinear',
            validation_split=0.1,
            subset=mode,
            seed=0
        )
    else:
        dataset = image_dataset_from_directory(
            directory=dir,
            label_mode='int',
            labels='inferred',
            color_mode='rgb',
            batch_size=BATCH_SIZE,
            image_size=(32, 32),
            shuffle=shuffle,
            interpolation='bilinear'
        )

    return dataset
