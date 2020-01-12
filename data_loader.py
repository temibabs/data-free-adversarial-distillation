import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.datasets import mnist


def get_dataset(batch_size, mode='train'):
    train, test = mnist.load_data()

    if mode == 'train':
        return get_data(train, batch_size)

    else:
        return get_data(test, batch_size)


def get_data(train, batch_size):
    x = np.reshape(train[0], (train[0].shape[0],
                              train[0].shape[1],
                              train[0].shape[2], 1)).astype(np.float64)
    y = train[1].astype(np.float64)
    return tf.data.Dataset.from_tensor_slices(
        (x / 255.0, y)).shuffle(10000).padded_batch(
        batch_size,
        padded_shapes=(
            [32, 32, 1],
            []))
