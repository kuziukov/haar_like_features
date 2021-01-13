import numpy as np


def pooling(obj, pool_size=(20, 10), strides=6) -> np.ndarray:
    _, _, channels = obj.shape
    block = np.zeros((pool_size[0], pool_size[1], channels))

    for i in range(pool_size[0]):
        height_offset = i * strides
        for j in range(pool_size[1]):
            width_offset = j * strides
            sub = obj[height_offset: height_offset + strides,
                  width_offset: width_offset + strides, :]
            block[i, j, :] = np.sum(sub)
    return block