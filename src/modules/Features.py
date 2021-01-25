import json

import numpy as np


class Features:

    def __init__(self, templates):
        self._features = None
        self._templates = templates
        self._feature_info = []

    def generate_features(self, block):
        self._features = []
        self._feature_info = []

        _, _, k_channels = block.shape

        for id, t in enumerate(self._templates):

            x, y, size, W = t
            w, h = size

            for k in range(k_channels):
                cell_block = np.copy(block[y:y + h, x:x + w, k])
                self._features.append(np.sum(np.multiply(cell_block, W)))
                #self._feature_info.append((x, y, size, W))
                self._feature_info.append((x, y, size, k))

        return self._features
