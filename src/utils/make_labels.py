import numpy as np


def _make_labels(n_pos, n_neg):
    """ Takes number of positive and negative images and returns appropriate label vector """

    Y = np.zeros((n_pos + n_neg))
    Y[:n_pos] = 1

    return Y