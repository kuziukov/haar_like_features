import cv2
import numpy as np
from math import trunc
from utils.pooling import pooling


class Channels:
    def __init__(self, cell_size=6, height=120, width=60):
        self._cell_size = cell_size
        self._height = height
        self._width = width
        self._height_cells = None
        self._width_cells = None
        self._number_channels = 11
        self._number_hog_bins = 6

    def _convert_luv(self, image) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        return pooling(image,
                       pool_size=(self._height_cells, self._width_cells),
                       strides=self._cell_size)

    def _compute_gradients(self, image) -> np.dstack:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_x = np.absolute(sobel_x)
        sobel_x = np.uint8(sobel_x)
        sobel_x = pooling(sobel_x.reshape(h, w, 1),
                          pool_size=(self._height_cells, self._width_cells),
                          strides=self._cell_size)

        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_y = np.absolute(sobel_y)
        sobel_y = np.uint8(sobel_y)
        sobel_y = pooling(sobel_y.reshape(h, w, 1),
                          pool_size=(self._height_cells, self._width_cells),
                          strides=self._cell_size)
        return np.dstack((sobel_x, sobel_y))

    def _compute_hogs(self, image) -> np.ndarray:
        hog = cv2.HOGDescriptor((self._width, self._height),
                                (self._cell_size, self._cell_size),
                                (self._cell_size, self._cell_size),
                                (self._cell_size, self._cell_size),
                                self._number_hog_bins,
                                1, 4, 0, 2.0000000000000001e-01, 0, 64)
        results = hog.compute(image)
        return results.reshape(self._height_cells, self._width_cells, self._number_hog_bins)

    def compute_channels(self, image) -> np.dstack:
        if (image.shape[0] != self._width) or (image.shape[1] != self._height):
            image = cv2.resize(image, (self._width, self._height))

        # Calculate height and width of cells using image
        self._height_cells = trunc(image.shape[0] / self._cell_size)
        self._width_cells = trunc(image.shape[1] / self._cell_size)
        image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=.87)

        # Convert to LUV channels
        luv = self._convert_luv(image)

        # Compute gradients
        gradients = self._compute_gradients(image)

        # Histogram gradients
        hogs = self._compute_hogs(image)

        return np.dstack((luv, gradients, hogs))

