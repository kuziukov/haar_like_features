import cv2
import numpy as np
from math import trunc
from utils.pooling import pooling


class Channels:
    CELL_SIZE = 6  # pixels
    IMG_HEIGHT = 120
    IMG_WIDTH = 60
    H_CELLS = trunc(IMG_HEIGHT / CELL_SIZE)
    W_CELLS = trunc(IMG_WIDTH / CELL_SIZE)
    N_CHANNELS = 11
    NUM_HOG_BINS = 6

    def __init__(self, cell_size=6, height=120, width=60):
        self._cell_size = cell_size
        self._height = height
        self._width = width
        self._height_cells = None
        self._width_cells = None
        self._number_channels = 11
        self._number_hog_bins = 6

    def _pool(self, vol, H_cells, W_cells):
        _, _, depth = vol.shape
        feats = np.zeros((H_cells, W_cells, depth))
        for i in range(H_cells - 1):
            h_offset = i * self.CELL_SIZE
            for j in range(W_cells - 1):
                w_offset = j * self.CELL_SIZE
                subluv = vol[h_offset:h_offset + self.CELL_SIZE, w_offset:w_offset + self.CELL_SIZE, :]
            feats[i, j, :] = np.sum(subluv)

        return feats

    def _compute_luv(self, img, H_cells, W_cells):
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        feats = self._pool(luv, H_cells, W_cells)
        return feats

    def _compute_gradients(self, img, H_cells, W_cells):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_x = np.uint8(abs_sobel64f)
        sobel_x = self._pool(sobel_x.reshape(H, W, 1), H_cells, W_cells)

        sobely64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel64f = np.absolute(sobely64f)
        sobel_y = np.uint8(abs_sobel64f)
        sobel_y = self._pool(sobel_y.reshape(H, W, 1), H_cells, W_cells)
        return np.dstack((sobel_x, sobel_y))

    def _compute_hog(self, img, H_cells, W_cells):
        winSize = (W_cells * self.CELL_SIZE, H_cells * self.CELL_SIZE)
        blockSize = (self.CELL_SIZE, self.CELL_SIZE)
        blockStride = (self.CELL_SIZE, self.CELL_SIZE)
        cellSize = (self.CELL_SIZE, self.CELL_SIZE)
        nbins = self.NUM_HOG_BINS
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog_feats = hog.compute(img)
        return hog_feats.reshape(H_cells, W_cells, self.NUM_HOG_BINS)

    def compute_channels(self, img, resize=False):

        if resize:
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_H, img_W, _ = img.shape
        H_cells = trunc(img_H / self.CELL_SIZE)
        W_cells = trunc(img_W / self.CELL_SIZE)
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=.87)

        # LUV channels
        luv = self._compute_luv(img, H_cells, W_cells)

        # Gradient magnitudes in X and Y (Sobel filters)
        grads = self._compute_gradients(img, H_cells, W_cells)

        # Gradient histogram channels (HOG, 6 bins)
        hog = self._compute_hog(img, H_cells, W_cells)

        channels = np.dstack((luv, grads, hog))
        return channels

