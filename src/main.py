import cv2
from modules import Channels
from modules.ImagePreprocessor import ImagePreprocessor

image = cv2.imread("images/person_064.bmp")

channels = Channels()
results = channels.compute_channels(image)


preprocessing = ImagePreprocessor()
preprocessing.preprocess_images()


