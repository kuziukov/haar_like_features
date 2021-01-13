import cv2
from modules import Channels

image = cv2.imread("images/person_064.bmp")

channels = Channels()
results = channels.compute_channels(image)

print(results)