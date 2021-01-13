import cv2
from modules import Templates, Features, Channels

feature_number = 100
dataset_dir = None
template_file = None


image = cv2.imread("images/person_064.bmp")

templates_generator = Templates()
templates_generator.generate_sizes()
templates = templates_generator.generate_templates()


feature_generator = Features(templates)
channels = Channels()


cfeats = channels.compute_channels(image)
feature_vec = feature_generator.generate_features(cfeats)

feature_generator.save_feature_info()
feature_generator.save_features()



