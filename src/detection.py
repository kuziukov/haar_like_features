import pickle

from modules import Templates, Features
from modules.exp_detector import Detector
from utils.draw_bbs import _draw_bbs

save_to_file = 'resources/model'
feature_number = 100
scaling_factor = 1
scaling_iters = 3
nms = 0.1
offset = 0
output_file_prefix = ''
img_path = 'images/photo_2021-01-15_01-11-56.jpg'


test_classification = pickle.load(open(save_to_file, 'rb'))

templates_generator = Templates()
templates_generator.generate_sizes()

templates = templates_generator.generate_templates()
templates = templates[:feature_number]

feature_generator = Features(templates)


detector = Detector(test_classification.clf,
                    feature_generator,
                    scaling_factor=scaling_factor,
                    scaling_iters=scaling_iters,
                    nms=nms)

if img_path:
    _, bbs = detector.detect_pedestrians(img_path)
    _draw_bbs(img_path, bbs)