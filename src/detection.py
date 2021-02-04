import pickle

from modules import Templates, Features
from modules.exp_detector import Detector
from utils.draw_bbs import _draw_bbs

save_to_file = 'resources/model'
feature_number = 1000
scaling_factor = 1.05
scaling_iters = 8
nms = 0.5
output_file_prefix = ''
img_path = 'images/11.png'

# crop_000012.png
# person_014.png
# person_036.png
# person_064.bmp

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
