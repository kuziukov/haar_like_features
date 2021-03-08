import pickle

from modules import Templates, Features
from modules.exp_detector import Detector
from utils.draw_bbs import _draw_bbs

save_to_file = 'resources/model_old'
load_templates = 'resources/top_templates.p'
feature_number = 1000
scaling_factor = 1.2
scaling_iters = 3
nms = 0.5
output_file_prefix = ''
img_path = 'images/crop_000012.png'

# crop_000012.png
# person_014.png
# person_036.png
# person_064.bmp

test_classification = pickle.load(open(save_to_file, 'rb'))

templates = pickle.load(open(load_templates, 'rb'))
print(templates)
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
