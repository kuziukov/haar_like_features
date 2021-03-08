import pickle
import glob
from modules import Templates, Features
from modules.exp_detector import Detector
from utils.draw_bbs import _draw_bbs

save_to_file = 'resources/model_best_1000'
feature_number = 1000
scaling_factor = 1.08
scaling_iters = 8
nms = 0.5
output_file_prefix = ''
img_path = 'images/crop001504.png'

images = glob.glob("images/*.png")

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


for image in images:
    detector = Detector(test_classification.clf,
                        feature_generator,
                        scaling_factor=scaling_factor,
                        scaling_iters=scaling_iters,
                        nms=nms)


    _, bbs = detector.detect_pedestrians(image)
    _draw_bbs(image, bbs)
