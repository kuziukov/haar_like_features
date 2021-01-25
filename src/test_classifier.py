import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from modules import Templates
from utils.formulate_stats import _formulate_stats
from utils.predict import _predict

save_to_file = 'resources/model'
feature_number = 100


test_classification = pickle.load(open(save_to_file, 'rb'))

templates_generator = Templates()
templates_generator.generate_sizes()

templates = templates_generator.generate_templates()
templates = templates[:feature_number]


raw_img_files = open('dataset/INRIAPerson/test_us/pos.lst')
pos_img_paths = ['dataset/INRIAPerson/test_us/pos/' + path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

raw_img_files = open('dataset/INRIAPerson/test_us/neg.lst')
neg_img_paths = ['dataset/INRIAPerson/test_us/neg/' + path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

n_pos = len(pos_img_paths)
n_neg = len(neg_img_paths)

print('-----> test ', str(n_pos + n_neg), ' images')

Y_pos = np.ones((len(pos_img_paths)))
pos_accuracy = accuracy_score(Y_pos, _predict(test_classification, pos_img_paths, templates))

print('-----> pos accuracy: ', pos_accuracy)

# =====[ Get negative image accuracy ]=====
Y_neg = np.zeros((len(neg_img_paths)))
neg_accuracy = accuracy_score(Y_neg, _predict(test_classification, neg_img_paths, templates))

print('-----> neg accuracy: ', neg_accuracy)

accuracy, f1 = _formulate_stats(n_pos, n_neg, pos_accuracy, neg_accuracy)

print('acc: ', accuracy)
print('Score: ', f1)

