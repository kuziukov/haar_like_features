import pickle

import numpy as np

from modules import Templates, Features, Channels
from utils.get_feature_matrix import _get_feature_matrix
from utils.get_image_paths import _get_image_paths
from utils.make_labels import _make_labels

dataset_dir = 'dataset/INRIAPerson/train_us'
pos_lst_dir = 'pos.lst'
neg_lst_dir = 'neg.lst'
file_name = 'resources/features.p'
template_file = None

feature_number = 100


templates_generator = Templates()
templates_generator.generate_sizes()
templates = templates_generator.generate_templates()
templates = templates[:feature_number]


feature_generator = Features(templates)
channels = Channels()

pos_images, neg_images = _get_image_paths(dataset_dir, pos_lst_dir, neg_lst_dir)

print('-----> Total images to process: ', len(pos_images) + len(neg_images))
X = np.zeros((len(pos_images) + len(neg_images), 11*len(templates)))
X = _get_feature_matrix(X, pos_images, 0, channels, feature_generator)
X = _get_feature_matrix(X, neg_images, len(pos_images) - 1, channels, feature_generator)
print('-----> Obtained feature matrix with shape {}'.format(str(X.shape)))

Y = _make_labels(len(pos_images), len(neg_images))

pickle.dump({'input': X, 'labels': Y}, open(file_name, 'wb'))
print('-----> Successfully formulated and saved X and Y')


"""

cfeats = channels.compute_channels(image)
feature_vec = feature_generator.generate_features(cfeats)

feature_generator.save_feature_info()
feature_generator.save_features()

"""
