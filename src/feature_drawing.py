import pickle

import matplotlib.pyplot as plt

from modules import Templates, Features
import numpy as np

save_to_file = 'resources/model'

test_classification = pickle.load(open(save_to_file, 'rb'))

template_generator = Templates()
template_generator.generate_sizes()
templates = template_generator.generate_templates()


features = Features(templates)
cfeats = np.zeros((20, 10, 11))
features_desription = features.generate_features(cfeats)

feat_info = features._feature_info
top_indices = test_classification.top_indices(100)

top_templates = []
for idx in top_indices:
    top_templates.append((templates[feat_info[idx][0]], feat_info[idx][1]))

pickle.dump(top_templates, open('top_templates.p', 'wb'))


def draw_weights(clf, feature_info):
    w_viz = np.zeros((11, 20, 10))
    w_viz_total = np.zeros((20, 10))

    f = plt.gcf()
    f.set_figheight(15)
    f.set_figwidth(15)

    for idx, weight in enumerate(clf.feature_importances_):
        x, y, size, k = feature_info[idx]
        w, h = size
        w_viz[k, y:y + h, x:x + w] += weight
        w_viz_total[y:y + h, x:x + w] += weight

    for channels in range(11):
        f = plt.figure()
        W = w_viz[channels, :, :] / np.max(w_viz[channels, :, :])
        plt.matshow(W)
    plt.show()

    return w_viz_total / np.max(w_viz_total)


w_viz = draw_weights(test_classification.clf, features._feature_info)
