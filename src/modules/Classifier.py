import pickle

import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from modules import Channels, Features, Templates


class Classifier:
    def __init__(self, n_estimators=None, max_depth=None, clf=None):
        if clf:
            self.clf = pickle.load(open(clf, 'rb')).clf
        else:
            # =====[ Initialize new classifier ]=====
            self.clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)

    def train(self, X, Y):
        """ Trains classifier and prints average cross validation score """
        self.clf.fit(X, Y)

    def predict(self, img_paths):
        """
            Tests classifier against given image img_paths

            1) Generates templates
            2) Extracts channel features
            3) Genereates feature vectors
            4) classifies image
        """

        template_generator = Templates()
        template_generator.generate_sizes()
        templates = template_generator.generate_templates()

        # =====[ Instantiate Channel Features ]=====
        cf = Channels()

        # =====[ Instantiate FeatureGenerator ]=====
        fg = Features(templates)

        # =====[ Will hold our generated feature vectors ]=====
        feature_vectors = []

        print('-----> Testing %d total images' % (len(img_paths)))
        for idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            # =====[ Extract channel features from images and make feature vector ]=====
            cfeats = cf.compute_channels(img)
            feature_vectors.append(fg.generate_features(cfeats))

            if idx % 100 == 0:
                print('-----> Processing Image ', idx + 1)

        print('-----> Processed all feature vectors')

        # =====[ predict class for each feature_vector ]=====
        ys = self.clf.predict(feature_vectors)

        return ys

    def top_indices(self, n):
        top_ft = self.clf.feature_importances_.argsort()
        return top_ft[::-1][:n] if n else top_ft[::-1]
