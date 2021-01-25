import cv2

from modules import Channels, Features


def _predict(clf, img_paths, templates):
    """
        1) Generates templates
        2) Extracts channel features
        3) Genereates feature vectors
        4) classifies image
    """

    cf = Channels()
    fg = Features(templates)
    feature_vectors = []

    print('-----> test %d total images' % (len(img_paths)))
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)

        cfeats = cf.compute_channels(img)
        feature_vectors.append(fg.generate_features(cfeats))

        if idx % 500 == 0:
            print('-----> process ', idx + 1)

    print('-----> completed all feature vectors')

    ys = clf.clf.predict(feature_vectors)

    return ys