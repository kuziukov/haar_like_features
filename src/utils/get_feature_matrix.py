import cv2


def _get_feature_matrix(X, images, offset=0, channels=None, feature_generator=None):
    """ Append feature vectors for each training example in images to X """

    # =====[ Iterate through images and calculate feature vector for each ]=====
    for idx, img in enumerate(images):

        try:
            cfeats = channels.compute_channels(cv2.imread(img))
            feature_vec = feature_generator.generate_features(cfeats)

            # =====[ Add feature vector to input matrix ]=====
            X[idx + offset, :] = feature_vec

        except Exception as e:
            print(e)
            print('Could not add image at index: ', idx + offset)

    return X