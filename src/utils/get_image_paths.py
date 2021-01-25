from os import path


def _get_image_paths(base_dir, pos_filename, neg_filename):
    """ Get list of image paths in base_dir from each file_name """

    with open(path.join(base_dir, pos_filename)) as f:
        pos_list = f.readlines()
        pos_list = [base_dir + '/pos/' + x.strip() for x in pos_list]
    with open(path.join(base_dir, neg_filename)) as f:
        neg_list = f.readlines()
        neg_list = [base_dir + '/neg/' + x.strip() for x in neg_list]

    print('-----> Loaded {} positive image paths and {} negative image paths'.format(str(len(pos_list)),
                                                                               str(len(neg_list))))
    return pos_list, neg_list