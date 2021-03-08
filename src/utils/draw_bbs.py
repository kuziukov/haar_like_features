import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _draw_bbs(img_path, bbs):
    """ Draws bounding boxes on specified image and displays it """

    img = cv2.imread(img_path)
    #scale = img.shape[0]/200.0

    #img = imutils.resize(img, height=300)
    scale = 1

    for box in bbs:
        cv2.rectangle(img, (int(box[2] * scale), int(box[1] * scale)),
                      (int((box[2] + box[4]) * scale), int((box[1] + box[3]) * scale)), (0, 255, 0), 2)
        #cv2.putText(img, str(box[0]), (int(box[2]*scale), int(box[1]*scale)),
        #                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Prepend detected to image
    if '/' in img_path:
        path = 'detected_'+img_path.split('/')[1]
        cv2.imwrite(path,img)
        print(path)
    else:
        path = 'detected_' + img_path
        cv2.imwrite(path, img)
        print(path)

    to_show = mpimg.imread(path)
    plt.imshow(to_show)


def _draw_bbs_save(img_path, bbs):
    """ Draws bounding boxes on specified image and displays it """

    img = cv2.imread(img_path)
    scale = 1

    for box in bbs:
        cv2.rectangle(img, (int(box[2] * scale), int(box[1] * scale)),
                      (int((box[2] + box[4]) * scale), int((box[1] + box[3]) * scale)), (0, 255, 0), 2)
        cv2.putText(img, str(box[0]), (int(box[2]*scale), int(box[1]*scale)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img