# http://www.ippatsuman.com/2014/08/13/day-and-night-an-image-classifier-with-scikit-learn/

import cv2
import numpy as np

def _histogram(img_path, hist_size=64):
    """ Return histogram of a single image
    :param img_path:
    :return: return histogram of an image
    """
    img = cv2.imread(img_path)

    hist_blue = cv2.calcHist([img], [0], None, [hist_size], [0, 256])
    hist_green = cv2.calcHist([img], [1], None, [hist_size], [0, 256])
    hist_red = cv2.calcHist([img], [2], None, [hist_size], [0, 256])
    return ((hist_blue + hist_green + hist_red)/img.size).flatten()


def histogram(img_list, hist_size=64):
    """ Return histogram of a list images
    :param img_list: List of image
    :return: return list of histogram
    """
    hist_list = []
    for img in img_list:
        hist_list.append(_histogram(img, hist_size))
    return np.array(hist_list)

if __name__ == "__main__":
    list_image = ['/Users/mac/Downloads/101_ObjectCategories/camera/image_0042.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0043.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0044.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0045.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0046.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0047.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0048.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0049.jpg', '/Users/mac/Downloads/101_ObjectCategories/camera/image_0050.jpg']
    a = histogram(list_image)
    print a
    print len(a)