import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


def bag_of_sifts(image_paths):

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.xfeatures2d.SIFT_create()
    # des_ext = cv2.DescriptorExtractor_create("SIFT")

    # List where all the descriptors are stored
    des_list = []

    for image_path in image_paths:
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kpts, des = fea_det.detectAndCompute(gray, None)
        des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    print des_list[0][1].shape
    print des_list[5][1].shape
    for image_path, descriptor in des_list[1:]:
        try:
            if descriptor is None:
                continue
            descriptors = np.vstack((descriptors, descriptor))
        except:
            print image_path + " causes error"
            print descriptor
            print descriptors
            raise
        # Perform k-means clustering
    k = 50
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        if des_list[i][1] is None:
            print i
            continue
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    return im_features

if __name__ == "__main__":
    list_image = ['/Users/mac/Downloads/101_ObjectCategories/camera/image_0042.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0043.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0044.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0045.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0046.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0047.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0048.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0049.jpg',
                  '/Users/mac/Downloads/101_ObjectCategories/camera/image_0050.jpg']
    bag_of_sifts(list_image)