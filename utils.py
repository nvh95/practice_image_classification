import os
import numpy as np
import cv2

def load_data(path, verbose=False, num_train=40, num_test=10, num_class=15):
    """ Load data to the memory
    :param path: path to the directory contains dataset
    :return: path and label to training and testing images
    X_train
    y_train
    X_test
    y_test
    """

    X_train =[]
    y_train = []
    X_test = []
    y_test = []
    all_classes = get_all_files(path)

    all_classes = all_classes[:num_class]

    for i, _class in enumerate(all_classes):
        class_path = os.path.join(path, _class)
        all_images_name = get_all_files(class_path)[:num_train]
        training_samples = [os.path.join(class_path, file_name) for file_name in all_images_name ]
        X_train.extend(training_samples)
        y_train += [i]*len(training_samples)

    for i, _class in enumerate(all_classes):
        class_path = os.path.join(path, _class)
        all_images_name = get_all_files(class_path)[-1*num_test:]
        testing_samples = [os.path.join(class_path, file_name) for file_name in all_images_name ]
        X_test.extend(testing_samples)
        y_test += [i]*len(testing_samples)

    if verbose:
        print "X_train" + str(X_train)
        print "y_train" + str(y_train)

    if verbose:
        print "X_test" + str(X_test)
        print "y_test" + str(y_test)

    return X_train, np.array(y_train), X_test, np.array(y_test)


def load_images(path, num_train=40, num_test=15, num_class=15):
    """Load training and testing images
    return X_train, y_train, X_test, y_test
    where X_train has form of (number_of_images, width, height, 3)
    """

    X_train, y_train, X_test, y_test = load_data(path,num_train=num_train, num_test=num_test, num_class=num_class)
    for i, sample in enumerate(X_train):
        X_train[i] = cv2.imread(X_train[i])

    for i, sample in enumerate(X_test):
        X_test[i] = cv2.imread(X_test[i])
    return X_train, y_train, X_test, y_test


def resize(list_images, size=(150,150)):
    for i, image in enumerate(list_images):
        list_images[i] = cv2.resize(image, size)
    return np.asanyarray(list_images)


def get_all_files(path):
    """Get all names of files in a directory"""
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files

if __name__ == "__main__":
    path = "/Users/mac/Downloads/101_ObjectCategories"
    # X_train, y_train, X_test, y_test = load_data(path, verbose=True)
    a = load_images(path)
    resize(a)
