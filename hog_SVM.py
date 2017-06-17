import cv2
import os
import numpy as np
from util import *
from skimage import feature as ft
from sklearn.decomposition import PCA
from sklearn import svm

def load_data():
    all_classes = os.listdir("data")
    print all_classes
    all_data = []
    for class_dir in all_classes:
        path = os.path.join("data", class_dir)
        all_images_path = os.listdir(path)
        all_images = []
        for image_path in all_images_path:
            image = cv2.imread(os.path.join(path, image_path))
            all_images.append(image)
        all_data[eval(class_dir) - 1] = all_images
    return all_data

def get_hog(image):
    width, height = image.shape[0], image.shape[1]
    image = image.reshape((width, height))
    return ft.hog(image, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(4, 4))

if __name__ == "__main__":
    import numpy as np
    import cv2
    from annotation import Annotation, seperate_train_val_data
    from keras.preprocessing import image as keras_image

    with_bbox = False
    color_mode = 'gray'  # or gray
    feature_center = True  # or False
    horizontal_flip = True  # or False
    vertical_flip = True  # or False
    train_val_ratio = 0.6

    d = Annotation(
        image_set='train',
        data_path="/home/aqrose/workspace/final/data/flower",
        with_bbox=with_bbox)

    x, y = d.prepare_keras_data(target_size=200, color_mode=color_mode)
    (x_train, y_train), (x_test, y_test) = seperate_train_val_data(x, y, ratio=train_val_ratio)

    datagen = keras_image.ImageDataGenerator(
        featurewise_center=feature_center,
        featurewise_std_normalization=feature_center,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )
    datagen.fit(x_train)
    datagen.fit(x_test)

    # get hog
    train_hog = []
    test_hog = []
    for train_image in x_train:
        train_hog.append(get_hog(train_image))
    for test_image in x_test:
        test_hog.append(get_hog(test_image))

    # pca
    pca = PCA(n_components=100)
    pca.fit(train_hog)
    x_train_pca = pca.transform(train_hog)
    x_test_pca = pca.transform(test_hog)
    print x_train_pca.shape

    y_test_cf = np.argmax(y_test, axis=1)
    y_train_cf = np.argmax(y_train, axis=1)
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train_pca, y_train_cf)
    result = clf.predict(x_test_pca)
    result_cf = result
    plot_confusion_matrix(y_test_cf, result_cf, range(1, 11))
    right_number = 0.0
    for index in xrange(len(result)):
        if y_test_cf[index] == result_cf[index]:
            right_number += 1.0
    print "acc:" + str(float(right_number) / float(len(result)))
