import os
import numpy as np
import cv2

if __name__ == "__main__":
    all_classes = os.listdir("data")
    print all_classes
    all_data = range(0, 11)
    print all_data
    for class_dir in all_classes:
        path = os.path.join("data", class_dir)
        all_images_path = os.listdir(path)
        all_images = []
        for image_path in all_images_path:
            image = cv2.imread(os.path.join(path, image_path))
            all_images.append(image)
        all_data[eval(class_dir)-1] = all_images
    train_data = range(0, 11)
    test_data = range(0, 11)
    for i in xrange(0, 11):
        data = all_data[i]
        bound = int(0.6*len(data))
        train_data[i] = data[0:bound]
        test_data[i] = data[bound:]

    np.save("train.npy", train_data)
    np.save("test.npy", test_data)

