from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from annotation import Annotation, seperate_train_val_data
from keras.preprocessing import image as keras_image
import keras

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(200,200,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



# with_bbox = False
# color_mode = 'rgb'  # or gray
# feature_center = True  # or False
# horizontal_flip = True  # or False
# vertical_flip = False  # or False
# train_val_ratio = 0.6
#
# d = Annotation(
#     image_set='train',
#     data_path="/home/aqrose/workspace/final/data/flower",
#     with_bbox=with_bbox)
#
# x, y = d.prepare_keras_data(target_size=200, color_mode=color_mode)
# (x_train, y_train), (x_test, y_test) = seperate_train_val_data(x, y, ratio=train_val_ratio)
#
# datagen = keras_image.ImageDataGenerator(
#     featurewise_center=feature_center,
#     featurewise_std_normalization=feature_center,
#     horizontal_flip=horizontal_flip,
#     vertical_flip=vertical_flip
# )
# datagen.fit(x_train)
# datagen.fit(x_test)
# this is the augmentation configuration we will use for training
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(200, 200),  # all images will be resized to 150x150
        batch_size=32)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(200, 200),
        batch_size=32)

model.fit_generator(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
steps_per_epoch=32,
validation_steps=32
)


model.save_weights('first_try.h5')  # always save your weights after training or during training