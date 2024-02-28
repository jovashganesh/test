import numpy as np
import glob 
import tensorflow as tf
from skimage.io import imread, imsave  
from skimage.transform import resize  
from sklearn.model_selection import train_test_split
import re
import pandas
from PIL import Image
import keras,os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

image_size = 224
RGB_channel = 3

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

image_list_train = sorted(glob.glob('/user/HS224/jg00980/Desktop/AI/train/*.jpg'), key = natural_keys)
image_list_test = sorted(glob.glob('/user/HS224/jg00980/Desktop/AI/test/*.jpg'), key = natural_keys)

new_list_train = sorted(glob.glob('/user/HS224/jg00980/Desktop/AI/train/*.jpg'), key = natural_keys)
new_list_test = sorted(glob.glob('/user/HS224/jg00980/Desktop/AI/test/*.jpg'), key = natural_keys)

x_train = np.empty((len(image_list_train), image_size, image_size, RGB_channel), dtype=np.uint8)
x_test = np.empty((len(image_list_test), image_size, image_size, RGB_channel), dtype=np.uint8)

# GET X_TRAIN VALUES - Unshrinked
def get_x_train():
    for i, img_path in enumerate(image_list_train):
        print("Processing train image", i)
        img = imread(img_path)
        img = resize(img, output_shape=(image_size, image_size, RGB_channel), preserve_range=True)
        new_path = "/user/HS224/jg00980/Desktop/AI/224/train/{}.jpg".format(i)
        imsave(new_path, img)

# GET X_TRAIN VALUES SHRINKED
def get_x_train_shrinked():
    for i, img_path in enumerate(new_list_train):
        print("Importing processed train image", i)
        x_train[i] = imread(img_path)

# GET X_TEST VALUES - Unshrinked
def get_x_test():
    for i, img_path in enumerate(image_list_test):
        print("Processing test image", i)
        img = imread(img_path)
        img = resize(img, output_shape=(image_size, image_size, RGB_channel), preserve_range=True)
        new_path = "/user/HS224/jg00980/Desktop/AI/224/test/{}.jpg".format(i)
        imsave(new_path, img)

# GET X_TEST VALUES SHRINKED
def get_x_test_shrinked():
    for i, img_path in enumerate(new_list_test):
        print("Importing processed test image", i)
        x_test[i] = imread(img_path)

# CODE TO GET THE Y_TRAIN VALUES
y = []
with open('/user/HS224/jg00980/Desktop/AI/train.txt', 'r') as fileobj:
    for row in fileobj:
        numbers = row.partition(" ")
        y.append(numbers[2].rstrip('\n'))

get_x_train()
get_x_test()

# get_x_train_shrinked()
# get_x_test_shrinked()


x_train = x_train.astype("uint8")/255
x_test = x_test.astype("uint8")/255

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

y_train = np.array(y)
y_train = to_categorical(y_train, 23)

gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.25, height_shift_range=0.25, rescale=1./255, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3)

VGG16 = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet", classes=23)
VGG16.trainable = False

model = keras.Sequential([
    VGG16,
    keras.layers.Dropout(0.5),
    keras.layers.GlobalAveragePooling2D(name="gap"),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=4096, activation="relu"),
    keras.layers.Dense(units=4096, activation="relu"),
    keras.layers.Dense(units=23, activation="softmax", activity_regularizer=l1(0.001))
])

gen.fit(x_train)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)

model.fit(gen.flow(x_train, y_train, batch_size=64),
          epochs=50,
          validation_data=(x_validation, y_validation),
          steps_per_epoch = len(x_train)//64,
          shuffle=True)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

data = pandas.DataFrame({"label":y_pred})
data.to_csv("y_pred.csv", index = False)
