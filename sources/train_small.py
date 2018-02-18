import sys
import numpy as np
import scipy.ndimage
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Cropping2D

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

# Model Information:
# model from paper is inspired by (and actually a modified version of) the VGG16 convnet
# VGG16 source code in keras: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
#

model = Sequential()

#convolution
model.add(Conv2D(64, 3, input_shape=(1500, 2000, 1), padding='same'))
model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

# model.add(Conv2D(128, 3, padding='same'))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

# model.add(Conv2D(256, 3, padding='same'))
# model.add(Activation("relu"))

#deconvolution
# model.add(Conv2DTranspose(256, 3, padding='same'))
# model.add(Activation("relu"))

# model.add(UpSampling2D(2))

# model.add(Conv2DTranspose(128, 3, padding='same'))
# model.add(Activation("relu"))

# model.add(UpSampling2D(2))

model.add(Conv2DTranspose(64, 3, padding='same'))
model.add(Activation("relu"))

model.add(Conv2DTranspose(1, 3, padding='same'))
model.add(Activation("sigmoid"))

model.compile("SGD", loss="binary_crossentropy", metrics=['accuracy'])

#generate train and test sets 

x_train = np.zeros([750, 1500, 2000, 1])
x_test = np.zeros((50, 1500, 2000, 1))
y_train = np.zeros([750, 1500, 2000, 1])
y_test = np.zeros((50, 1500, 2000, 1))

model.fit(x=x_train, y=y_train, batch_size=1)
