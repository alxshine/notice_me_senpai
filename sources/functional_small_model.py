import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Cropping2D, ThresholdedReLU
from keras.utils import *
from keras.models import load_model
from keras import optimizers

model_path = 'small_functional.h5'

#load images and ground-truth maps
print("Loading training images...")
images = np.load("../dataset/extracted.npy")
print("Done, min: {}, max: {}, mean: {}".format(images.min(), images.max(), images.mean()))
print("Loading ground truth maps...")
maps = np.load("../dataset/maps.npy")
print("Done, min: {}, max: {}, mean: {}".format(maps.min(), maps.max(), maps.mean()))

# Model Information:
# model from paper is inspired by (and actually a modified version of) the VGG16 convnet
# VGG16 source code in keras: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
#
# Functional Model Input
#
input_img = Input(shape=(750, 1000, 1))

x = Conv2D(18, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(input_img)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(30, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(54, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(60, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

# x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(60, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(54, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(30, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(18, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)

x = Cropping2D((1,0))(x)
y = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal')(x)

y = ThresholdedReLU(theta=0.5)(y)

model = Model(inputs=input_img, outputs=y)

opt = optimizers.SGD()
print("Compiling model...")
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
print("Done, fitting...")

model.fit(images, maps, epochs=2, batch_size=3, validation_split=0.07, shuffle=True)

print("Done, saving model to {}...".format(model_path))
model.save(model_path)
print("Done")

prediction = model.predict(images[6:7])

plt.figure()
plt.subplot(131)
plt.imshow(images[6,:,:,0])
plt.title("image")

plt.subplot(132)
plt.imshow(maps[6,:,:,0])
plt.title("map")

plt.subplot(133)
plt.imshow(prediction[0,:,:,0])
plt.title("prediction")

plt.show()
