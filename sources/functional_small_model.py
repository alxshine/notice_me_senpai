import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Cropping2D
from keras.utils import *
from keras.models import load_model

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

num_samples = 800

x_train = np.zeros([num_samples, 750, 1000, 1])
y_train = np.zeros([num_samples, 750, 1000, 1])

x_train[:,:,:,0] = images[:num_samples,:,:]
y_train[:,:,:,0] = maps[:num_samples,:,:]

# Model Information:
# model from paper is inspired by (and actually a modified version of) the VGG16 convnet
# VGG16 source code in keras: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
#
# Functional Model Input
#
input_img = Input(shape=(750, 1000, 1))

x = Conv2D(18, (3, 3), padding='same', activation='relu')(input_img)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(30, (3, 3), padding='same', activation='relu')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(54, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

x = Conv2D(60, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

#x = UpSampling2D((2,2))
x = Conv2DTranspose(60, (3, 3), padding='same', activation='relu')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(54, (3, 3), padding='same', activation='relu')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(30, (3, 3), padding='same', activation='relu')(x)

x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(18, (3, 3), padding='same', activation='relu')(x)

x = Cropping2D((1,0))(x)
y = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)

model = Model(inputs=input_img, outputs=y)

model.compile(optimizer='SGD', loss='hinge', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=8, validation_split=0.07, shuffle=True)

model.save('small_functional.h5')

prediction = model.predict(x_train[6:7])

plt.figure()
plt.subplot(131)
plt.imshow(x_train[6,:,:,0])
plt.title("image")

plt.subplot(132)
plt.imshow(y_train[6,:,:,0])
plt.title("map")

plt.subplot(133)
plt.imshow(prediction[0,:,:,0])
plt.title("prediction")

plt.show()
