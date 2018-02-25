import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Cropping2D, ThresholdedReLU
from keras.utils import *
from keras.models import load_model
from keras import optimizers
import tensorflow as tf

def pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
    return - tf.reduce_sum(target * tf.log(output))

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

num_samples = 800

x_train = np.zeros([num_samples,750,1000,1])
y_train = np.zeros([num_samples,750,1000,1])

x_train[:,:,:,:] = images[:num_samples,:,:,:]
y_train[:,:,:,:] = maps[:num_samples,:,:,:]

input_img = Input(shape=(750, 1000, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Cropping2D((1,0))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss=pixelwise_crossentropy,metrics=['accuracy'])

autoencoder.fit(x_train, y_train,
                epochs=2,
                batch_size=1,
                shuffle=True)

pred = autoencoder.predict(x_train[11:12])

plt.figure()
plt.subplot(131)
plt.imshow(x_train[11,:,:,0])
plt.title("image")

plt.subplot(132)
plt.imshow(y_train[11,:,:,0])
plt.title("map")

plt.subplot(133)
plt.imshow(pred[0,:,:,0])
plt.title("prediction")

plt.show()
