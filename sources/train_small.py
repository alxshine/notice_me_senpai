import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Cropping2D
from keras.utils import *
from keras.models import load_model

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

# Model Information:
# model from paper is inspired by (and actually a modified version of) the VGG16 convnet
# VGG16 source code in keras: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
#

model = Sequential()

#convolution
model.add(Conv2D(18, 3, input_shape=(750, 1000, 1), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(30, 3, padding='same'))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(54, 3, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(60, 3, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

#model.add(Conv2D(256, 3, padding='same'))
#model.add(Activation("relu"))

#deconvolution
#model.add(Conv2DTranspose(256, 3, padding='same'))
#model.add(Activation("relu"))

#model.add(UpSampling2D(2))

model.add(Conv2DTranspose(60, 3, padding='same'))
model.add(Activation("relu"))

model.add(UpSampling2D(2))

model.add(Conv2DTranspose(54, 3, padding='same'))
model.add(Activation("relu"))

model.add(UpSampling2D(2))

model.add(Conv2DTranspose(30, 3, padding='same'))
model.add(Activation("relu"))

model.add(UpSampling2D(2))

model.add(Conv2DTranspose(18, 3, padding='same'))
model.add(Activation("relu"))

#model.add(UpSampling2D(2))

model.add(Cropping2D((1,0)))
model.add(Conv2DTranspose(1, 3, padding='same'))
#model.add(Conv2D(1, 3, padding='same'))
model.add(Activation("sigmoid"))

# consider changing optimization function
# cosine_proximity kinda works as well
model.compile("SGD", loss="hinge", metrics=['accuracy'])

#generate train and test sets 

num_samples = 800

x_train = np.zeros([num_samples, 750, 1000, 1])
y_train = np.zeros([num_samples, 750, 1000, 1])

x_train[:,:,:,0] = images[:num_samples,:,:]
y_train[:,:,:,0] = maps[:num_samples,:,:]


model.fit(x=x_train, y=y_train, batch_size=8, epochs=20, validation_split=0.07, shuffle=True)

model.save('small.h5')

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
