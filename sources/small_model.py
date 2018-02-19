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
# Import saved model 
#
print("Loading model...")
model = load_model('small.h5')

def image_test(imageNo):
	imageNo -= 1
	prediction = model.predict(images[imageNo])
	plt.figure()
	plt.subplot(131)
	plt.imshow(images[imageNo])
	plt.title("image")

	plt.subplot(132)
	plt.imshow(maps[imageNo])
	plt.title("map")

	plt.subplot(133)
	plt.imshow(prediction)
	plt.title("prediction")

	plt.show()

image_test(5)
