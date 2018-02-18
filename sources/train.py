import numpy as np
import scipy.ndimage
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Reshape

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

# Model Information:
# model from paper is inspired by (and actually a modified version of) the VGG16 convnet
# VGG16 source code in keras: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
#

model = Sequential()

#Conv1
#model.add(Reshape((1500,2000,1),input_shape=(1500, 2000)))
model.add(Conv2D(64, 3, input_shape=(1500, 2000, 1),padding='same'))
#model.add(Conv2D(64, 3,padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool1
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

#Conv2
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool2
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

#Conv3
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool3
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

# possible skip connection with 2 (1,1) kernel conv followed by cropping to 2nd addition layer

#Conv4
model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool4
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

# possible skip connection with 2 (1,1) kernel conv followed by cropping to 1st addition layer

#Conv5
model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool5
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

#Conv6-7
model.add(Conv2D(4096, 7, padding='same'))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Conv2D(4096, 1, padding='same'))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Conv2D(2, 1, padding='same'))
print(model.layers[-1].output_shape)

#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))

model.add(UpSampling2D(2))
model.add(Activation("relu"))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))

model.add(UpSampling2D(size=(2,2)))
model.add(Activation("relu"))
model.add(UpSampling2D(size=(8,8)))

# addition layer 1
# deconvolution (2x)
# addition layer 2
# deconvolution (8x)
# cropping (8x)

#model.add(Cropping2D(cropping=((0, 0), (0, 0))))
#model.add(Activation("softmax"))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))

model.compile("SGD", loss="binary_crossentropy", metrics=['accuracy'])

#x_train = np.zeros((750, 500, 667, 1))
#x_train = np.zeros([750, 1500, 2000, 1])
x_train = images[:750]
x_train = np.reshape(x_train, (750,1500,2000,1))
print(x_train.shape)

x_test = np.zeros((50, 1500, 2000, 1))
#x_test = images[750:]
print(x_test.shape)

#y_train = np.zeros((750, 500, 667, 1))
#y_train = np.zeros([750, 1500, 2000, 1])
y_train = maps[:750]
y_train = np.reshape(y_train, (750,1500,2000,1))
print(y_train.shape)

y_test = np.zeros((50, 1500, 2000, 1))
#y_test = maps[750:]
print(y_test.shape)

model.fit(x=x_train, y=y_train, batch_size=32)
