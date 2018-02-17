import numpy as np
import scipy.ndimage
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose

#load images and ground-truth maps
images = np.load("../dataset/extracted.npy")
maps = np.load("../dataset/maps.npy")

model = Sequential()

#Conv1
model.add(Conv2D(64, 3, input_shape=(500, 667, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool1
model.add(MaxPooling2D(strides=(2,2)))

#Conv2
model.add(Conv2D(128, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(128, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool2
model.add(MaxPooling2D(strides=(2,2)))

#Conv3
model.add(Conv2D(256, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(256, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(256, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool3
model.add(MaxPooling2D(strides=(2,2)))

#Conv4
model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool4
model.add(MaxPooling2D(strides=(2,2)))

#Conv5
model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(512, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

#MaxPool5
model.add(MaxPooling2D(strides=(2,2)))

#Conv6-7
model.add(Conv2D(4096, 7))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Conv2D(4096, 1))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Conv2D(2, 1))
print(model.layers[-1].output_shape)
model.add(UpSampling2D(size=(2,2)))
model.add(Activation("relu"))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))

model.add(UpSampling2D(size=(2,2)))
model.add(Activation("relu"))
model.add(UpSampling2D(size=(8,8)))
#model.add(Cropping2D(cropping=((0, 0), (0, 0))))
model.add(Activation("softmax"))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))
#model.add(Conv2DTranspose(2,kernel_size=(2,2)))

model.compile("SGD", loss="binary_crossentropy")

x_train = np.zeros((750, 500, 667, 1))
# x_train = images[:750]
print(x_train.shape)
x_test = images[750:]
y_train = np.zeros((750, 500, 667, 1))
# y_train = maps[:750]
y_test = maps[750:]

model.fit(x=x_train, y=y_train, batch_size=32)
