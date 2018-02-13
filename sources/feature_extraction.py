from __future__ import division

import scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys

from keras.layers import Input, Dense, Dropout
from keras.models import Model


def splitIntoBins(num_bins, original):
    temp = original - original.min()
    temp /= temp.max()
    temp *= num_bins
    return np.floor(temp).clip(0, num_bins - 1)

# Approach:
# 1: calculate histograms over sliding windows
# 2: recreate histograms with autoencoder_x (trained with all windows)
# 3: iterate
# windows with higher error are outliers and thus suspects for splicing


# take the image index as parameter
try:
    index = int(sys.argv[1])
except IndexError:
    index = 1

# Load image
try:
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index))
except IOError:
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.tif".format(index))
truth_map = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index))
    
arr = np.array(im)
arr = arr.sum(axis=2) / 3 / arr.max()

# define the horizontal filter
kernel = np.array([1, -3, 3, -1])
filtered_x = np.zeros_like(arr)
filtered_y = np.zeros_like(arr)

for i in range(arr.shape[0]):
    filtered_x[i] = signal.convolve(arr[i], kernel, mode='same')
    
for j in range(arr.shape[1]):
    filtered_y[:, j] = signal.convolve(arr[:, j], kernel, mode='same') 

# binning
num_bins = 2
if len(sys.argv) >= 3:
    num_bins = int(sys.argv[2])

filtered_x = splitIntoBins(num_bins, filtered_x)
filtered_y = splitIntoBins(num_bins, filtered_y)

# search co-occurences in windows
w = 4
num_hist_bins = num_bins ** w
pattern_kernel = num_bins ** np.arange(w)
pattern_x = np.zeros_like(filtered_x)
pattern_y = np.zeros_like(filtered_y)

# this is part of the original paper
for j in range(filtered_x.shape[1]):
    pattern_x[:, j] = signal.convolve(filtered_x[:, j], pattern_kernel, mode='same')
    
for i in range(filtered_y.shape[0]):
    pattern_y[i] = signal.convolve(filtered_y[i], pattern_kernel, mode='same')
    
windows_per_row = int(arr.shape[0] / w)
windows_per_col = int(arr.shape[1] / w)
hist_x = np.zeros((windows_per_row * windows_per_col, num_hist_bins))
hist_y = np.zeros_like(hist_x)

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        window_index = int(i / w * windows_per_row) + int(j / w)
        hist_x[window_index][int(pattern_x[i, j])] += 1
        hist_y[window_index][int(pattern_y[i, j])] += 1
        
# normalize and make zero-mean
hist_x = hist_x / 81 - 0.5
hist_y = hist_y / 81 - 0.5

pattern_pooles = ((pattern_x + pattern_y).astype(float) / 2)
n = 4
resample_kernel = np.ones((n,n))/n**2
resampled = signal.convolve2d(pattern_pooles, resample_kernel, mode='valid') 

# create autoencoders
h = int(num_hist_bins * 1.5)

encoder_input = Input(shape=(num_hist_bins,))
encoded = Dense(h, activation='relu')(encoder_input)
dropout = Dropout(0.5)(encoded)
decoded = Dense(num_hist_bins, activation='sigmoid')(dropout)

autoencoder_x = Model(encoder_input, decoded)
autoencoder_x.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder_y = Model(encoder_input, decoded)
autoencoder_y.compile(optimizer='adadelta', loss='binary_crossentropy')

# train encoder with image
num_epochs = 12
train_sample_rate = 0.8

# generate training and test set
rand_set = np.random.rand(windows_per_col * windows_per_row)
training_set_x = hist_x[np.where(rand_set <= train_sample_rate)]
test_set_x = hist_x[np.where(rand_set > train_sample_rate)]
training_set_y = hist_y[np.where(rand_set <= train_sample_rate)]
test_set_y = hist_y[np.where(rand_set > train_sample_rate)]

autoencoder_x.fit(training_set_x, training_set_x, epochs=num_epochs, batch_size=256, shuffle=True, validation_data=(test_set_x, test_set_x))
autoencoder_y.fit(training_set_y, training_set_y, epochs=num_epochs, batch_size=256, shuffle=True, validation_data=(test_set_y, test_set_y))

predictions_x = autoencoder_x.predict(hist_x, batch_size=256, verbose=False)
predictions_y = autoencoder_y.predict(hist_y, batch_size=256, verbose=False)
# calculate absolute prediction error
abs_error_x = np.sum(np.abs(predictions_x - hist_x), axis=1).reshape((windows_per_row, windows_per_col))
abs_error_y = np.sum(np.abs(predictions_y - hist_y), axis=1).reshape((windows_per_row, windows_per_col))

abs_error = abs_error_x * 0.5 + abs_error_y * 0.5

plt.figure(figsize=(18, 9))
plt.subplot(131)
plt.imshow(im)
plt.title("original")
 
 
plt.subplot(132)
plt.imshow(resampled)
plt.title("pooled pattern")

plt.subplot(133)
plt.imshow(abs_error)
plt.title("Error per window")

plt.show()
