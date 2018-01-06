from __future__ import division

import scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys


def splitIntoBins(num_bins, original):
    temp = original - original.min()
    temp /= temp.max()
    temp *= num_bins
    return np.floor(temp).clip(0, num_bins - 1)

# Approach:
# 1: calculate histograms over sliding windows
# 2: recreate histograms with autoencoder (trained with all windows)
# 3: iterate
# windows with higher error are outliers and thus suspects for splicing


#take the image index as parameter
try:
    index = int(sys.argv[1])
except IndexError:
    index = 3

# Load image
try:
    im = scipy.ndimage.imread("dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index))
except IOError:
    im = scipy.ndimage.imread("dataset/dev-dataset-forged/dev_{:04d}.tif".format(index))
truth_map = scipy.ndimage.imread("dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index))
    
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
num_bins = 3
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

matched_x = np.zeros_like(filtered_x)
matched_y = np.zeros_like(filtered_y)

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        window_index = int(i / w * windows_per_row) + int(j / w)
        matched_x[i, j] = pattern_x[i, j] == np.argmax(hist_x[window_index])
        matched_y[i, j] = pattern_y[i, j] == np.argmax(hist_y[window_index])

# pooling (average), currently just for visualization
pooled = (matched_x + matched_y) / 2

plt.figure()
plt.imshow(im)
plt.title("original")

plt.figure()
plt.imshow(truth_map)
plt.title("truth_map")

plt.figure()
plt.imshow(pooled, cmap='gray')
plt.title("pooled")

plt.show()
