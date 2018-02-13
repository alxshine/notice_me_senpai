from __future__ import division

from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load image
im = Image.open("dataset/dev-dataset-forged/dev_0001.tif")
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

# the paper does not clearly explain how their truncation works,
# we just use three different bins
num_bins = 2
filtered_x = np.floor(filtered_x * num_bins).astype(int)
filtered_y = np.floor(filtered_y * num_bins).astype(int)

print("Image binned in x and y, searching for patterns")

# search most occuring pattern in x direction
patternDict_x = {}
for i in range(filtered_x.shape[0] - 3):
    for j in range(filtered_x.shape[1]):
        pattern = ''.join(str(elem) for elem in filtered_x[i:i + 4, j])
        try:
            patternDict_x[pattern] += 1
        except KeyError:
            patternDict_x[pattern] = 1

maxCount_x = 0
targetPattern_x = ""
for pattern, count in patternDict_x.items():
    if count > maxCount_x:
        maxCount_x = count
        targetPattern_x = pattern

matching_x = np.zeros_like(filtered_x)
for i in range(matching_x.shape[0] - 3):
    for j in range(matching_x.shape[1]):
        pattern = ''.join(str(elem) for elem in filtered_x[i:i + 4, j])
        if pattern == targetPattern_x:
            matching_x[i, j] = 1

print("X direction matched")

# same for y direction
patternDict_y = {}
for i in range(filtered_y.shape[0]):
    for j in range(filtered_y.shape[1] - 3):
        pattern = ''.join(str(elem) for elem in filtered_y[i, j:j + 4])
        try:
            patternDict_y[pattern] += 1
        except KeyError:
            patternDict_y[pattern] = 1

# TODO: extract into function
maxCount_y = 0
targetPattern_y = ""
for pattern, count in patternDict_y.items():
    if count > maxCount_y:
        maxCount_y = count
        targetPattern_y = pattern
# print("Most common pattern is {}, occured {} times".format(targetPattern_y, maxCount_x))

matching_y = np.zeros_like(filtered_y)
for i in range(matching_y.shape[0]):
    for j in range(matching_y.shape[1] - 3):
        pattern = ''.join(str(elem) for elem in filtered_y[i, j:j + 4])
        if pattern == targetPattern_y:
            matching_y[i, j] = 1
        
print("Y direction matched")

#TODO: actually test which is better
#average pooling
matching_pooled = ((matching_x + matching_y).astype(float) / 2)
#OR pooling
# matching_pooled = np.logical_or(matching_x, matching_y)

# resample and combine areas
n = 2
resample_kernel = np.ones((n,n))/n**2
resampled = signal.convolve2d(matching_pooled, resample_kernel, mode='valid')  

plt.figure()
plt.imshow(matching_x)
plt.title("X")

plt.figure()
plt.imshow(matching_y)
plt.title("Y")

plt.figure()
plt.imshow(matching_pooled)
plt.title("Pooled")

plt.figure()
plt.imshow(resampled)
plt.title("resampled")
plt.show()
