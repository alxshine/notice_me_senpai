import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt

print("loading features")
features = np.load("../dataset/correlations.npy")
print("done")
print("loading maps")
maps = np.load("../dataset/maps.npy")
print("done")

index = 5

im = features[index]
im[np.isnan(im)] = im[0,0]
im = np.abs(im)
m = maps[index,:,:,0]

#do stuff here
for i in range(30):
    im = scipy.ndimage.gaussian_filter(im, 0.6)

n = 3
mean_kernel_small = np.ones([n,n])/n**2

n = 7
mean_kernel_large = np.ones([n,n])/n**2

for i in range(50):
    im = scipy.signal.convolve2d(im, mean_kernel_small, mode='same')
    im = scipy.signal.convolve2d(im, mean_kernel_large, mode='same')

mean = im.mean()
print("Mean: {}".format(mean))
result = np.zeros_like(im)
result[im<mean] = 1
# result = im

plt.figure()
plt.subplot(121)
plt.imshow(result, cmap='gray')
plt.title("result")

plt.subplot(122)
plt.imshow(m)
plt.title("map")

plt.show()
