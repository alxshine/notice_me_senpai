import numpy as np
import scipy.ndimage

num_images = 800

width = 2000
height = 1500
resample_rate = 2
test = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_0001.bmp")
test = test[::resample_rate, ::resample_rate]
resized = np.zeros((num_images, test.shape[0], test.shape[1]))

for index in range(num_images):
    print("Resizing image {:04d}\r".format(index+1), end="")
    im = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index+1))
    im = im[::resample_rate, ::resample_rate]
    resized[index] = im

np.save("../dataset/maps.npy", resized)
