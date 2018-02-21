import numpy as np
import scipy.ndimage

num_images = 800

width = 2000
height = 1500
resample_rate = 2
test = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_0001.tif")
test = test[::resample_rate, ::resample_rate]
resized = np.zeros((num_images, test.shape[0], test.shape[1]))

for index in range(num_images):
    print("Resizing image {:04d}\r".format(index+1), end="")
    try:
        im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index+1))
    except IOError:
        im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.tif".format(index+1))

    im = im[::resample_rate, ::resample_rate]
    im = im.sum(axis=2)/3
    resized[index] = im

y = np.zeros([num_images, test.shape[0], test.shape[1], 1])
y[:,:,:,0] = resized

y = y.astype("float32")

#normalize
y -= y.min()
y /= y.max()

np.save("../dataset/images.npy", y)
