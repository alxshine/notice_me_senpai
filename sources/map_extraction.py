import scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_images = 800
    width = 2000
    height = 1500
    #resample_rate = 3
    test = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_0001.bmp")
    #test = test[::resample_rate, ::resample_rate]
    extracted = np.zeros((800, test.shape[0], test.shape[1]))

    im = test.sum()/3/test.max()

    plt.figure(3)
    plt.imshow(test)
    plt.show()
    print(test.shape)

    for index in range(num_images):
        print("Extracting features from image {:04d}\r".format(index), end="")
        # Load image
        try:
            im = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index+1))
        except IOError:
            im = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.tif".format(index+1))

        #im = im.sum() / 3 / im.max()
        #im = im[::resample_rate,::resample_rate]

        extracted[index] = im

    np.save("../dataset/maps.npy", extracted)