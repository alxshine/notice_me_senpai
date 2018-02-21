import scipy.ndimage
import numpy as np
from scipy import signal

def splitIntoBins(num_bins, original):
    temp = original - original.min()
    temp /= temp.max()
    temp *= num_bins
    return np.floor(temp).clip(0, num_bins - 1)

def spliceDetection(im, num_bins = 3):

    # define the horizontal filter
    kernel = np.array([1, -3, 3, -1])
    filtered_x = np.zeros_like(im)
    filtered_y = np.zeros_like(im)

    for i in range(im.shape[0]):
        filtered_x[i] = signal.convolve(im[i], kernel, mode='same')
        
    for j in range(im.shape[1]):
        filtered_y[:, j] = signal.convolve(im[:, j], kernel, mode='same') 

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
        
    pattern_pooled = ((pattern_x + pattern_y).astype(float) / 2)
    n = 3
    resample_kernel = np.ones((n,n))/n**2
    resampled = signal.convolve2d(pattern_pooled, resample_kernel, mode='same') 
    return resampled

if __name__ == "__main__":
    num_images = 800
    width = 2000
    height = 1500
    resample_rate = 2
    test = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_0001.tif")
    test = test[::resample_rate, ::resample_rate]
    extracted = np.zeros((800, test.shape[0], test.shape[1]))

    for index in range(num_images):
        print("Extracting features from image {:04d}\r".format(index), end="")
        # Load image
        try:
            im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index+1))
        except IOError:
            im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.tif".format(index+1))

        im = im.sum(axis=2) / 3 / im.max()
        im = im[::resample_rate,::resample_rate]

        extracted[index] = spliceDetection(im)

    x = np.zeros([num_images, 750, 1000, 1])
    x[:,:,:,0] = extracted
    #normalize
    x -= x.min()
    x /= x.max()
    x = x.astype("float32")

    np.save("../dataset/extracted.npy", x)

