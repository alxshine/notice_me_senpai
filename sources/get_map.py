import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import prnu

def filter_edge_artifacts(im, top, bottom):
    edge_depths = np.zeros([bottom-top])
    for row in range(top, bottom):
        column = 0
        while column < im.shape[1] and im[row, column] == 1:
            column += 1
        edge_depths[row-top] = column

    edge_threshold = int(np.median(edge_depths))
    im[:edge_threshold,:] = 0
    im[:,:edge_threshold] = 0
    im[-edge_threshold:,:] = 0
    im[:,-edge_threshold:] = 0

def do_kernel_magic(im):
    print("Doing kernel magic")

    n = 3
    mean_kernel_small = np.ones([n,n])/n**2
    n = 5
    mean_kernel_large = np.ones([n,n])/n**2
    gaussian_kernel = np.array(
            [[0.003765, 0.015019, 0.025792, 0.015019, 0.003765],
             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
             [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
             [0.003765, 0.015019, 0.025792, 0.015019, 0.003765]])


    im = scipy.signal.convolve2d(im, gaussian_kernel, mode='same')
    for i in range(100):
        im = scipy.signal.convolve2d(im, mean_kernel_small, mode='same')
    for i in range(100):
        im = scipy.signal.convolve2d(im, mean_kernel_large, mode='same')

    return im

def find_spliced_areas(imagepath):
    #load, convert to grayscale and normalize
    im = scipy.ndimage.imread(imagepath)

    if len(im.shape) == 3:
        im = im.sum(axis=2)/im.shape[2]

    im -= im.min()
    im /= im.max()

    correlation = prnu.get_best_image_correlation(im)
    correlation[np.isnan(im)] = correlation[0,0]
    correlation = np.abs(correlation)

    correlation = do_kernel_magic(correlation)

    #thresholding
    print("Doing thresholding")
    threshold = correlation.mean()*0.97
    result = np.zeros_like(correlation)
    result[correlation<threshold] = 1

    #fix the edge artifacts
    top = 400
    bottom = 1400
    filter_edge_artifacts(result, top, bottom)

    return result

if __name__ == '__main__':
    #this is just for testing
    #any actual evaluation should be done using the find_spliced_areas function
    result = find_spliced_areas("../dataset/dev-dataset-forged/dev_0001.tif")

    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title("result")
    plt.show()

    # plt.show()
