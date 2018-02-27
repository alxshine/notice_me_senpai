import SUPPORT.prnu as prnu

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.misc
import os
import os.path

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
    correlation[np.isnan(correlation)] = correlation[0,0]
    correlation = np.abs(correlation)

    correlation = do_kernel_magic(correlation)
    correlation[np.isnan(correlation)] = correlation[0,0]

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
    #############################################
    #
    # Change source and target image folders here
    #
    #############################################
    source_dir = "demo_images"
    target_dir = "DEMO_RESULTS"

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for f in os.listdir(source_dir):
        print("generating estimate for {}".format(f))
        #get base name of f
        basename = os.path.splitext(f)[0]
        target_name = basename + '.bmp'

        result = find_spliced_areas(os.path.join(source_dir, f))
        scipy.misc.imsave(os.path.join(target_dir, target_name), result)
