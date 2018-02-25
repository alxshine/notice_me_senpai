import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt

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
    im = scipy.signal.convolve2d(im, gaussian_kernel, mode='same')
    for i in range(100):
        im = scipy.signal.convolve2d(im, mean_kernel_small, mode='same')
    for i in range(100):
        im = scipy.signal.convolve2d(im, mean_kernel_large, mode='same')

    return im


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

print("loading features")
features = np.load("../dataset/correlations.npy")
print("done")
print("loading maps")
maps = np.load("../dataset/maps.npy")
print("done")

accuracies = 0
baselines = 0

indices = range(10)
for index in indices:
    print("image {}".format(index+1))

    im = features[index]
    im[np.isnan(im)] = im[0,0]
    im = np.abs(im)
    im -= im.min()
    im /= im.max()
    m = maps[index,:,:,0]
    m -= m.min()
    m /= m.max()

    im = do_kernel_magic(im)

    #thresholding
    threshold = im.mean()*0.97
    result = np.zeros_like(im)
    result[im<threshold] = 1

    #fix the edge artifacts
    top = 400
    bottom = 1400
    filter_edge_artifacts(result, top, bottom)

    #calculate accuracy
    incorrect_rate = np.sum(np.abs(result-m))/np.prod(result.shape)
    accuracy = (1-incorrect_rate)*100
    accuracies += accuracy

    baseline = (1-np.sum(m)/np.prod(result.shape))*100
    baselines += baseline
    print("accuracy: {:4.2f}%".format(accuracy))
    print("baseline: {:4.2f}%".format(baseline))


    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(result, cmap='gray')
    # plt.title("result")

    # plt.subplot(122)
    # plt.imshow(m)
    # plt.title("map")

    # plt.show()

print("Average accuracy: {}".format(accuracies/len(indices)))
print("Average baseline: {}".format(baselines/len(indices)))
