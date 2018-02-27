import scipy
import prnu
import numpy as np
import matplotlib.pyplot as plt




all_corrs = np.zeros([10,1500,2000,1])
for image_index in range(1, 11):
    try:
        test_image = scipy.ndimage.imread('../dataset/dev-dataset-forged/dev_{:04d}.tif'.format(image_index))
    except IOError:
        test_image = scipy.ndimage.imread('../dataset/dev-dataset-forged/dev_{:04d}.jpg'.format(image_index))
    test_image = test_image.sum(axis=2)/3


    plt.figure()
    plt.imshow(prnu.get_best_image_correlation(test_image))
    plt.show()
    # all_corrs[image_index-1,:,:,0] = correlation

# np.save("../dataset/correlations.npy", all_corrs)[:,::2,::2,:]
