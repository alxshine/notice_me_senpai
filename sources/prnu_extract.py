import scipy
import prnu
import numpy as np
import matplotlib.pyplot as plt




camera_patterns = [np.load('cam{}_pattern.npy'.format(i+1)) for i in range(4)]

all_corrs = np.zeros([10,1500,2000,1])
for image_index in range(1, 11):
    try:
        test_image = scipy.ndimage.imread('../dataset/dev-dataset-forged/dev_{:04d}.tif'.format(image_index))
    except IOError:
        test_image = scipy.ndimage.imread('../dataset/dev-dataset-forged/dev_{:04d}.jpg'.format(image_index))
    test_image = test_image.sum(axis=2)/3
    image_noise = test_image - prnu.denoise(test_image, 0)
    image_noise -= image_noise.min()
    image_noise /= image_noise.max()

    corrs = [np.corrcoef(camera_patterns[i].flatten(), image_noise.flatten())[0,1] for i in range(4)]
    cam_pattern = camera_patterns[np.argmax(corrs)]

    correlation = np.zeros_like(cam_pattern)
    im_height = cam_pattern.shape[0]
    im_width = cam_pattern.shape[1]
    patch_half_height = 1
    patch_half_width = 1
    search_window_half_size = 20

    print("Image {:03d}".format(image_index))
    for y in range(im_height):
        for x in range(im_width):
            noise_patch = prnu.get_patch_at(image_noise, y, x, patch_half_height, patch_half_width)
            cam_patch = prnu.get_patch_at(cam_pattern, y, x, patch_half_height, patch_half_width)
            # image_patch = prnu.get_patch_at(test_image, y, x, patch_half_height, patch_half_width)

            # IK = np.matmul(cam_patch,image_patch)
            # rs = np.conj(np.fft.fft(IK.flatten()))
            # ls = np.fft.fft(noise_patch.flatten())
            # numerator = np.multiply(rs,ls)
            # denominator = np.linalg.norm(numerator)
            # frac = np.divide(numerator,denominator)
            # R = np.fft.ifft(frac)

            # correlation[y,x] = np.argmax(R) == len(R)/2+1
            max_coeff = np.corrcoef(noise_patch.flatten(), cam_patch.flatten())[0,1]
            correlation[y,x] = max_coeff


    all_corrs[image_index-1,:,:,0] = correlation

np.save("../dataset/correlations.npy", all_corrs)[:,::2,::2,:]
