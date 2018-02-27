import scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import pywt
from sklearn.feature_extraction import image
from numpy import unravel_index


# single level discrete wavelet transformation 2D for denoising
# cA = approximation, cH = horizontal detail, cV = vertical detail, cD = diagonal detail
# wiener filter of approximation to further refine denoising 
# (produce estimate of desired random process by LTI filtering of noise)
#
# returns denoised image
#
# TODO: try different de-noising options (coefficients, other wavelet like haar or db8, only wiener or wavelet filtering etc.)
#		also check when to apply second denoising filter - and if other filter than wiener might be better 
#		try multilevel decomposition instead
#
# different possibilities for handling coeffs denoted in mode:
# 	0	-	filter picture after inverse dwt 
#	1	-	filter each one seperately
# 	2	-	only filter with wavelets	
#	3	- 	only wiener (or gaussian or other salt and pepper) filter	
#	4	- 	comparison mode - show all different modes with noise extraction
#
def denoise(imarr,mode):
    coeffs = pywt.dwt2(imarr, 'db2', 'per')
    cA, (cH, cV, cD) = coeffs
    if (np.equal(mode,0) or np.equal(mode,4)):
    	dwi = pywt.idwt2(coeffs, 'db2', 'per')
    	wdi = scipy.signal.wiener(dwi)
    	d = wdi
    if (np.equal(mode,1) or np.equal(mode,4)):
    	wCA = scipy.signal.wiener(cA)
    	wCH = scipy.signal.wiener(cH)
    	wCV = scipy.signal.wiener(cV)
    	wCD = scipy.signal.wiener(cD)
    	wi = pywt.idwt2((wCA,(wCH,wCV,wCD)), 'db2', 'per')
    	d = wi
    if (np.equal(mode,2) or np.equal(mode,4)):
    	owi = pywt.idwt2(coeffs, 'db2', 'per')
    	d = owi
    if (np.equal(mode,3) or np.equal(mode,4)):
    	ospf = scipy.signal.wiener(imarr)
    	d = ospf
    if (np.equal(mode,4)):
    	nwdi = np.subtract(imarr,wdi)
    	nwi = np.subtract(imarr,wi)
    	nowi = np.subtract(imarr,owi)
    	nospf = np.subtract(imarr,ospf)
    	plt.figure(2,figsize=(18, 9))
    	plt.subplot(241)
    	plt.imshow(wdi)
    	plt.title("filter after dtw")
    	plt.subplot(245)
    	plt.imshow(nwdi)
    	plt.subplot(242)
    	plt.imshow(wi)
    	plt.title("filter each coeff")
    	plt.subplot(246)
    	plt.imshow(nwi)
    	plt.subplot(243)
    	plt.imshow(owi)
    	plt.title("only dwt denoisation")
    	plt.subplot(247)
    	plt.imshow(nowi)
    	plt.subplot(244)
    	plt.imshow(ospf)
    	plt.title("only salt&pepper (wiener) filter")
    	plt.subplot(248)
    	plt.imshow(nospf)
    	plt.show()
    return d

# retrieve noise of a given camera via a given number of images (with given resolution)
# and a specific noise mode
# showMode determines wether extracted camera prnu is shown
def camera_noise(camNo, noIm, imX, imY, mode, showMode):
	images = np.zeros([noIm,imX,imY])
	denoised = np.zeros([noIm,imX,imY])
	patterns = np.zeros([noIm,imX,imY])
	for i in range(1, noIm+1):
		im = scipy.ndimage.imread("../dataset/flat-camera-{:d}/flat_c{:d}_{:03d}.tif".format(camNo,camNo,i))
		im = np.array(im)
		im = im.sum(axis=2)/3/im.max()
		images[i-1] = im
		denoised[i-1] = denoise(im,mode)
		patterns[i-1] = im - denoised[i-1]
	print(len(images))
	K1 = (patterns[0] * images[0])
	K2 = (images[0] * images[0])
	for i in range (1,len(images)):
		K1 += (patterns[i]*images[i])
	for i in range (1,len(images)):
		K2 += (images[0] * images[0])
	K = K1/K2
	if(showMode):
		plt.figure(3)
		plt.imshow(K)
		plt.show()
	return K

# compair, plot and return noise patterns of each camera
# comairing plotting only works for 4 cameras!
def compair_camera_noise(noCam, noIm, imX, imY, mode):
	noises = np.zeros([noCam,imX,imY])
	for i in range(1, noCam+1):
		noises[i-1] = camera_noise(i, noIm, imX, imY, mode, 0)
	if(np.equal(noCam,4)):
		print(np.array_equal(noises[0],noises[1]))
		print(np.array_equal(noises[1],noises[2]))
		print(np.array_equal(noises[2],noises[3]))
		plt.figure(2,figsize=(18, 9))
		plt.subplot(221)
		plt.imshow(noises[0])
		plt.title("prnu camera 1")
		plt.subplot(222)
		plt.imshow(noises[1])
		plt.title("prnu camera 2")
		plt.subplot(223)
		plt.imshow(noises[2])
		plt.title("prnu camera 3")
		plt.subplot(224)
		plt.imshow(noises[3])
		plt.title("prnu camera 4")
		plt.show()
	return noises

def get_patch_at(im, y, x, patch_half_height, patch_half_width):
    """ We use y before x and height before width because that is the way they are saved in numpy.
    Also, height and width are given by their offset from the center pixel in the patch ([2,2] will result in a 5x5 patch"""

    im_height = im.shape[0]
    im_width = im.shape[1]
    patch_height = 2*patch_half_height+1
    patch_width = 2*patch_half_width+1

    offset_top = min(patch_half_height, y)
    offset_bottom = min(patch_half_height, im_height-y-1)
    offset_left = min(patch_half_width, x)
    offset_right = min(patch_half_width, im_width-x-1)
    # print("top: {}, bottom: {}, left: {}, right: {}".format(offset_top, offset_bottom, offset_left, offset_right))

    patch = np.zeros([patch_height, patch_width])
    patch[patch_half_height-offset_top:patch_half_height+offset_bottom+1, patch_half_width-offset_left:patch_half_width+offset_right+1] = im[y-offset_top:y+offset_bottom+1, x-offset_left:x+offset_right+1]
    return patch

camera_patterns = None
def get_best_image_correlation(im):
    global camera_patterns

    #initialize if necessary
    if type(camera_patterns) == type(None):
        camera_patterns = np.load('SUPPORT/cam_patterns.npy')

    #get the image noise
    image_noise = im - denoise(im, 0)
    image_noise -= image_noise.min()
    image_noise /= image_noise.max()

    #find the best prnu pattern
    corrs = [np.corrcoef(camera_patterns[i].flatten(), image_noise.flatten())[0,1] for i in range(camera_patterns.shape[0])]
    cam_pattern = camera_patterns[np.argmax(corrs)]

    
    #get the correlation between noise and the camera pattern
    correlation = np.zeros_like(cam_pattern)
    im_height = cam_pattern.shape[0]
    im_width = cam_pattern.shape[1]
    patch_half_height = 1
    patch_half_width = 1
    search_window_half_size = 20

    print("Generating correlation map")
    old_progress = -0
    for y in range(im_height):
        #print the progress
        progress = int(y*100/im_height)
        if progress > old_progress:
            print("Progress: {:02d}%".format(progress))
            old_progress = progress

        for x in range(im_width):
            noise_patch = get_patch_at(image_noise, y, x, patch_half_height, patch_half_width)
            cam_patch = get_patch_at(cam_pattern, y, x, patch_half_height, patch_half_width)
            max_coeff = np.corrcoef(noise_patch.flatten(), cam_patch.flatten())[0,1]
            correlation[y,x] = max_coeff

    print()

    return correlation
