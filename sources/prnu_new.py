from __future__ import division

import scipy.ndimage
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import pywt
from sklearn.feature_extraction import image
from numpy import unravel_index

try:
    mode = int(sys.argv[1])
except IndexError:
    mode = 4

try:
    index = int(sys.argv[2])
except IndexError:
    index = 5


# Load test image
try:
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index))
except IOError:
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.tif".format(index))
truth_map = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index))

# convert to grayscale
im = im.astype('float32')
arr = np.array(im)
arr = arr.sum(axis=2) / 3 / arr.max()

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
# different possibilities for handeling coeffs denoted in mode:
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

# padding function to aid computing prnu noise centered around each pixel
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

# returns padded and patched version of image 
# pX and pY should be symmetrical and odd to be suitable for centering
def patch_image(im, pX, pY):
	padded = np.pad(im,1,pad_with,padder=0.0)
	patches = image.extract_patches_2d(padded,(pX,pY))
	print(len(patches))
	# TODO: image size as argument and patches center variable!
	center_pixels = np.zeros([1500*2000])
	for i in range(0,len(patches)):
		center_pixels[i] = patches[i][1][1]
	center_pixels = center_pixels.reshape(1500,2000)
	return patches, center_pixels


denoised = denoise(arr,mode)
noise_image = np.subtract(arr,denoised)
#K1 = camera_noise(1,5,1500,2000,0,1)
#noises = compair_camera_noise(4,5,1500,2000,0)
#print(np.array_equal(noises[0],K1))
patched_image, center_pixels = patch_image(denoised,3,3)
#print(np.array_equal(denoised,center_pixels))

