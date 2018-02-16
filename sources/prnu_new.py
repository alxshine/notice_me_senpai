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
    index = int(sys.argv[1])
except IndexError:
    index = 1

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

def getCameraNoise(camNo, imageNo, imSizeX, imSizeY):
    imageStore = np.zeros([imageNo-1,imSizeX,imSizeY])
    imageStore2 = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    imageStoreDenoise = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    noisePattern = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    for i in range (1, imageNo):
    	imageTemp = scipy.ndimage.imread("../dataset/flat-camera-{:d}/flat_c{:d}_{:03d}.tif".format(camNo,camNo,i))
    	imageTemp = np.array(imageTemp)
    	imageTemp = imageTemp.sum(axis=2) / 3 / imageTemp.max()
    	imageStore[i-1] = imageTemp
    	#resample before to conserve image resolution?
    	imageStore2[i-1] = imageTemp[::2, ::2]
    	imageStoreDenoise[i-1] = denoise(imageTemp)
    	noisePattern[i-1] = imageTemp[::2, ::2] - imageStoreDenoise[i-1]
    	#plt.figure(2,figsize=(18, 9))
    	#plt.subplot(231)
    	#plt.imshow(imageStore[i-1])
    	#plt.subplot(232)
    	#plt.imshow(imageStoreDenoise[i-1])
    	#plt.subplot(233)
    	#plt.imshow(noisePattern[i-1])
    	#plt.show()
    print(imageStore)
    K = (noisePattern[0] * imageStore2[0])
    K /= (imageStore2[0] * imageStore2[0])
    for i in range (1,len(imageStore2)):
        K += (noisePattern[i] * imageStore2[i])
    for i in range (1,len(imageStore2)):
        (imageStore2[i] * imageStore2[i])
    plt.figure(3)
    plt.imshow(K)
    plt.show()
    return K

denoised = denoise(arr,4)
#K1 = getCameraNoise(1,100,1500,2000)
#K2 = getCameraNoise(2,100,1500,2000)
#K3 = getCameraNoise(3,100,1500,2000)
#K4 = getCameraNoise(4,100,1500,2000)