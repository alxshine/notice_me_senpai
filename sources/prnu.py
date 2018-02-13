from __future__ import division

import scipy.ndimage
#import imageio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import pywt
from sklearn.feature_extraction import image

try:
    index = int(sys.argv[1])
except IndexError:
    index = 1

# Load test image
try:
    #im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index), mode='L')
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index))
    #im = scipy.imageio.imread("../dataset/dev-dataset-forged/dev_{:04d}.jpg".format(index), mode='L')
except IOError:
    im = scipy.ndimage.imread("../dataset/dev-dataset-forged/dev_{:04d}.tif".format(index))
truth_map = scipy.ndimage.imread("../dataset/dev-dataset-maps/dev_{:04d}.bmp".format(index))

# convert to grayscale
arr = np.array(im)
arr = arr.sum(axis=2) / 3 / arr.max()

# single level discrete wavelet transformation 2D for denoising
# cA = approximation, cH = horizontal detail, cV = vertical detail, cD = diagonal detail
# wiener filter of approximation to further refine denoising 
# (produce estimate of desired random process by LTI filtering of noise)
def denoise(imarr):
    coeffs = pywt.dwt2(imarr, 'db2', 'per')
    cA, (cH, cV, cD) = coeffs
    wiener = scipy.signal.wiener(cA)
    return wiener

def getCameraNoise(camNo, imageNo, imSizeX, imSizeY):
    imageStore = np.empty([imageNo-1,imSizeX,imSizeY])
    imageStore2 = np.empty([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    imageStoreDenoise = np.empty([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    noisePattern = np.empty([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    for i in range (1, imageNo):
    	imageTemp = scipy.ndimage.imread("../dataset/flat-camera-{:d}/flat_c{:d}_{:03d}.tif".format(camNo,camNo,i))
    	imageTemp = np.array(imageTemp)
    	imageTemp = imageTemp.sum(axis=2) / 3 / arr.max()
    	imageStore[i-1] = imageTemp
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
    K = (noisePattern[0] * imageStore2[0]) / (imageStore2[0] * imageStore2[0])
    for i in range (1,len(imageStore2)):
    	K *= (noisePattern[i] * imageStore2[i]) / (imageStore2[i] * imageStore2[i])
    plt.figure(3)
    plt.imshow(K)
    plt.show()
    return K

# re-define functions above for block image

# Approach:
# 1. obtain camera's PRNU (Wn) via Wn = In - D(In) with D(.) being a denoising function (discrete wavelet transformation)
#    for 5-10 flat (bright) pictures for each of the 4 cameras
# 2. segmentate images into overlapping blocks centered around pixel coordinates (ib, jb)
# 3. build map by comparing PRNU for each image block with camera noise fingerprint (Wn) via phase-correlation
#

# Load camera flat images
# selected images: to be selected
# 
#K1 = getCameraNoise(1,80,1500,2000)
K2 = getCameraNoise(2,80,1500,2000)
# very skimpy test
#print(K1 == K2)

#(cA, cD) = pywt.dwt(arr, 'db2', 'per')
#wiener = scipy.signal.wiener(cA)
#iDWT = pywt.idwt(cA, cD, 'db2', 'per')

# single level discrete wavelet transformation 2D for denoising
# cA = approximation, cH = horizontal detail, cV = vertical detail, cD = diagonal detail
coeffs = pywt.dwt2(arr, 'db2', 'per')
cA, (cH, cV, cD) = coeffs

# wiener filter of approximation to further refine denoising 
# (produce estimate of desired random process by LTI filtering of noise)
wiener = scipy.signal.wiener(cA)
resample = arr[::2, ::2]
resampleSub = resample - wiener

# still need to determine an apropiate patch size - chose 3x3 for smalloverlap in y direction
patches = image.extract_patches_2d(resampleSub, (3, 3))
print(len(patches))
print(patches)
#noise_patch = image.extract_patches_2d(K1, (3, 3))
noise_patch = image.extract_patches_2d(K2, (3, 3))

prnu_map = np.zeros_like(patches)
print(patches[0][0+1][0+1])
print(patches[1][0+1][0+1])
print(patches[2][0+1][0+1])

prnu_arr = np.empty([len(patches)])

for i in range(0,len(patches)):
	if np.isclose(patches[i][0+1][0+1],noise_patch[i][0+1][0+1],atol=0.2):
		prnu_map[i][0+1][0+1] = 1
		prnu_arr[i] = 1
	#if patches[i][0+1][0+1] == noise_patch[i][0+1][0+1]:
		#prnu_map[i][0+1][0+1] = 1
	else:
		prnu_map[i][0+1][0+1] = 0
		prnu_arr[i] = 0

print(prnu_map)

#for i in range(0,len(prnu_map)):
	#prnu_arr = np.append(prnu_arr, prnu_map[i][0+1][0+1])

print(prnu_arr)
#prnu_arr_map = prnu_arr.reshape(750,1000)

#C1 = scipy.signal.correlate2d(K1, resampleSub)
#plt.figure(3)
#plt.imshow(prnu_arr_map)
#plt.show()
#C2 = scipy.signal.correlate2d(K2, resampleSub)
#plt.figure(3)
#plt.imshow(C2)
#plt.show()

plt.figure(figsize=(18, 9))
plt.subplot(131)
plt.imshow(arr)
plt.title("original")
 
plt.subplot(132)
plt.imshow(resampleSub)
plt.title("normal image - denoised image")

plt.subplot(133)
plt.imshow(cA)
plt.title("approximation")

plt.show()