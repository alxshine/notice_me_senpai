from __future__ import division

import scipy.ndimage
#import imageio
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
    imageStore = np.zeros([imageNo-1,imSizeX,imSizeY])
    imageStore2 = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    imageStoreDenoise = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
    noisePattern = np.zeros([imageNo-1,np.int(imSizeX/2),np.int(imSizeY/2)])
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
    #K = (noisePattern[0] * imageStore2[0]) / (imageStore2[0] * imageStore2[0])
    #for i in range (1,len(imageStore2)):
    	#K *= (noisePattern[i] * imageStore2[i]) / (imageStore2[i] * imageStore2[i])
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

# re-define functions above for block image
# padding function for centering blocks per pixel coordinate
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

# try different blocksizes -> choice here 3x3
# block-wise approach:
# 1. define patched
# 2. for each patch correltate with phase corr for each camera in function
# 3. return result for every pixel around which the correlated block is centered (pixel center = pc)
#       and store in result array of picture size
# 4. compute offset estimate of result array and populate the prnu map

def block_fingerprint(block):
    wb = block[::2, ::2] - denoise(block)
    return wb

# corrletation function with K - camera noise, Wb - block\patch fingerprint, I - image
# R as the block's resulting 2D correlation to the camera's PRNU
# build PRNU map (via offset estimation decision) afterwards
def correlate(K, Wb, I, pc):
    # zero-padding for discrete fourier transform needed?
    # again: whole image for I and K or only blocks?
    #np.asmatrix(K)
    #np.asmatrix(Wb)
    #np.asmatrix(I)
    #frac1 = np.fft.fft2(Wb) * np.fft.fft2(np.conj(I*K))
    fWb = np.fft.fft2(Wb)
    fKI = np.fft.fft2(np.conj(I*K))
    frac1 = signal.convolve2d(fKI, fWb, mode='same')
    frac2 = np.absolute(frac1)
    R = np.fft.ifft2(frac1/frac2)
    offset_estimate = unravel_index(np.argmax(R),R.shape)
    print(R)
    print(len(R))
    print(offset_estimate)
    ibf, jbf = offset_estimate
    # build map -> return 1 only if PRNU corresponds to block centered around ib,jb - the center pixel coordinates
    #           -> return 0 if it does not pass the test
    if (np.array_equal([ibf,jbf],pc)):
    #if (np.array_equal([ibf,jbf],[1,1])):
        return 1
    else:
        return 0

# give all camera PRNUs (Ks) as argument
# test calculating noise per image patch vs noise of whole image and segmenting the whole image
#   noise into patches in terms of accruacy\time trade-off
def correlate_blocks(image,sizeX,sizeY,K):
    x = np.pad(image,1,pad_with,padder=0.0)
    #y = np.pad(K,1,pad_with,padder=0.0)
    result = np.zeros([sizeX,sizeY])
    # for PRNU map
    Mprnu = np.zeros([sizeX,sizeY])
    # for utilizing whole image noise instead of per block -> divide into patches after
    #imageResampled = np.repeat(image,2,axis=0)
    #imageResampled = np.repeat(imageResampled,2,axis=1)
    #imageNoise = block_fingerprint(imageResampled)
    for i in range(1,(sizeX+1)):
        for j in range(1,(sizeY+1)):
            pc = x[i,j]
            # blocks for image
            patch = np.zeros([3,3])
            patch[0,0] = x[i-1,j-1]
            patch[1,0] = x[i,j-1]
            patch[2,0] = x[i+1,j-1]
            patch[0,1] = x[i-1,j]
            patch[1,1] = x[i,j]
            patch[2,1] = x[i+1,j]
            patch[0,2] = x[i-1,j+1]
            patch[1,2] = x[i,j+1]
            patch[2,2] = x[i+1,j+1]
            #print(patch)
            # blocks for camera noise
            #Kn = np.zeros([3,3])
            #Kn[0,0] = y[i-1,j-1]
            #Kn[1,0] = y[i,j-1]
            #Kn[2,0] = y[i+1,j-1]
            #Kn[0,1] = y[i-1,j]
            #Kn[1,1] = y[i,j]
            #Kn[2,1] = y[i+1,j]
            #Kn[0,2] = y[i-1,j+1]
            #Kn[1,2] = y[i,j+1]
            #Kn[2,2] = y[i+1,j+1]
            # insert correlation function here
            patchResampled = np.repeat(patch,2,axis=0)
            patchResampled = np.repeat(patchResampled,2,axis=1)
            pc_result = block_fingerprint(patchResampled)
            #R = correlate(K,pc_result,image,pc)
            Mprnu[i-1][j-1] = correlate(K,pc_result,image,[i,j])
            print(i,j)
            print(Mprnu[i-1][j-1])
            result[i-1,j-1] = pc_result[1,1]
    #plt.figure(figsize=(18, 9))
    plt.figure(figsize=(24, 12))
    #plt.subplot(131)
    plt.subplot(141)
    plt.imshow(image)
    plt.title("original")
    #plt.subplot(132)
    plt.subplot(142)
    plt.imshow(x)
    plt.title("padded image")
    #plt.subplot(133)
    plt.subplot(143)
    plt.imshow(result)
    plt.title("resulting image")
    #plt.subplot(144)
    #plt.imshow(imageNoise)
    #plt.title("image noise")
    plt.subplot(144)
    plt.imshow(Mprnu)
    plt.title("PRNU map")
    plt.show()
    #print(image == result)
    #print(imageNoise == result)
    return result

# Approach:
# 1. obtain camera's PRNU (Wn) via Wn = In - D(In) with D(.) being a denoising function (discrete wavelet transformation)
#    for 5-10 flat (bright) pictures for each of the 4 cameras
# 2. segmentate images into overlapping blocks centered around pixel coordinates (ib, jb)
# 3. build map by comparing PRNU for each image block with camera noise fingerprint (Wn) via phase-correlation
#

# Load camera flat images
# selected images: to be selected
# 
#K1 = getCameraNoise(1,100,1500,2000)
K2 = getCameraNoise(2,100,1500,2000)
#K3 = getCameraNoise(3,100,1500,2000)
#K4 = getCameraNoise(4,100,1500,2000)

# block test
# result = correlate_blocks(K2,750,1000)
testpic = arr[::2, ::2]
result = correlate_blocks(testpic,750,1000,K2)

# for a quicker test
#testpic = arr[::10, ::10]
#KResampled = np.repeat(K2,2,axis=0)
#KResampled = np.repeat(K1,2,axis=0)
#KResampled = np.repeat(KResampled,2,axis=1)
#result = correlate_blocks(testpic,75,100,KResampled[::10,::10])

# tests with the test image

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

# split image into overlapping blocks

# patch approach
# still need to determine an appropiate patch size - chose 3x3 for smalloverlap in y direction
patches = image.extract_patches_2d(resampleSub, (3, 3))
#print(len(patches))
#print(patches)
#noise_patch = image.extract_patches_2d(K1, (3, 3))
noise_patch = image.extract_patches_2d(K2, (3, 3))

prnu_map = np.zeros_like(patches)
#print(patches[0][0+1][0+1])
#print(patches[1][0+1][0+1])
#print(patches[2][0+1][0+1])

prnu_arr1 = np.empty([len(patches)])

for i in range(0,len(patches)):
	if np.isclose(patches[i][0+1][0+1],noise_patch[i][0+1][0+1],atol=1):
		prnu_map[i][0+1][0+1] = 1
		prnu_arr1[i] = 1
	if patches[i][0+1][0+1] == noise_patch[i][0+1][0+1]:
		prnu_map[i][0+1][0+1] = 1
	else:
		prnu_map[i][0+1][0+1] = 0
		prnu_arr1[i] = 0

print(prnu_map)
prnu_arr = np.zeros([750,1000])

# block traversal:
# for each block/patch in image prnu and camera prnu check for corrletaion 
# if pristine -> block around pixel (block center) coordinates in block (ib,jb) fit with those of the camera
#               -> 1 in map at pixel's coordinates
# else fake -> 0 in map at pixel's coordinates


#for i in range(0,len(prnu_map)):
	#prnu_arr = np.append(prnu_arr, prnu_map[i][0+1][0+1])

print(len(prnu_arr1))


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