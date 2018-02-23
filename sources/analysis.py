import numpy as np
import matplotlib.pyplot as plt

print("loading lighting features...")
features = np.load("../dataset/extracted.npy")
print("done")
print("loading prnu patterns...")
patterns = np.load("../dataset/patterns.npy")
print("done")
print("loading truth maps...")
maps = np.load("../dataset/maps.npy")
print("done")

for i in range(patterns.shape[0]):
    plt.figure()
    plt.subplot(131)
    plt.imshow(patterns[i,:,:,0], cmap='gray')
    plt.title("prnu pattern")

    plt.subplot(132)
    plt.imshow(maps[i,:,:,0], cmap='gray')
    plt.title("truth map")

    plt.subplot(133)
    plt.imshow(features[i,:,:,0], cmap='gray')
    plt.title("lighting features")

    plt.show()
