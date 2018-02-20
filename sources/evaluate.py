import sys
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

#load images and ground-truth maps
print("Loading training images...")
images = np.load("../dataset/extracted.npy")
print("Done, min: {}, max: {}, mean: {}".format(images.min(), images.max(), images.mean()))
print("Loading ground truth maps...")
maps = np.load("../dataset/maps.npy")
print("Done, min: {}, max: {}, mean: {}".format(maps.min(), maps.max(), maps.mean()))

print("Loading model...")
model = load_model('small_functional.h5')
print("Done")

prediction = model.predict(images[6:7])

# print("Evaluating...")
# metrics = model.evaluate(x=images, y=maps, batch_size=3)
# print("Done, metrics: {}".format(metrics))

try:
    index = int(sys.argv[1])
except IndexException:
    index = 7

plt.figure()
plt.subplot(131)
plt.imshow(images[index,:,:,0])
plt.title("image")

plt.subplot(132)
plt.imshow(maps[index,:,:,0])
plt.title("map")

plt.subplot(133)
plt.imshow(prediction[0,:,:,0])
plt.title("prediction")

plt.show()
