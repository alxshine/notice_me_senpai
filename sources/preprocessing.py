import numpy as np

def fill(im):
    #find the median distance to the nearest neigbor
    print("Finding median distance to neighbor")
    distances = []
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if not im[y,x]:
                continue

            for y_other in range(im.shape[0]):
                for x_other in range(im.shape[1]):
                    if not im[y_other, x_other]:
                        continue

                    x_dist = x-x_other
                    y_dist = y-y_other
                    distances.append(np.sqrt(x_dist*x_dist + y_dist*y_dist))

    distance_threshold = np.median(distances)
    print("Done, threshold is {}".format(distance_threshold))
    ret = np.copy(im)

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if not im[y,x]:
                continue

            #in a circle around the current pixel with a radius of the threshold,
            #connect the pixel to all its "neighbors"
            for y_offset in range(-distance_threshold,distance_threshold):
                for x_offset in range(-distance_threshold,distance_threshold):
                    x_other = x + x_offset
                    y_other = y + y_offset

                    if x_other < 0 or x_other >= im.shape[1] or y_other < 0 or y_other >= im.shape[0]:
                        continue

                    if not im[y_other, x_other]:
                        continue
                
                    x_dist = x_other-x
                    y_dist = y_other-y

                    dist = int(np.sqrt(x_dist*x_dist + y_dist*y_dist))
                    x_step = x_dist/dist
                    y_step = y_dist/dist
                    for i in range(dist):
                        x_loc = int(np.clip(x + x_step * i, 0, im.shape[0]))
                        y_loc = int(np.clip(y + y_step * i, 0, im.shape[1]))
                        ret[y_loc, x_loc] = 1

print("Loading images...")
images = np.load("../dataset/extracted.npy")
print("Done")
filled = np.zeros_like(images)
for i in range(filled.shape[0]):
    print("Filling image {:04d}\r".format(i+1), end='')
    filled[i] = fill(images[i])

print("Done")

path = "../dataset/filled.npy"
print("Saving to {}".format(path))
np.save(path, filled)
print("Done")
