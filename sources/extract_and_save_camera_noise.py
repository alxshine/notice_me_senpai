import prnu
import numpy as np

for index in range(4):
    print('Extracting noise pattern for camera {}'.format(index+1))
    cam_pattern = prnu.camera_noise(index+1, 100, 1500, 2000, 0, False)
    cam_pattern -= cam_pattern.min()
    cam_pattern /= cam_pattern.max()
    np.save('cam{}_pattern.npy'.format(index+1), cam_pattern)
