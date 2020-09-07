from helpers import *
import cv2
from glob import glob as globlin
import matplotlib.pyplot as plt
import pickle

def calibrate_camera():
    calibration_img_fnames = globlin('./camera_cal/*.*')
    imgpoints = []
    objpoints = []
    for path in calibration_img_fnames:
        img = cv2.imread(path)
        imgpoints, objpoints = calc_obj_img_points(img, objpoints, imgpoints)

    img = cv2.imread(calibration_img_fnames[6])
    undistorted_img, mtx, dist = camera_calibration(img, objpoints, imgpoints)

    pickle_out = open("wide_dist_pickle.p", "wb")
    pickle.dump({'mtx': mtx, 'dist': dist}, pickle_out)
    pickle_out.close()

    return undistorted_img

if __name__ == '__main__':
    undistorted_img = calibrate_camera()

    plt.imshow(undistorted_img)
    plt.show()