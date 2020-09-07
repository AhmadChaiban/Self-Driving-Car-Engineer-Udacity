from helpers import *
import cv2
from glob import glob as globlin
import matplotlib.pyplot as plt

if __name__ == '__main__':

    calibration_img_fnames = globlin('./camera_cal/*.*')
    print(calibration_img_fnames)
    imgpoints = []
    objpoints = []
    for path in calibration_img_fnames:
        img = cv2.imread(path)
        imgpoints, objpoints = calc_obj_img_points(img, objpoints, imgpoints)

    img = cv2.imread(calibration_img_fnames[6])
    undistorted_img, mtx, dist = camera_calibration(img, objpoints, imgpoints)

    plt.imshow(undistorted_img)
    plt.show()



