from helpers import *
import cv2
from glob import glob as globlin
import matplotlib.pyplot as plt
import pickle
import numpy as np

class CameraCalibrator:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def calc_obj_img_points(self, img, objpoints, imgpoints):
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
        return imgpoints, objpoints


    def camera_calibration(self, img, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist, mtx, dist


    def undistort_image(self, img):
        dist_pickle = pickle.load( open("wide_dist_pickle.p", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def corners_unwarp(self, undist_img, offset = 400):
        # gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = undist_img.shape[0], undist_img.shape[1]

        img_size = (undist_img.shape[1], undist_img.shape[0])

        # src = np.float32([[0.15*img_width, img_height*0.95],
        #                   [0.45*img_width, 455],
        #                   [0.55*img_width, 455],
        #                   [0.85*img_width, img_height*0.95]])
        #
        # dst = np.float32([[0.15*img_width, img_height*0.95],
        #                   [0.45*img_width - offset, 455 - offset],
        #                   [0.55*img_width + offset, 455 - offset],
        #                   [0.85*img_width, img_height*0.95]])

        src = np.float32([[0, img_height],
                          [0.4*img_width, 455],
                          [0.6*img_width, 455],
                          [img_width, img_height]])

        dst = np.float32([[0, img_height],
                          [0, 0],
                          [img_width, 0],
                          [img_width, img_height]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist_img, M, img_size)

        return warped, M, src

    def calibrate_camera(self):
        calibration_img_fnames = globlin('./camera_cal/*.*')
        imgpoints = []
        objpoints = []
        for path in calibration_img_fnames:
            img = cv2.imread(path)
            imgpoints, objpoints = self.calc_obj_img_points(img, objpoints, imgpoints)

        img = cv2.imread(calibration_img_fnames[6])
        undistorted_img, mtx, dist = self.camera_calibration(img, objpoints, imgpoints)

        pickle_out = open("wide_dist_pickle.p", "wb")
        pickle.dump({'mtx': mtx, 'dist': dist}, pickle_out)
        pickle_out.close()

        return undistorted_img

if __name__ == '__main__':
    undistorted_img = CameraCalibrator(9, 6).calibrate_camera()

    plt.imshow(undistorted_img)
    plt.show()