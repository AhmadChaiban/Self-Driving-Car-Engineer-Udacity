import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from glob import glob as globlin ## The 7bb globlin

nx = 9
ny = 6

def calc_obj_img_points(img, objpoints, imgpoints):
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return imgpoints, objpoints


def camera_calibration(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist


def undistort_image(img):
    dist_pickle = pickle.load( open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def corners_unwarp(undist_img):
    gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    print(undist_img.shape)
    img_height, img_width = undist_img.shape[0], undist_img.shape[1]

    offset = 500
    img_size = (gray.shape[1], gray.shape[0])

    src = np.float32([[0.2*img_width, img_height*0.90],
                    [0.45*img_width, 440],
                    [0.55*img_width, 440],
                    [0.8*img_width, img_height*0.90]])

    dst = np.float32([[0.2*img_width, img_height*0.90],
                      [0.45*img_width - offset, 440 - offset],
                      [0.55*img_width + offset, 440 - offset],
                      [0.8*img_width, img_height*0.90]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist_img, M, img_size)

    return warped, M, src

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_kernel)
    magnitude = np.sqrt(sobelx*sobelx + sobely*sobely)
    scaled_mag_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_mag_sobel)
    mag_binary[(scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_kernel)
    ## Remember the absolute limits the angle between +/- pi/2
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(grad_direction)
    dir_binary[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    return dir_binary

def apply_gradient(gray):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(25, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(25, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(25, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0, np.pi/2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

