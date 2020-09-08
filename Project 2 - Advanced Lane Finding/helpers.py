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
    img_height, img_width = undist_img.shape[0], undist_img.shape[1]

    offset = 600
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

def get_s_channel(pers_tranformed_img):
    hls_img = cv2.cvtColor(pers_tranformed_img, cv2.COLOR_RGB2HLS)

    S = hls_img[:, :, 2]

    thresh = (120, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary

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

def lane_histogram(img):
    y = int(len(img)/2)
    x = int(len(img[0]))
    h = len(img)
    bottom_half = img[y:y+h, 0:x]

    histogram = sum(bottom_half)
    return histogram

def apply_sliding_window(filtered_img):
    out_img = np.dstack((filtered_img, filtered_img, filtered_img))
    histogram = np.sum(filtered_img[filtered_img.shape[0]//2:,:], axis=0)

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = np.int(filtered_img.shape[0]//nwindows)

    nonzero = filtered_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = filtered_img.shape[0] - (window+1)*window_height
        win_y_high = filtered_img.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((win_xleft_low <= nonzerox) & (nonzerox < win_xleft_high)
                          & (win_y_low <= nonzeroy) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((win_xright_low <= nonzerox) & (nonzerox < win_xright_high)
                           & (win_y_low <= nonzeroy) & (nonzeroy < win_y_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(leftx, lefty, rightx, righty, sliding_windows_img):
    left_fit = np.polyfit(lefty, leftx, deg = 2)
    right_fit = np.polyfit(righty, rightx, deg = 2)

    ploty = np.linspace(0, sliding_windows_img.shape[0]-1, sliding_windows_img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    sliding_windows_img[lefty, leftx] = [255, 0, 0]
    sliding_windows_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return sliding_windows_img

