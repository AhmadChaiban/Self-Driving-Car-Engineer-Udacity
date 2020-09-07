from helpers import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

def img_pipeline(img):
    undistorted_img = undistort_image(img)

    pers_transform, M, src = corners_unwarp(img)



    hls_img = cv2.cvtColor(pers_transform, cv2.COLOR_RGB2HLS)

    S = hls_img[:, :, 2]

    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # plt.imshow(binary)
    # plt.show()
    #
    converted_img = apply_gradient(binary)
    #
    plt.figure(figsize = (5, 5))

    plt.figure(1)
    plt.subplot(211)
    plt.imshow(undistorted_img)
    plt.scatter(src[:, 0], src[:, 1], color = 'red')

    plt.subplot(212)
    plt.imshow(pers_transform)
    plt.show()

    plt.figure(figsize = (10,10))
    plt.imshow(converted_img)
    plt.show()

if __name__ == '__main__':

    img = cv2.imread('./test_images/straight_lines2.jpg')
    img_pipeline(img)



