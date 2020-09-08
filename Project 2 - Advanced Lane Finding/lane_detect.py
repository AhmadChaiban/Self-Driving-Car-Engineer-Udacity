from helpers import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

def img_pipeline(img):
    undistorted_img = undistort_image(img)

    pers_transform, M, src = corners_unwarp(img)

    binary_img = get_s_channel(pers_transform)

    converted_img = apply_gradient(binary_img)

    filtered_img = cv2.GaussianBlur(converted_img, (3, 3), 0)

    hist = lane_histogram(filtered_img)

    leftx, lefty, rightx, righty, sliding_windows_img = apply_sliding_window(filtered_img)

    poly_fit_img = fit_polynomial(leftx, lefty, rightx, righty, sliding_windows_img)

    return poly_fit_img
    # plt.imshow(poly_fit_img)
    # plt.show()

    # plt.figure(figsize = (5, 5))
    #
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(undistorted_img)
    # plt.scatter(src[:, 0], src[:, 1], color = 'red')
    #
    # plt.subplot(212)
    # plt.imshow(pers_transform)
    # plt.show()
    #
    # plt.figure(figsize = (10,10))
    # plt.imshow(filtered_img, cmap="gray")
    # plt.show()

if __name__ == '__main__':

    img = cv2.imread('./test_images/test4.jpg')
    img_pipeline(img)

    white_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("videos/project_video.mp4")
    white_clip = clip1.fl_image(img_pipeline)
    white_clip.write_videofile(white_output, audio=False)



