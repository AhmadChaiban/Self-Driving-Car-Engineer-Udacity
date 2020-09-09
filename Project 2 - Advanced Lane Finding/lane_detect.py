from helpers import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

def img_pipeline(img):
    undistorted_img = undistort_image(img)

    binary_img, L_channel = get_s_channel(undistorted_img)

    combined_binary = apply_gradient(undistorted_img, binary_img, L_channel)

    pers_transform, M, src = corners_unwarp(combined_binary)

    # filtered_img = cv2.GaussianBlur(pers_transform, (3, 3), 0)

    leftx, lefty, rightx, righty, sliding_windows_img = apply_sliding_window(pers_transform)

    poly_fit_img, left_fitx, right_fitx, ploty = fit_polynomial(leftx, lefty, rightx, righty, sliding_windows_img)

    result = project_to_video(pers_transform, undistorted_img, left_fitx, right_fitx, ploty, M, src)
    return result
    # return poly_fit_img


    plt.imshow(undistorted_img)
    plt.scatter(src[:, 0], src[:, 1], color = 'red')
    plt.show()

    plt.imshow(binary_img, cmap="gray")
    plt.show()

    plt.imshow(combined_binary, cmap="gray")
    plt.show()

    plt.imshow(pers_transform)
    plt.show()

    plt.imshow(poly_fit_img)
    plt.show()

    plt.imshow(result)
    plt.show()

if __name__ == '__main__':

    # img = cv2.imread('./test_images/test4.jpg')
    # img_pipeline(img)

    white_output = 'output_images/harder_challenge_video.mp4'
    clip1 = VideoFileClip("videos/harder_challenge_video.mp4")
    white_clip = clip1.fl_image(img_pipeline)
    white_clip.write_videofile(white_output, audio=False)

