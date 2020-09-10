from helpers import *
import matplotlib.pyplot as plt
from calibrate_camera import CameraCalibrator
from gradients import GradientApplier
from polynomial_fit import PolyFitter
from VehicleClassifier.yolo_pipeline import *
# from VehicleClassifier.visualizations import *
import cv2
import numpy as np

camera = CameraCalibrator(9, 6)
gradient_applier = GradientApplier()
poly_fitter = PolyFitter()

def pipeline_yolo(img):

    output = vehicle_detection_yolo(img, img)

    return output

def img_pipeline(img):
    undistorted_img = camera.undistort_image(img)

    binary_img, L_channel = gradient_applier.get_s_channel(undistorted_img)

    combined_binary = gradient_applier.apply_gradient(undistorted_img, binary_img, L_channel)

    pers_transform, M, src = camera.corners_unwarp(combined_binary)

    # filtered_img = cv2.GaussianBlur(pers_transform, (3, 3), 0)

    try:
        sliding_windows_img, left_fitx, right_fitx, ploty = poly_fitter.search_around_poly(pers_transform)
    except:
        leftx, lefty, rightx, righty, sliding_windows_img = poly_fitter.apply_sliding_window(pers_transform)
        poly_fit_img,  left_fitx, right_fitx, ploty = poly_fitter.fit_polynomial(leftx, lefty, rightx, righty, sliding_windows_img)

    result = project_to_video(pers_transform, undistorted_img, left_fitx, right_fitx, ploty, M, src)

    return result
    # return sliding_windows_img
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

    # return result

if __name__ == '__main__':

    # img = cv2.imread('./test_images/test4.jpg')
    # img_pipeline(img)

    white_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("videos/project_video.mp4")#.subclip(18, 24)
    white_clip = clip1.fl_image(img_pipeline)
    clip = white_clip.fl_image(pipeline_yolo)
    clip.write_videofile(white_output, audio=False)

    # white_output = 'output_images/challenge_video.mp4'
    # clip1 = VideoFileClip("videos/challenge_video.mp4")
    # white_clip = clip1.fl_image(img_pipeline)
    # clip = white_clip.fl_image(pipeline_yolo)
    # clip.write_videofile(white_output, audio=False)
    #
    # white_output = 'output_images/harder_challenge_video.mp4'
    # clip1 = VideoFileClip("videos/harder_challenge_video.mp4")
    # white_clip = clip1.fl_image(img_pipeline)
    # clip = white_clip.fl_image(pipeline_yolo)
    # clip.write_videofile(white_output, audio=False)
    #
