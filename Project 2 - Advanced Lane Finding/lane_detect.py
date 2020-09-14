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

    ## Sobel and HSV
    undistorted_img = camera.undistort_image(img)

    binary_img, L_channel, S_channel = gradient_applier.get_s_L_channel(undistorted_img)

    combined_binary_L = gradient_applier.apply_gradient(undistorted_img, binary_img, L_channel, camera)

    combined_binary_S = gradient_applier.apply_gradient(undistorted_img, binary_img, S_channel, camera)

    combined_binary = cv2.bitwise_or(combined_binary_S, combined_binary_L)

    pers_transform, M, src = camera.corners_unwarp(combined_binary)

    stacked_pers_transform = np.dstack((pers_transform, pers_transform, pers_transform))

    color_masked_image_warped = gradient_applier.combine_color_mask(undistorted_img, camera)

    final_transform = cv2.bitwise_or(stacked_pers_transform, color_masked_image_warped)[:, :, 0]

    try:
        sliding_windows_img, left_fitx, right_fitx, ploty = poly_fitter.search_around_poly(final_transform)
    except:
        leftx, lefty, rightx, righty, sliding_windows_img = poly_fitter.apply_sliding_window(final_transform)
        poly_fit_img,  left_fitx, right_fitx, ploty = poly_fitter.fit_polynomial(leftx, lefty, rightx, righty, sliding_windows_img)

    result = project_to_video(pers_transform, undistorted_img, left_fitx, right_fitx, ploty, M, src)

    left_curverad, right_curverad = poly_fitter.measure_curvature_pixels()

    cv2.putText(result, str(round(left_curverad, 0)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result, str(round(right_curverad, 0)), (1100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

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

    plt.imshow(pers_transform, cmap="gray")
    plt.show()

    plt.imshow(final_transform)
    plt.title('final_transform')
    plt.show()

    plt.imshow(poly_fit_img)
    plt.show()

    plt.imshow(result)
    plt.show()

    exit()
    # return result

if __name__ == '__main__':

    # img = cv2.imread('./test_images/test4.jpg')
    # img_pipeline(img)

    white_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("videos/project_video.mp4")#.subclip(18, 24)
    white_clip = clip1.fl_image(img_pipeline)
    clip = white_clip.fl_image(pipeline_yolo)
    clip.write_videofile(white_output, audio=False)

    white_output = 'output_images/challenge_video.mp4'
    clip1 = VideoFileClip("videos/challenge_video.mp4")
    white_clip = clip1.fl_image(img_pipeline)
    clip = white_clip.fl_image(pipeline_yolo)
    clip.write_videofile(white_output, audio=False)

    white_output = 'output_images/harder_challenge_video.mp4'
    clip1 = VideoFileClip("videos/harder_challenge_video.mp4")#.subclip(7, 8)
    white_clip = clip1.fl_image(img_pipeline)
    clip = white_clip.fl_image(pipeline_yolo)
    clip.write_videofile(white_output, audio=False)

