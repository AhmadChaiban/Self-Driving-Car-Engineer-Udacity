import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from glob import glob as globlin ## The 7bb globlin

def project_to_video(pers_transform, undistorted_img, left_fitx, right_fitx, ploty, M, src):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(pers_transform).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (undistorted_img.shape[1], undistorted_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    return result

def save_img_pipelines(img,
                       undistorted_img,
                       L_channel,
                       S_channel,
                       combined_binary_S,
                       combined_binary_L,
                       combined_binary,
                       pers_transform,
                       color_masked_image_warped,
                       final_transform,
                       sliding_windows_img,
                       result):

    cv2.imwrite('./pipeline_imgs/1_initial_img.jpg',               img)
    cv2.imwrite('./pipeline_imgs/2_undistorted_img.jpg',           undistorted_img)
    cv2.imwrite('./pipeline_imgs/3_L_channel.jpg',                 L_channel)
    cv2.imwrite('./pipeline_imgs/4_S_channel.jpg',                 S_channel)
    cv2.imwrite('./pipeline_imgs/5_combined_binary_S.jpg',         combined_binary_S * 255)
    cv2.imwrite('./pipeline_imgs/6_combined_binary_L.jpg',         combined_binary_L * 255)
    cv2.imwrite('./pipeline_imgs/7_combined_binary.jpg',           combined_binary * 255)
    cv2.imwrite('./pipeline_imgs/8_pers_transform.jpg',            pers_transform * 255)
    cv2.imwrite('./pipeline_imgs/9_color_masked_image_warped.jpg', color_masked_image_warped)
    cv2.imwrite('./pipeline_imgs/10_final_transform.jpg',           final_transform)
    cv2.imwrite('./pipeline_imgs/11_sliding_windows_img.jpg',       sliding_windows_img)
    cv2.imwrite('./pipeline_imgs/12_result.jpg',                    result)

