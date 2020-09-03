#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from helpers import *

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

os.listdir("test_images/")

def process_image(image):
    # returns a color image (3 channel) for processing the videos below

    img_height, img_width = image.shape[0], image.shape[1]

    vertices = np.array([[(0.13*img_width, img_height),
                          (0.5*img_width, (305/img_height)*img_height),
                          (img_width - 0.0521*img_width, img_height)]], dtype=np.int32)

    # vertices = np.array([[(0, img_height),
    #                       (img_width / 2, img_height / 2),
    #                       (img_width, img_height)]], dtype=np.int32)

    region_selected_img = region_of_interest(image, vertices=vertices)

    gray_image = grayscale(region_selected_img)
    gblur_img = gaussian_blur(gray_image, kernel_size=1)

    edge_det_img = canny(gblur_img, low_threshold=100, high_threshold=200)

    region_selected_img = region_of_interest(edge_det_img, vertices=vertices)

    try:
        hough_image = hough_lines(region_selected_img,
                                  rho=1,
                                  theta=1*np.pi/180,
                                  threshold=15,
                                  min_line_len=40,
                                  max_line_gap=25)
    except:
        plt.imshow(region_selected_img)
        plt.show()
        exit()

    result = weighted_img(hough_image, image)

    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

challenge_output = 'test_videos_output/Toronto_youtube.mp4'
clip3 = VideoFileClip('test_videos/Toronto_youtube.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)