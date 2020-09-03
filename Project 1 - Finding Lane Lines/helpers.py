import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color, )

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # imshape = img.shape
    #
    # ymin_global = img.shape[0]
    # ymax_global = img.shape[0]
    #
    # all_left_grad = []
    # all_left_y = []
    # all_left_x = []
    #
    # all_right_grad = []
    # all_right_y = []
    # all_right_x = []
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for line in lines:
        for x1,y1,x2,y2 in line:
           # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif slope > 0:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    min_y = int(img.shape[0] * (3 / 5))
    max_y = int(img.shape[0])

    poly_left = np.poly1d(np.polyfit(
        left_y,
        left_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    cv2.line(img, (left_x_start, max_y), (left_x_end, min_y), color, thickness)

    poly_right = np.poly1d(np.polyfit(
        right_y,
        right_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    cv2.line(img, (right_x_start, max_y), (right_x_end, min_y), color, thickness)



#         gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
    #         ymin_global = min(min(y1, y2), ymin_global)
    #
    #         if (gradient > 0):
    #             all_left_grad += [gradient]
    #             all_left_y += [y1, y2]
    #             all_left_x += [x1, x2]
    #         else:
    #             all_right_grad += [gradient]
    #             all_right_y += [y1, y2]
    #             all_right_x += [x1, x2]
    #
    # left_mean_grad = np.mean(all_left_grad)
    # left_y_mean = np.mean(all_left_y)
    # left_x_mean = np.mean(all_left_x)
    # left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    #
    # right_mean_grad = np.mean(all_right_grad)
    # right_y_mean = np.mean(all_right_y)
    # right_x_mean = np.mean(all_right_x)
    # right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    #
    # if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
    #     upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
    #     lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
    #     upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
    #     lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)
    #
    #     cv2.line(img, (upper_left_x, ymin_global),
    #              (lower_left_x, ymax_global), color, thickness)
    #     cv2.line(img, (upper_right_x, ymin_global),
    #              (lower_right_x, ymax_global), color, thickness)






def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)