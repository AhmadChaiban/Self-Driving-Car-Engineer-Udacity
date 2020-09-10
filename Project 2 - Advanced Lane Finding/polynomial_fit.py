import numpy as np
import cv2
import matplotlib.pyplot as plt

class PolyFitter:
    def __init__(self):
        pass

    def lane_histogram(self, img):
        y = int(len(img)/2)
        x = int(len(img[0]))
        h = len(img)
        bottom_half = img[y:y+h, 0:x]

        histogram = sum(bottom_half)
        return histogram

    def apply_sliding_window(self, filtered_img, nwindows=9, margin=100, minpix=50):
        out_img = np.dstack((filtered_img, filtered_img, filtered_img))
        histogram = np.sum(filtered_img[filtered_img.shape[0]//2:, :], axis=0)

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

    def fit_polynomial(self, leftx, lefty, rightx, righty, sliding_windows_img):
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

        return sliding_windows_img, left_fitx, right_fitx, ploty