import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class PolyFitter:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.ploty = None

    def lane_histogram(self, img):
        y = int(len(img)/2)
        x = int(len(img[0]))
        h = len(img)
        bottom_half = img[y:y+h, 0:x]

        histogram = sum(bottom_half)
        return histogram

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq  # Nyquist Frequency
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y


    def apply_sliding_window(self, filtered_img, nwindows=9, margin=100, minpix=50):
        out_img = np.dstack((filtered_img, filtered_img, filtered_img))
        histogram = np.sum(filtered_img[filtered_img.shape[0]//2:, :], axis=0)

        butter_low = self.butter_lowpass_filter(histogram, cutoff=3, fs=400, order=2)

        # plt.plot(histogram)
        # plt.plot(butter_low)
        # plt.show()

        midpoint = np.int(butter_low.shape[0]//2)
        leftx_base = np.argmax(butter_low[:midpoint])
        rightx_base = np.argmax(butter_low[midpoint:]) + midpoint

        window_height = np.int(filtered_img.shape[0]//nwindows)

        nonzero = filtered_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        count = 0
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

            # elif (len(good_left_inds) <= minpix) and (len(good_right_inds) <= minpix):
            #     count += 1
            # if count > 1:
            #     break

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
        left_fit = np.polyfit(lefty, leftx, deg=2)
        right_fit = np.polyfit(righty, rightx, deg=2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        ploty = np.linspace(0, sliding_windows_img.shape[0]-1, sliding_windows_img.shape[0])
        self.ploty = ploty

        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        # sliding_windows_img[lefty, leftx] = [255, 0, 0]
        # sliding_windows_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return sliding_windows_img, left_fitx, right_fitx, ploty

    def measure_curvature_pixels(self):

        y_eval = np.max(self.ploty)

        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])

        return left_curverad, right_curverad

    def search_around_poly(self, binary_warped, margin=100):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_constraint1 = self.left_fit[0]*nonzeroy**2 + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin
        left_constraint2 = self.left_fit[0]*nonzeroy**2 + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin
        right_constraint1 = self.right_fit[0]*nonzeroy**2 + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin
        right_constraint2 = self.right_fit[0]*nonzeroy**2 + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin

        left_lane_inds = ((nonzerox >= left_constraint1) & (nonzerox <= left_constraint2))
        right_lane_inds = ((nonzerox >= right_constraint1) & (nonzerox <= right_constraint2))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        sliding_windows_img, left_fitx, right_fitx, ploty = self.fit_polynomial(leftx, lefty, rightx, righty, binary_warped)

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return result, left_fitx, right_fitx, ploty