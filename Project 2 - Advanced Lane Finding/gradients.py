import cv2
import numpy as np

class GradientApplier:

    def __init__(self):
        pass

    def get_s_channel(self, pers_tranformed_img, thresh=(170, 255)):
        hls_img = cv2.cvtColor(pers_tranformed_img, cv2.COLOR_RGB2HLS)

        S = hls_img[:, :, 2]

        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1

        return binary, hls_img[:, :, 1]

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Apply threshold
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return grad_binary

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_kernel)
        magnitude = np.sqrt(sobelx*sobelx + sobely*sobely)
        scaled_mag_sobel = np.uint8(255*magnitude/np.max(magnitude))
        # Apply threshold
        mag_binary = np.zeros_like(scaled_mag_sobel)
        mag_binary[(scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1])] = 1
        return mag_binary

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_kernel)
        ## Remember the absolute limits the angle between +/- pi/2
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)
        # Apply threshold
        dir_binary = np.zeros_like(grad_direction)
        dir_binary[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
        return dir_binary

    def apply_gradient(self, undistorted_img, binary_S_img, L_channel):
        # Choose a Sobel kernel size
        ksize = 11 # Choose a larger odd number to smooth gradient measurements
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(L_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        # grady = self.abs_sobel_thresh(L_channel, orient='y', sobel_kernel=ksize, thresh=(90, 250))
        # mag_binary = self.mag_thresh(L_channel, sobel_kernel=ksize, mag_thresh=(90, 255))
        # dir_binary = self.dir_threshold(L_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

        # combined = np.zeros_like(dir_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        combined_binary = np.zeros_like(gradx)
        combined_binary[(gradx == 1) | (binary_S_img == 1)] = 1

        return combined_binary