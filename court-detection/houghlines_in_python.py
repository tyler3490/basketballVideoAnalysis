import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from houghlines_in_python_helpers import draw_lines, weighted_img
 
image = mpimg.imread("/Users/tyler/Documents/GitHub/basketballVideoAnalysis/court-detection/input/tennis.jpg")
# image1copy = np.uint8(image)
# gray_image = cv2.cvtColor(image1copy, cv2.COLOR_RGB2GRAY)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
# blurred_imageCopy = np.uint8(blurred_image)
# edges_image = cv2.Canny(blurred_imageCopy, 50, 120)
edges_image = cv2.Canny(blurred_image, 50, 120)

rho_resolution = 1
theta_resolution = np.pi/180
threshold = 155
 
hough_lines = cv2.HoughLines(edges_image, rho_resolution , theta_resolution , threshold)
 
hough_lines_image = np.zeros_like(image)
draw_lines(hough_lines_image, hough_lines)
original_image_with_hough_lines = weighted_img(hough_lines_image,image)
 
plt.figure(figsize = (30,20))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(edges_image, cmap='gray')
plt.subplot(133)
plt.imshow(original_image_with_hough_lines, cmap='gray') 
plt.show()