#
# Author Stephan Janssen
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
# image = '/Users/tyler/Documents/GitHub/basketballVideoAnalysis/swingvisioncap.jpg'
# image = '/Users/tyler/Documents/GitHub/basketballVideoAnalysis/courtScreenCap.jpg'
image1 = '/Users/tyler/Documents/GitHub/basketballVideoAnalysis/courtLevelCap.jpg'
image2 = '/Users/tyler/Documents/GitHub/basketballVideoAnalysis/courtLevelCap.jpg'
# Read source image.
# img_src = cv2.imread('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/courtScreenCap.jpg')


img1 = cv2.imread(image1)
img2 = cv2.imread(image2)
print(img1.shape)
print(img2.shape)

img1=img1[396:, :]
img2=img2[396:, :]
img2 = cv2.rotate(img2, cv2.ROTATE_180)
cv2.imshow("cut in half1?", img1)
cv2.imwrite('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/cutinhalf1.jpg', img1)
cv2.imshow("cut in half2?", img2)
cv2.imwrite('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/cutinhalf2.jpg', img2)
# stich them together....
# img_src = cv2.imread(image)

# Four corners of the 3D court + mid-court circle point in source image
# Start top-left corner and go anti-clock wise + mid-court circle point
# courtScreenCap
pts_src = np.array([
    [180, 1100],       # left bottom - bottom corner
    # [400, 308],     # middle bottom corner
    [1915, 1083],     # right bottom - bottom corner
    # [798, 220],     # right bottom - top corner
    [1607, 276],     # top right rorner
    [486, 286],     # top left corner
    # [3, 201]        # left bottom - top corner
    ])   

# courtlevelcap
pts_srcimg1 = np.array([
    [879,9],     # top left corner
    [11,782],    # bottom left corner
    [2660,759],     # bottom right corner
    [1849,1],     # top right corner
    ])  
pts_srcimg2 = np.array([
    [210,92],     # bottom right corner
    [978,849],     # top right corner
    [1994,842],     # top left corner
    [2864,67],    # bottom left corner
    ])    

# swingvisioncap
# pts_src = np.array([
#     [101, 880],       # left bottom - bottom corner
#     # [400, 308],     # middle bottom corner
#     [2449, 864],     # right bottom - bottom corner
#     # [798, 220],     # right bottom - top corner
#     [1474, 311],     # top right rorner
#     [906, 324],     # top left corner
#     # [3, 201]        # left bottom - top corner
#     ])   

# pts_src = np.array([
#     [1, 258],       # left bottom - bottom corner
#     [400, 308],     # middle bottom corner
#     [798, 280],     # right bottom - bottom corner
#     [798, 220],     # right bottom - top corner
#     [612, 176],     # top right rorner
#     [186, 168],     # top left corner
#     [3, 201]        # left bottom - top corner
#     ])   

# cv2.fillPoly(img_src, [pts_src], 255)
# cv2.polylines(img_src, [pts_src], isClosed=True, color=[255,0,0], thickness=2)

# plt.imshow(img_src)
# plt.title('Original')
# plt.show()

# Read destination image.
img_dst = cv2.imread('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/court-detection/court_configurations/court_reference.png')

# Four corners of the court + mid-court circle point in destination image 
# Start top-left corner and go anti-clock wise + mid-court circle point
pts_dst = np.array([
    [289, 2937],       # left bottom - bottom corner
    # [317, 351],      # middle bottom corner
    [1380, 2937],     # right bottom - bottom corner
    # [629, 293],     # right bottom - top corner
    [1380, 562],     # top right rorner
    [284, 559],     # top left corner
    # [2, 299]        # left bottom - top corner
    ])   

# court with mid line
pts_dst_wMid = np.array([
    [421, 559],     # top left corner
    [421, 1748],     # mid left
    [421, 2937],    # bottom left corner
    [1244, 2937],     # bottom right corner
    [1244, 1748],     # mid right 
    [1244, 559],     # top right corner
    ])   
# top court with mid line
pts_dst_top_wMid = np.array([
    [421, 559],     # top left corner
    [421, 1748],     # mid left
    [1244, 1748],     # mid right 
    [1244, 559],     # top right corner
    ])   
# bottom court with mid line
pts_dst_bottom_wMid = np.array([
    [421, 1748],     # mid left
    [421, 2937],    # bottom left corner
    [1244, 2937],     # bottom right corner
    [1244, 1748],     # mid right 
    ])   

# og
# pts_dst = np.array([
#     [43, 355],       # left bottom - bottom corner
#     [317, 351],      # middle bottom corner
#     [563, 351],     # right bottom - bottom corner
#     [629, 293],     # right bottom - top corner
#     [628, 3],     # top right rorner
#     [8, 4],     # top left corner
#     [2, 299]        # left bottom - top corner
#     ])   

# cv2.fillPoly(img_dst, [pts_dst], 255)

# plt.figure()
# plt.imshow(img_dst)
# plt.show()

# Calculate Homography
# h, status = cv2.findHomography(pts_src, pts_dst)
im1homo, status = cv2.findHomography(pts_srcimg1, pts_dst_bottom_wMid)
im2homo, status = cv2.findHomography(pts_srcimg2, pts_dst_top_wMid)
  
# Warp source image to destination based on homography
# img_src2 = cv2.imread(img1)

# img_src2 = cv2.imread(image)
# img_out = cv2.warpPerspective(img_src2, h, (img_dst.shape[1], img_dst.shape[0]))

img_out = cv2.warpPerspective(img2, im2homo, (img_dst.shape[1], img_dst.shape[0]))
# img_out = cv2.warpPerspective(img1, im1homo, (img_dst.shape[1], img_dst.shape[0]))
cv2.imshow("Warped", img_out)
cv2.waitKey(0)

cv2.imwrite("/Users/tyler/Documents/GitHub/basketballVideoAnalysis/homography-mapping/output/courtViewResult.jpg", img_out)		