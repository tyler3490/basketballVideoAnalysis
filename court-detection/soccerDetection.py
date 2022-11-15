

# import cv2
# from matplotlib import pyplot as plt
# import numpy

# im = cv2.imread("/Users/tyler/Documents/GitHub/basketballVideoAnalysis/ApQFp.png")
# # im = cv2.imread("foot.png")
# B = im[:,:,2]
# Y = 255-B

# thresh = cv2.adaptiveThreshold(Y,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY_INV,35,5)

# contours, hierarchy = cv2.findContours(thresh,  
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

# x=[]
# for i in range(0, len(contours)):
#     if cv2.contourArea(contours[i]) > 100:
#         x.append(contours[i])
# cv2.drawContours(im, x, -1, (255,0,0), 2) 

# plt.imshow(im)

import cv2
import numpy as np

# video = "/Users/tyler/Documents/GitHub/basketballVideoAnalysis/soccerdemo.mp4"
video = "/Users/tyler/Documents/GitHub/basketballVideoAnalysis/soccercamerapan.mp4"
cap = cv2.VideoCapture(video)
while(cap.isOpened()): 
    ret, frame = cap.read()
    if ret == True:

        frame = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255)) # green mask to select only the field
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)

        gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        canny = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Hough line detection
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)
        # Line segment detection
        lsd = cv2.createLineSegmentDetector(0)
        lines_lsd = lsd.detect(canny)[0]
        #Draw detected lines in the image
        drawn_img = lsd.drawSegments(frame,lines)

        #Show image
        cv2.imshow("LSD",drawn_img )

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break
    else:
        break
cap.release() 
cv2.destroyAllWindows()