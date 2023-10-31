import numpy as np
import math
import cv2

def nothing(val):
    pass

threshold_value = 200 # default value
blur_value = 3 # default value
canny_value = 50 # default value

# read image file
image = cv2.imread('2.png')

# image specs
height, width, channels = image.shape

# trackbar
scale_window = cv2.namedWindow('trackbar')
cv2.createTrackbar('threshold', 'trackbar', threshold_value, 255, nothing)
cv2.createTrackbar('blur', 'trackbar', blur_value, 10, nothing)
cv2.createTrackbar('Canny', 'trackbar', canny_value, 200, nothing)

while True:
    # create image mask
    mask = np.zeros_like(image)
    channel_count = image.shape[2]
    match_mask_color = (255,) * channel_count

    roi = [
        ((width/4), (height-height/3)),
        ((width/3), (height/4)),
        ((width-width/3), (height/4)),
        ((width-width/4), (height-height/3)),
    ]
    
    vertices = np.array([roi], np.int32)
    cv2.fillPoly(mask, vertices, match_mask_color)
    #image = cv2.bitwise_and(image, mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get trackbar info
    threshold_value = cv2.getTrackbarPos('threshold', 'trackbar')
    blur_value = cv2.getTrackbarPos('blur', 'trackbar')

    # preprocessing
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    blur = cv2.blur(thresh, (blur_value, blur_value))
    canny = cv2.Canny(blur, canny_value, 200, apertureSize=3)

    # detection
    lines = cv2.HoughLines(canny, 5, np.pi/20, 175)
    lines_image = image.copy()

    if lines is not None:
        for i in range(0, len(lines)):
            for r, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*r
                y0 = b*r
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    """ if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(lines_image, pt1, pt2, (0,255,0), 3, cv2.LINE_AA) """

    
    cv2.imshow('result image', thresh)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break