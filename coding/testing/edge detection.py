import cv2 as cv
import numpy as np
import math

img1 = cv.imread("../images/dashcam.png")

# img1 = cv.resize(img1, None, fx=0.1, fy=0.1)


dst = cv.Canny(img1, 50, 200)

# cv.imshow("image", img1)
# cv.imshow("canny", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv.HoughLines(dst, 1, np.pi / 180, 115, None, 0, 0)


# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
#         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
#         cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
#
linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 100, None)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

cv.imshow("Source", img1)
cv.imshow("Standard Hough Line Transform", dst)
cv.imshow("Probabilistic Line Transform", cdstP)

cv.waitKey()
cv.destroyAllWindows()

# # code from https://docs.opencv.org/3.4.0/d9/db0/tutorial_hough_lines.html
