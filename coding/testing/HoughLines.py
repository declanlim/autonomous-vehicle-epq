import cv2 as cv
import numpy as np
import functions

# reads in the image
img = cv.imread("../images/dashcam.png")
img = functions.ROI(img)
img = functions.Blur(img)
canny = cv.Canny(img, 200, 200)

lines = cv.HoughLines(canny, 1, np.pi / 180, 150)
points = cv.HoughLinesP(canny, 1, np.pi/180, 100)

# for line in lines:
#     for rho, theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv.line(img, (x1, y1), (x2, y2), (0, 255, 00), 2)

for set in points:
    for x1, y1, x2, y2 in set:
        cv.line(img, (x1,y1),(x2,y2),(255,0,255),2)


cv.imshow("image", img)
cv.waitKey()
cv.destroyAllWindows()
