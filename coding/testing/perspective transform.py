import cv2 as cv
import numpy as np

img = cv.imread("../images/plainroad1.jpg")
img = cv.resize(img, None, fx=0.1, fy=0.1)

# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

height, width = img.shape[0], img.shape[1]

cv.circle(img, (int(width / 5),int(height / 2)), 3, (0, 0, 0), -1)
cv.circle(img, (int(4 * width / 5),int(height / 2)), 3, (0, 0, 0), -1)
cv.circle(img, (0,height), 3, (0, 0, 0), -1)
cv.circle(img, (width,height), 3, (0, 0, 0), -1)

src = []

cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()
