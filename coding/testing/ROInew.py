import cv2 as cv
import numpy as np

img = cv.imread("../images/dashcam.png")

height, width, _ = img.shape
mask = np.zeros((height, width), np.uint8)

pts = np.array([[0, height], [width, height], [2 * width / 3, height / 2], [width / 3, height / 2]],
               np.int32)
cv.fillConvexPoly(mask, pts, (255, 255, 255))

cv.imshow("mask", mask)
cv.waitKey(0)
cv.destroyAllWindows()

newImg = cv.bitwise_and(img, img, mask=mask)

cv.imshow("image", img)
cv.imshow("ROI", newImg)
cv.waitKey()
cv.destroyAllWindows()
