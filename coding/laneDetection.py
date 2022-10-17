import functions
import cv2 as cv

# import time

# start = time.clock()
img = cv.imread("images/road7.jpg")

print(functions.RoadAngle(img))
cv.imshow("image", cv.resize(img, (1024, 581)))
cv.waitKey()
cv.destroyAllWindows()