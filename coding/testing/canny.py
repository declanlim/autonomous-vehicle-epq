import cv2 as cv
import time
import functions

start = time.clock()
img = cv.imread("../images/dashcam.png")

# normal is 50 for x and 200 for y
thresh1 = 50
thresh2 = 50

blur = functions.Blur(img)

# for i in range(1, 5):
#     for j in range(1,5):
#         img = cv.imread("../images/dashcam.png")
#         dst = cv.Canny(blur, thresh1 + 50*i, thresh2+50*j)
#         cv.imshow("image" + str(thresh1 + 50*i)+str(thresh2 + 50*j), dst)


dst = functions.HoughLinesPPoints(img, 100,120)

print(str(time.clock()-start) + "seconds")
cv.imshow("image", dst)
cv.waitKey(0)
cv.destroyAllWindows()
