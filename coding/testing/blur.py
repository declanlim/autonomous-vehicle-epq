import cv2 as cv
import numpy as np

img = cv.imread("../images/dashcam.png")

blur = cv.blur(img, (5, 5))
gaussianBlur = cv.GaussianBlur(img, (5, 5), 1)

canny1 = cv.Canny(img, 50, 200)
canny2 = cv.Canny(gaussianBlur, 50, 200)
canny3 = cv.Canny(blur, 50, 200)


# cv.imshow("image", img)
# cv.imshow("blur", blur)
cv.imshow("imgcanny", canny1)

cv.imshow("gaussianblurcanny", canny2)
cv.imshow("blurcanny", canny3)
cv.waitKey()
cv.destroyAllWindows()
