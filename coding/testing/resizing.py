import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("../images/straightRd.jpg")
img2 = cv2.imread("../images/rightTurn1.jpg")


cv2.imshow("image", img1)
cv2.imshow('image resized', cv2.resize(img1, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()

roi = np.float32(
    [[0, 4160], [0.125 * img1.shape[1], 0.5 * img1.shape[0]], [0.875 * img1.shape[1], 0.5 * img1.shape[0]],
     [3120, 4160]])
final = np.float32([[0, 300], [0, 0], [300, 0], [300, 300]])

m = cv2.getPerspectiveTransform(roi, final)

dst = cv2.warpPerspective(img1, m, (300, 300))
dst2 = cv2.warpPerspective(img2, m, (300, 300))

# plt.subplot(121), plt.imshow(img1), plt.title('Input1')
plt.subplot(121), plt.imshow(dst), plt.title('Output1')
# plt.subplot(121), plt.imshow(img2), plt.title('Input2')
plt.subplot(122), plt.imshow(dst2), plt.title('Output2')
plt.show()
