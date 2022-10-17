import cv2 as cv
import numpy as np
import functions
import matplotlib.pyplot as plt
import math

# img = cv.imread("../images/dashcam.png")
# img = cv.imread("../images/road7.jpg")
img = cv.imread("../../rpiimages/test2.jpg")

img = cv.resize(img, (1024, 581))

cv.imshow("photo", functions.HoughLinesPPoints(img,200,200))
cv.waitKey()
cv.destroyAllWindows()


finalImg = np.copy(img)
height, width, _ = img.shape

img = functions.ROI(img, 5)
img = functions.Blur(img)

x, y = functions.GetCoordinates(img, 150, 150)

leftLaneX = []
leftLaneY = []
rightLaneX = []
rightLaneY = []

# FINDING THE STARTING POINTS OF THE LANES
xBottom = [value for value in x if y[x.index(value)] > 28 * height / 30]
xBottomL = [value for value in xBottom if value < (width / 2)]
xBotttomR = [value for value in xBottom if value > (width / 2)]

# calculated starting points of the lane
leftCenter = int(np.mean(xBottomL))
rightCenter = int(np.mean(xBotttomR))
# Predefined starting points of the lane
# leftCenter = int(1.75 * width / 10)
# rightCenter = int(8.25 * width / 10)

rectWidth = int(width / 30)
rectHeight = int(height / 30)

topLimit = int(height - rectHeight)
bottomLimit = int(height)

functions.HoughLinesPPoints(img, 150, 150)

for i in range(15):
    rightLimitL = int(leftCenter + rectWidth)
    leftLimitL = int(leftCenter - rectWidth)

    rightLimitR = int(rightCenter + rectWidth)
    leftLimitR = int(rightCenter - rectWidth)

    xCoordL = [value for value in x if rightLimitL >= value >= leftLimitL]
    xCoordL = [value for value in xCoordL if bottomLimit >= y[x.index(value)] >= topLimit]
    yCoordL = [y[x.index(value)] for value in xCoordL if
               bottomLimit >= y[x.index(value)] >= topLimit]

    xCoordR = [value for value in x if rightLimitR >= value >= leftLimitR]
    xCoordR = [value for value in xCoordR if bottomLimit >= y[x.index(value)] >= topLimit]
    yCoordR = [y[x.index(value)] for value in xCoordR if
               bottomLimit >= y[x.index(value)] >= topLimit]

    cv.rectangle(img, (rightLimitL, bottomLimit), (leftLimitL, topLimit), (0, 255, 255))
    cv.rectangle(img, (rightLimitR, bottomLimit), (leftLimitR, topLimit), (0, 0, 255))

    if len(xCoordL) != 0:
        xMeanL = int(sum(xCoordL) / len(xCoordL))
        cv.line(img, (xMeanL, topLimit), (xMeanL, bottomLimit), (255, 255, 255))
        cv.line(img, (leftCenter, topLimit), (leftCenter, bottomLimit), (255, 0, 255))

        if i != 0 and i != 16:
            for j in range(len(xCoordL)):
                leftLaneX.append(xCoordL[j])
                leftLaneY.append(yCoordL[j])

        leftCenter = xMeanL
    else:
        bufferL = width / 50
        leftCenter += int(bufferL)

    if len(xCoordR) != 0:
        xmeanR = int(sum(xCoordR) / len(xCoordR))
        cv.line(img, (xmeanR, topLimit), (xmeanR, bottomLimit), (255, 255, 255))
        cv.line(img, (xmeanR, topLimit), (xmeanR, bottomLimit), (255, 255, 255))

        if i != 0 or i != 16:
            for j in range(len(xCoordR)):
                rightLaneX.append(xCoordR[j])
                rightLaneY.append(yCoordR[j])

        rightCenter = xmeanR
    else:
        bufferR = -(width / 50)
        rightCenter += int(bufferR)

    bottomLimit -= rectHeight
    topLimit -= rectHeight

# REGRESSION
try:
    if len(leftLaneX) != 0:
        leftLaneFound = True
    else:
        leftLaneFound = False
    if len(rightLaneX) != 0:
        rightLaneFound = True
    else:
        rightLaneFound = False

    if leftLaneFound:
        gradL, interceptL = functions.Regression(leftLaneX, leftLaneY)
        xTestL = [1, width / 2]
        yTestL = np.multiply(xTestL, gradL) + interceptL
        # cv.line(img, (int(xTestL[0]), int(yTestL[0])), (int(xTestL[1]), int(yTestL[1])), (255, 255, 0), 3)
        cv.line(finalImg, (int(xTestL[0]), int(yTestL[0])), (int(xTestL[1]), int(yTestL[1])), (0, 255, 0), 3)
    else:
        print("No left points found")

    if rightLaneFound:
        gradR, interceptR = functions.Regression(rightLaneX, rightLaneY)
        xTestR = [width / 2, width]
        yTestR = np.multiply(xTestR, gradR) + interceptR
        # cv.line(img, (int(xTestR[0]), int(yTestR[0])), (int(xTestR[1]), int(yTestR[1])), (255, 255, 0), 3)
        cv.line(finalImg, (int(xTestR[0]), int(yTestR[0])), (int(xTestR[1]), int(yTestR[1])), (0, 255, 0), 3)
    else:
        print("No right points found")

    if leftLaneFound and rightLaneFound:
        angle = round(gradL + gradR, 2)
        print(angle)

    xTurn = [width / 2, (width / 2) + (height * math.tan(math.radians(angle)))]
    yTurn = [height, 0]

    # cv.line(img, (int(xTurn[0]), int(yTurn[0])), (int(xTurn[1]), int(yTurn[1])), (255, 255, 255), 2)
    cv.line(finalImg, (int(xTurn[0]), int(yTurn[0])), (int(xTurn[1]), int(yTurn[1])), (255, 255, 255), 2)

    cv.imshow("image", img)
    cv.imshow("Output", finalImg)
    cv.waitKey()
    cv.destroyAllWindows()
except:
    print("error during regression calculation")
    cv.imshow("image", img)
    cv.waitKey()
    cv.destroyAllWindows()
