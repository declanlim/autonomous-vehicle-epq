import functions
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def Cost(xMat, yMat, theta):
    m = len(xMat)
    hyp = theta[0, 0] + theta[1, 0] * xMat
    cost = (1 / 2 * m) * ((hyp - yMat) * np.matrix.transpose(hyp - yMat))[0, 0]
    return cost


img = cv.imread("../images/dashcam.png")

height, width, _ = img.shape
img = functions.ROI(img)
img = functions.Blur(img)
# creates a copy of the image to show the detected lane lines without steps
finalimg = np.copy(img)

x, y = functions.GetCoordinates(img, 250, 250)

# creates lists to store the points of the
leftLaneX = []
leftLaneY = []
rightLaneX = []
rightLaneY = []

# Height of each rectangle checked is 1/30 of the height of the image
# The height of the ROI is 2/3 of the image (20/30)
leftCenter = int(1.5 * width / 10)
rightCenter = int(8.25 * width / 10)
rectWidth = int(width / 30)
rectHeight = int(height / 30)

# sets the values of the edges of the original rectangle
# two pairs of limits are created for the left and right edges of each rectangle on each lane
rightLimitL = int(leftCenter + rectWidth)
leftLimitL = int(leftCenter - rectWidth)
rightLimitR = int(rightCenter + rectWidth)
leftLimitR = int(rightCenter - rectWidth)

# the top and bottom limits are the same for both rectangle
topLimit = int(height - rectHeight)
bottomLimit = int(height)

# Plots the points on the image to allow evaluation of algorithm
functions.HoughLinesPPoints(img, 250, 250)

# finds points detected on the lane and stores them in two lists
for i in range(16):
    # uses list comprehensions to select all points that fall within a small range
    # the ranges should be the natural staring point of the road from the view of the camera

    # selects the x coordinates that fall between the left and right boundaries
    # selects points from the above array that also fall between the top and bottom boundaries
    # selects the y coordinates that correspond to the selected x coordinates

    xCoordL = [value for value in x if value <= rightLimitL and value >= leftLimitL]
    xCoordL = [value for value in xCoordL if y[x.index(value)] <= bottomLimit and y[x.index(value)] >= topLimit]
    yCoordL = [y[x.index(value)] for value in xCoordL if
               y[x.index(value)] <= bottomLimit and y[x.index(value)] >= topLimit]

    xCoordR = [value for value in x if value <= rightLimitR and value >= leftLimitR]
    xCoordR = [value for value in xCoordR if y[x.index(value)] <= bottomLimit and y[x.index(value)] >= topLimit]
    yCoordR = [y[x.index(value)] for value in xCoordR if
               y[x.index(value)] <= bottomLimit and y[x.index(value)] >= topLimit]

    # rectangle shows area searched for points
    cv.rectangle(img, (rightLimitL, bottomLimit), (leftLimitL, topLimit), (0, 255, 255))
    cv.rectangle(img, (rightLimitR, bottomLimit), (leftLimitR, topLimit), (255, 255, 0))

    # stores the points that are on the lane in two lists
    # if there are points detected, calulates the mean of the points and moves the left and right edges of the rectangle
    if len(xCoordL) != 0:
        # stores the found coordinates in the left lane in a list
        for j in range(len(xCoordL)):
            # leftLaneX.append([1, xCoordL[j]])
            leftLaneX.append([xCoordL[j]])
            leftLaneY.append([yCoordL[j]])

        xMeanL = int(sum(xCoordL) / len(xCoordL))
        leftLimitL = (xMeanL - rectWidth)
        rightLimitL = (xMeanL + rectWidth)

    if len(xCoordR) != 0:
        # stores the found coordinates in the right lane in a list
        for j in range(len(xCoordR)):
            # rightLaneX.append([1, xCoordR[j]])
            rightLaneX.append([xCoordR[j]])
            rightLaneY.append([yCoordR[j]])

        xMeanR = int(sum(xCoordR) / len(xCoordR))
        leftLimitR = (xMeanR - rectWidth)
        rightLimitR = (xMeanR + rectWidth)

    # adjusts the limits of the lane to follow the path from the camera angle
    leftLimitL += int(rectWidth / 2)
    rightLimitL += int(rectWidth / 2)
    leftLimitR -= int(rectWidth / 2)
    rightLimitR -= int(rectWidth / 2)

    # decreases the top and bottom limits of the rectangle by the width of the rectangle
    bottomLimit -= rectHeight
    topLimit -= rectHeight

# LINEAR REGRESSION

thetaL = np.matrix([[0], [1]])
thetaR = np.matrix([[0], [1]])
mL = len(leftLaneX)
mR = len(rightLaneX)
xMatL = np.matrix(leftLaneX)
yMatL = np.matrix.transpose(np.matrix(leftLaneY))
xMatR = np.matrix(rightLaneX)
yMatR = np.matrix.transpose(np.matrix(rightLaneY))
alphaL = 0.0000014
alphaR = 0.0000043

thetaR = functions.linearRegression(xMatR, yMatR, thetaR, alphaR, 100)
thetaL = functions.linearRegression(xMatL, yMatL, thetaL, alphaL, 100)



# cv.imshow("testing", img)
# cv.waitKey()
# cv.destroyAllWindows()
