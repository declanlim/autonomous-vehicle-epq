import cv2 as cv
import numpy as np
import math


# this file stores functions that are used in the main file "laneDetection.py"

def ROI(img, divisions):
    # stores the height and width of the image
    height, width, _ = img.shape
    # creates a black image with the same dimensions as the original image
    mask = np.zeros((height, width), np.uint8)
    # sets fills the ROI with white
    pts = np.array([[0, height], [width, height], [4 * width / 5, height / 2], [width / 5, height / 2]],
                   np.int32)
    cv.fillConvexPoly(mask, pts, (255, 255, 255))
    # performs a bitwise and to isolate the ROI
    newImg = cv.bitwise_and(img, img, mask=mask)
    return newImg


def Blur(img):
    # applies a kernel blur to the image to reduce sharp edges
    blur = cv.blur(img, (3, 3))
    # blur = cv.GaussianBlur(img, (3,3),10)
    return blur


def HoughLinesP(img, thresh1, thresh2):
    # applies the canny edge detector on the image
    dst = cv.Canny(img, thresh1, thresh2)

    # applies the probabilistic Hough line transform on the result of the canny edge detector
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 100, None)

    # For each pair of coordinates received, lines are drawn between the points
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    return img


def HoughLinesPPoints(img, thresh1, thresh2):
    # applies a Canny edge detector to the image
    dst = cv.Canny(img, thresh1, thresh2)

    # applies the probabilistic hough transform on the result of the canny edge detector
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 100, None)

    # For each of the pairs of coordinates received, a point (small circle) is printed to the image
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.circle(img, (l[0], l[1]), 2, (0, 255, 0), -1)
            cv.circle(img, (l[2], l[3]), 2, (0, 255, 0), -1)

    return img


def GetCoordinates(img, thresh1, thresh2):
    # applies a Canny edge detector to the image
    dst = cv.Canny(img, thresh1, thresh2)
    # applies the probabilistic hough transform on the result of the canny edge detector
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 100, None)
    # creates two arrays to store the coordinates
    x = []
    y = []

    # Stores the x and y coordinates in the two arrays
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x.append(l[0])
            x.append(l[2])
            y.append(l[1])
            y.append(l[3])
    return x, y


def Cost(x, y, theta):
    m = np.shape(x)[0]
    hyp = theta[0, 0] + (theta[1, 0] * x)
    J = (1 / (2 * m)) * ((hyp - y).transpose() * (hyp - y))
    return J


def Regression(x, y):
    xMean = np.mean(x)
    yMean = np.mean(y)
    xStd = np.std(x)
    yStd = np.std(y)

    grad = np.corrcoef(np.transpose(x), np.transpose(y))[1, 0] * (yStd / xStd)
    intercept = yMean - (grad * xMean)
    return grad, intercept


def RoadAngle(img):
    img = cv.resize(img, (1024, 581))

    # finalImg = np.copy(img)
    height, width, _ = img.shape

    img = ROI(img, 5)
    img = Blur(img)

    x, y = GetCoordinates(img, 150, 150)

    leftLaneX = []
    leftLaneY = []
    rightLaneX = []
    rightLaneY = []

    # FINDING THE STARTING POINTS OF THE LANES
    xBottom = [value for value in x if y[x.index(value)] > 29 * height / 30]
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

    HoughLinesPPoints(img, 150, 150)

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
            gradL, interceptL = Regression(leftLaneX, leftLaneY)
            xTestL = [1, width / 2]
            yTestL = np.multiply(xTestL, gradL) + interceptL
            # cv.line(img, (int(xTestL[0]), int(yTestL[0])), (int(xTestL[1]), int(yTestL[1])), (255, 255, 0), 3)
            # cv.line(finalImg, (int(xTestL[0]), int(yTestL[0])), (int(xTestL[1]), int(yTestL[1])), (0, 255, 0), 3)
        else:
            print("No left points found")

        if rightLaneFound:
            gradR, interceptR = Regression(rightLaneX, rightLaneY)
            xTestR = [width / 2, width]
            yTestR = np.multiply(xTestR, gradR) + interceptR
            # cv.line(img, (int(xTestR[0]), int(yTestR[0])), (int(xTestR[1]), int(yTestR[1])), (255, 255, 0), 3)
            # cv.line(finalImg, (int(xTestR[0]), int(yTestR[0])), (int(xTestR[1]), int(yTestR[1])), (0, 255, 0), 3)
        else:
            print("No right points found")

        if leftLaneFound and rightLaneFound:
            angle = round(gradL + gradR, 2)
            # print(angle)

        xTurn = [width / 2, (width / 2) + (height * math.tan(math.radians(angle)))]
        yTurn = [height, 0]

        cv.line(img, (int(xTurn[0]), int(yTurn[0])), (int(xTurn[1]), int(yTurn[1])), (255, 255, 255), 2)
        # cv.line(finalImg, (int(xTurn[0]), int(yTurn[0])), (int(xTurn[1]), int(yTurn[1])), (255, 255, 255), 2)

        # cv.imshow("image", img)
        # cv.imshow("Output", finalImg)
        # cv.waitKey()
        # cv.destroyAllWindows()

        return angle
    except:
        return False
