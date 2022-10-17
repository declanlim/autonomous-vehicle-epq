import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import functions


def cost(x, y, theta):
    m = np.shape(x)[0]
    hyp = theta[0, 0] + (theta[1, 0] * x)
    J = (1 / (2 * m)) * ((hyp - y).transpose() * (hyp - y))
    return J


def linearRegression(x, y, theta, alpha, numiters):
    m = np.shape(x)[0]
    for i in range(numiters):
        hyp = theta[0, 0] + (theta[1, 0] * x)
        temp1 = theta[0, 0] - (alpha * (1 / m) * np.sum(hyp - y))
        temp2 = theta[1, 0] - (alpha * (1 / m) * np.sum(np.multiply((hyp - y), x)))
        theta[0, 0] = temp1
        theta[1, 0] = temp2
        print(cost(x, y, theta))

    return theta


# theta = np.matrix([0, 1], dtype=float)
# alpha = 0.000004
#
# xMatR = np.matrix(
#     [[791], [792], [810], [787], [730], [731], [722], [703], [684], [697], [624], [618], [632], [625], [599], [607],
#      [618], [605], [600], [546]])
# yMatR = np.matrix(
#     [[537], [538], [534], [533], [474], [475], [465], [446], [427], [425], [367], [353], [365], [367], [342], [342],
#      [353], [340], [342], [289]])
#
# xMatL = np.matrix(
#     [[216], [216], [194], [237], [291], [275], [320], [318], [359], [357], [416], [417], [442], [433], [446], [464],
#      [448], [456], [470]])
#
# yMatL = np.matrix(
#     [[542], [542], [540], [499], [450], [464], [423], [424], [401], [401], [337], [336], [320], [327], [309], [298],
#      [309], [301], [289]])
#
# # xMat = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).transpose()
# # yMat = np.matrix([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).transpose()
#
# # for loop to loop through diferent values of alpha and see which gives the lowest cost
# # value of alpha that gives the lowest cost function  = 4.3 e-6
#
# # print("For the right lane")
# # for i in range(20):
# #     theta = linearRegression(xMatR, yMatR, theta, 0.000004 + 0.0000001 * i, 1000)
# #     hyp = theta[0, 0] + (theta[0, 1] * xMatR)
# #     print("learning rate: " + str(0.000004 + 0.0000001 * i))
# #     print("cost : %f" % (functions.cost(xMatR, yMatR, theta)))
# #     print("")
#
# print("For the left lane")
# for i in range(20):
#     theta = linearRegression(xMatL, yMatL, theta, 0.000001 + 0.000001 * i, 1000)
#     hyp = theta[0, 0] + (theta[0, 1] * xMatL)
#     print("learning rate: %f" % (0.000001 + 0.000001 * i))
#     print("cost : %f" % (functions.cost(xMatL, yMatL, theta)))
#     print("")
#
# # print(functions.cost(xMat, yMat, theta))
# # print(functions.cost(xMat, yMat, theta))
#
# # plt.scatter(xMat.tolist(), hyp.tolist())
# # plt.scatter(xMat.tolist(), yMat.tolist())
# # plt.show()
#
# # alpha = 0.000004
