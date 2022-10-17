import numpy as np
import matplotlib.pyplot as plt

rightLaneX = [[791], [792], [810], [787], [730], [731], [722], [703], [684], [697], [624], [618], [632], [625], [599],
              [607], [618], [605], [600], [546]]
rightLaneY = [[537], [538], [534], [533], [474], [475], [465], [446], [427], [425], [367], [353], [365], [367], [342],
              [342], [353], [340], [342], [289]]

leftLaneX = [[216], [216], [194], [237], [291], [275], [320], [318], [359], [357], [416], [417], [442], [433], [446],
             [464], [448], [456], [470]]
leftLaneY = [[542], [542], [540], [499], [450], [464], [423], [424], [401], [401], [337], [336], [320], [327], [309],
             [298], [309], [301], [289]]

xMeanR = np.mean(rightLaneX)
yMeanR = np.mean(rightLaneY)
xStdR = np.std(rightLaneX)
yStdR = np.std(rightLaneY)

xMeanL = np.mean(leftLaneX)
yMeanL = np.mean(leftLaneY)
xStdL = np.std(leftLaneX)
yStdL = np.std(leftLaneY)

gradL = (np.corrcoef(np.transpose(leftLaneX), np.transpose(leftLaneY))[1, 0]) * (yStdL / xStdL)
gradR = (np.corrcoef(np.transpose(rightLaneX), np.transpose(rightLaneY))[1, 0]) * (yStdR / xStdR)

interceptL = yMeanL - (gradL * xMeanL)
interceptR = yMeanR - (gradR * xMeanR)

testXL = np.linspace(0,500,501)
testYL = (testXL * gradL) + interceptL

testXR = np.linspace(500,800,501)
testYR = (testXR * gradR) + interceptR