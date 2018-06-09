import matplotlib.pyplot as plt
import numpy as np

import BpGradientAscent
import SbsGradientAscent
import TxtToNumpy

#散点图
dataMat, labelList = TxtToNumpy.TxtToNumpy("testSet.txt")

type0_x = []; type0_y = []
type1_x = []; type1_y = []

for i in range(len(labelList)):
    if labelList[i] == 0:
        type0_x.append(dataMat[i][0])
        type0_y.append(dataMat[i][1])
    if labelList[i] == 1:
        type1_x.append(dataMat[i][0])
        type1_y.append(dataMat[i][1])

fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(111)

type0 = ax.scatter(type0_x, type0_y, s = 30, c = 'r')
type1 = ax.scatter(type1_x, type1_y, s = 30, c = 'b')

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend((type0, type1), ("Class 0", "Class 1"), loc=0)
plt.show()


#使用批处理梯度上升画决策边界
#Batch Processing Gradient Ascent
BpGradientAscent.decisionBoundary(BpGradientAscent.bpGradientAscent("testSet.txt"), "testSet.txt")

#使用小批量随机梯度上降画决策边界
#small Batch Stochastic Gradient Ascent
SbsGradientAscent.decisionBoundary(SbsGradientAscent.sbsGradientAscent("testSet.txt"), "testSet.txt")