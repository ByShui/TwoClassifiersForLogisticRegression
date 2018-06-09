
# coding: utf-8
#Small Batch Stochastic Gradient Ascent


import numpy as np
import matplotlib.pyplot as plt


#将txt文件中储存的数据和标签分别存储在列表dataMat和labelMat中
def loadDataSet(filename):
    dataList = []
    labelList = []
    fr = open(filename)
    for line in fr.readlines():
        #将每一行的各个元素取出存放在列表lineArr中
        lineArr = line.strip().split()
        #[ , , ]中三个参数代表了公式 z = W^T X中的X，第一个X的值为1
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelList.append(int(lineArr[2]))
    return dataList, labelList


#sigmoid函数，用于分类
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#small Batch Stochastic Gradient Ascent，小批量随机梯度上升求权重；maxCycles表示梯度上升算法的最大迭代次数
def sbsGradientAscent(filename, maxCycles = 300):
    dataList, labelList = loadDataSet(filename)
    m, n = np.shape(dataList)
    weights = np.ones(n)
    for i in range(maxCycles):
        dataIndex = range(m)
        for j in range(m):
            #alpha表示步长
            alpha = 4 / (1.0 + i + j) + 0.001
            #uniform()表示在参数范围内随机取值
            randomIndex = int(np.random.uniform(0, len(dataIndex)))
            error = labelList[randomIndex] - sigmoid(sum(dataList[randomIndex] * weights))
            weights = weights + alpha * error * np.array(dataList[randomIndex])
            #从列表中移除刚刚被随机选取的值
            del(list(dataIndex)[randomIndex])
    return weights


def decisionBoundary(weights, filename):
    dataMat, labelMat = loadDataSet(filename)
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    
    type0_x = []; type0_y = []
    type1_x = []; type1_y = []

    for i in range(n):
        if labelMat[i] == 0:
            type0_x.append(dataMat[i][1])
            type0_y.append(dataMat[i][2])
        if labelMat[i] == 1:
            type1_x.append(dataMat[i][1])
            type1_y.append(dataMat[i][2])

    fig = plt.figure(figsize = (8, 4))
    ax = fig.add_subplot(111)

    type0 = ax.scatter(type0_x, type0_y, s = 30, c = 'r')
    type1 = ax.scatter(type1_x, type1_y, s = 30, c = 'b')
    
    x1 = np.arange(-4.5, 4.5, 0.1)
    x2 = (-weights[0]-weights[1]*x1) / weights[2]

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend((type0, type1), ("Class 0", "Class 1"), loc=0)
    ax.plot(x1, x2)
    plt.show()

if __name__ == "__main__":
    print("Code Run as a Program!")
