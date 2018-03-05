
"""
手写体数字识别问题算法"""
import gzip
import random
import _pickle as cPickle
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import operator
class logisticRegModel(object):
    """
    Desc: 逻辑回归模型
    Params: 
        None
    Return: 
        weights: 模型参数 size:10*784
    """

    def __init__(self, weights, eta, lam, mini_batch_size, epochs):
        """
        Desc: 构造函数
        Params: 
            weights: size 784*10
        Return: 
            None
        """
        self.weights = weights
        self.eta = eta
        self.lam = lam
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs

    def h(self, x):
        """
        Desc: 假设函数  注意处理exp上溢出问题每一行需要减去改行的最大值！！！
        Params: 
            x: 输入
        Return: 
            yhat: 估计值
        """
        theta = np.dot(self.weights.T, x)  # 10*784  784*m
        theta -= np.max(theta, axis = 0)  # 求每一列的最大值
        yhat = np.exp(theta)
        sumy = np.sum(yhat, axis = 0) 
        yhat = yhat / sumy
        return yhat 
    def calTotalLoss(self, x, y):
        """
        Desc: 计算cost
        Params: 
            data: 输入样本
            lam: 正则化参数
            convert: 标示位 False表示输入样本为训练样本 True表示输入样本为验证集或测试集样本
        Return: 
            cost: 定义的损失
        """
        theta = np.dot(self.weights.T, x) # 10*m
        l1 = np.multiply(theta, y)
        thetaMax = np.max(theta, axis = 0)
        theta -= thetaMax
        l2 = np.sum(np.log(np.exp(theta)), axis = 0)
        loss = np.sum(-l1 + l2 + thetaMax)
        loss = loss / float(x.shape[1]) + 0.5 * self.lam * np.linalg.norm(self.weights)**2
        return loss
    def calAccuracy(self, x, y):  # y 10*m 
        yhats = self.h(x)
        results = [(np.argmax(yhat), np.argmax(label)) for yhat, label in zip(yhats.T, y.T)]
        accuracy = np.sum([int(x==y) for (x, y) in results])
        accuracy = accuracy / float(y.shape[1]) * 100.0
        return accuracy 
    def calGradient(self, x, y):
        """
        Desc:计算梯度函数
        Params: 
            self: logisticRegModel对象
            x: 训练样本输入
            y: 样本标签
        Return: 
            grad: 梯度
        """
        yhat = self.h(x)
        grad = 1.0 / x.shape[1] * ((yhat - y) * x.T).T + self.lam * self.weights
        return grad
    def SGDTrain(self, tr_d, evaluation_data):
        """
        Desc: 随机小批量梯度下降
        Params: 
            training_data: 训练数据集
            epochs: 迭代次数
            mini_batch_size: 小批量数据集大小
            eta: 学习率
            lam: 正则化参数
            evaluation_data: 验证集或测试集数据
        """
        trainData, trainLabel = tr_d
        trainData = np.insert(trainData, 0, values = np.ones(trainData.shape[0]), axis = 1)
        m, n = trainData.shape
        d_class = len(list(set(trainLabel)))
        x_train = np.transpose(np.matrix(trainData))
        y_train = np.zeros((d_class, m))
        y_train[trainLabel, np.arange(m)] = 1
        if (evaluation_data):
            evData, evLabels = evaluation_data
            evData = np.insert(evData, 0, values = np.ones(evData.shape[0]), axis = 1)
            evData = np.transpose(np.matrix(evData))
            evy = np.zeros((d_class, evLabels.shape[0]))
            evy[evLabels, np.arange(evLabels.shape[0])] = 1
        Max_iter = self.epochs
        loss_thresh = 1e-4
        Loss_old = 0
        Loss = []; ev_Loss = []; train_acc = []; ev_acc = []
        stepCnt = 0
        for iter in range(Max_iter):
            batchIndex = np.arange(m)
            np.random.shuffle(batchIndex)
            batches = [
                x_train[:, batchIndex[k : k + self.mini_batch_size]] for k in np.arange(0, m, self.mini_batch_size)
            ]
            labels =[
                y_train[:, batchIndex[k : k + self.mini_batch_size]] for k in np.arange(0, m, self.mini_batch_size)
            ]
            for batch, label in zip(batches, labels):
                # 计算梯度
                grad = self.calGradient(batch, label)
                self.weights = self.weights - 1.0 / batch.shape[1] * self.eta * grad  
            #计算损失
            loss = self.calTotalLoss(x_train, y_train)
            Loss.append(loss)
            acc = self.calAccuracy(x_train, y_train)
            train_acc.append(acc)
            if (evaluation_data):
                loss = self.calTotalLoss(evData, evy)
                ev_Loss.append(loss)
                acc = self.calAccuracy(evData, evy)
                ev_acc.append(acc)
            if (abs(Loss_old - Loss[-1]) < loss_thresh): break
            Loss_old = Loss[-1]
            stepCnt += 1
            if stepCnt == 10:
                stepCnt = 0
                self.eta *= 0.8
                #lr*=0.8
            print(("Epoch: {} completed").format(iter))
        return self.weights, Loss, ev_Loss, train_acc, ev_acc
    def test(self, te_d):
        testData, testLabel = te_d
        n = testData.shape[0]
        testData = np.insert(testData, 0, values = np.ones(n), axis = 1)
        x_test = np.transpose(np.matrix(testData))
        y_test = np.array(testLabel)
        mu =  self.h(x_test)    # 10*m
        results = [(np.argmax(yhat), label) for yhat, label in zip(mu.T, testLabel)]
        accuracy = np.sum([int(x==y) for (x,y) in results])
        accuracy = accuracy / float(n) * 100.0 
        print(("Accuracy for test set is {}").format(accuracy))
        return accuracy 

def logisticMainFun():
    tr_d, va_d, te_d = loadData()
    eta = 0.5
    lam = 0.0
    mini_batch_size = 10000
    epochs = 200
    weights = np.zeros((785, 10))
    model = logisticRegModel(weights, eta, lam, mini_batch_size, epochs)
    weights, Loss, ev_Loss, train_acc, ev_acc = model.SGDTrain(tr_d, te_d) 
    plt.figure(111)
    plt.plot(Loss, label = "Loss on training data")
    plt.plot(ev_Loss, label = "Loss on test data")
    plt.xlabel("Epoch")
    plt.legend(loc = "best")
    plt.show()
    plt.figure(222)
    plt.plot(train_acc, label = "Accuracy on training data")
    plt.plot(ev_acc, label = "Accuracy on test data")
    plt.xlabel("Epoch")
    plt.legend(loc = "best")
    plt.show()

class knnModel(object):
    """
    Desc: k近邻算法模型
    """

    def __init__(self, k):
        """
        Desc: 初始化k近邻模型
        """
        self.k = k

    def knnClassifier(self, trainData, trainLabel, testData):
        """
        Desc: k近邻分类函数
        Params: 
            trainData: 训练集数据
            trainLabel：训练集标签
            testData: 测试样本
        Return:
            predictLabel: 预测标签
        """
        m, n = trainData.shape
        diffMat = trainData - testData
        sqDiffMat = diffMat**2
        distance = sqDiffMat.sum(axis=1)**0.5  # 按行求和求出测试样本和训练样本集的距离
        sortedDisInd = np.argsort(distance)  # 排序
        classCount = {}
        for i in range(self.k):
            voteLabel = trainLabel[sortedDisInd[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True)
        predictLabel = sortedClassCount[0][0]
        return predictLabel  #返回票数最多的标签


def knnModelTrain():
    tr_d, va_d, te_d = loadData()
    trainData, trainLabel = tr_d
    testData, testLabel = te_d
    accuracy = 0.0
    kList = [1, 3, 5, 10, 20]  #
    for k in kList:
        knn = knnModel(k)
        for data, label in zip(testData, testLabel):
            result = knn.knnClassifier(trainData, trainLabel, data)
            if result == label:
                accuracy += 1.0
        accuracy = accuracy / testData.shape[0] * 100.0
        print("The k of knn model is {} and accuracy is {}".format(
            k, accuracy))



class SVM(object):
    def SMO(self, C, tol, max_passes, trainingData):
        """
        Desc: SMO算法求解支持向量机问题 二分类问题 
        Paramas: 
            C: 正则化参数
            tol: 一般求解优化问题的容限值dat
            max_passes: 当alpha不发生变化的最大迭代次数
            trainingData:(x1,y1),...(xm,ym)为训练样本集
        Returns: 
            alphas: 拉格朗日乘子
            b: 截距项
        """
        m = len(trainingData)
        trainData = [ x for x, y in trainingData]
        trainLabels=[ y for x, y in trainingData]
        b = 0
        alphas = [0] * m
        passes = 0
        while (passes < max_passes):
            num_changed_alphas = 0
            for i in range(m):
                # calculate Ei
                xi, yi= trainingData[i]
                fxi = np.sum([float(alpha)*y*x.T*xi for x, y, alpha in zip(trainData, trainLabels, alphas)])
                Ei = fxi - float(yi)
                if ((yi * Ei < -tol and alphas[i]< C) or (yi * Ei > tol and alphas[i] > 0)):
                    # select j random such that j \= i
                    j = i
                    while (j == i):
                        j = int(random.uniform(0, m))
                    # calculate Ej
                    xj, yj = trainData[j]
                    fxj = np.sum([float(alpha)*y*x.T*xj for x, y, alpha in zip(trainData, trainLabels, alphas)])
                    Ej = fxj - float(yj)
                    # save old alpha 
                    alphaiOld = alphas[i].copy()
                    alphajOld = alphas[j].copy()
                    # 
                    if (yi != yj):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - C)
                        H = min(C, alphas[i] + alphas[j])
                    if (L == H): print("L==H"); continue  # if L==H continue to next i
                    # calculate eta 
                    eta = 2 * xi.T * xj -xi.T * xi - xj.T * xj 
                    if (eta >= 0): print("eta >= 0"); continue
                    # Compute and clip new value for alphaj
                    alphas[j] -= yj * (Ei - Ej) / eta
                    if (alphas[j] > H):
                        alphas[j] = H
                    if (alphas[j] < L):
                        alphas[j] = L
                    if (abs(alphas[j] - alphajOld) < 10e-5):
                        print("j not moving enough")
                        continue
                    # Determine value for alphai
                    alphas[i] += yi * yj * (alphajOld - alphas[j])
                    # Compute b1 and b2 and b
                    b1 = b - Ei - yi * (alphas[i] - alphaiOld) * \
                    (xi.T * xi) - yj * (alphas[j] - alphaiOld) * (xi.T * xj)
                    b2 = b - Ej - yi * (alphas[i] - alphaiOld) * \
                    (xi.T * xj) - yj * (alphas[j] - alphajOld) * (xj.T * xj)
                    if (alphas[i] > 0 and alphas[i] < C):
                        b = b1
                    if (alphas[j] > 0 and alphas[j] < C):
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    num_changed_alphas += 1
                    print(("iter: {0} i: {1}, pair changed {2}").format(passes, i, num_changed_alphas))
            if (num_changed_alphas == 0):
                passes += 1
            else:
                passes = 0
            print(("iteration number: {}").format(passes))
        return alphas, b
       

def loadDataSet(filename):
    dataMat=[]; labelMat=[]
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
def loadData():
    """
    Desc: 加载数据集函数
    Params: 
        无
    Return: 
        training_data: 训练样本 tuple(x,y) of ndarray类型
        validation_data: 验证集样本 tuple of ndarray类型
        test_data: 测试集样本 tuple of ndarray类型
    """
    f = gzip.open(
        r"E:\machinelearninginaction\digitRecognition\data\mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = cPickle.load(
        f, encoding="bytes")
    f.close()
    return (training_data, validation_data, test_data)


# test git
def loadDataWrapper():
    """
    Desc: 加载数据装饰器函数
    Params: 
        None
    Return: 
        同loadData
    """
    tr_d, va_d, te_d = loadData()
    training_inputs = [np.reshape(x, (784, 1))
                       for x in tr_d[0]]  # 将tuple转成list
    traing_results = [vectorized_result(y)
                      for y in tr_d[1]]  #将tuple中的标签转成list，训练集需要做oneHot编码
    training_data = zip(training_inputs, traing_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (list(training_data), list(validation_data), list(test_data))


def vectorized_result(j):
    """
    Desc: 对数据标签做OneHot编码
    Params: 
        j: 代表类别标签
    Return: 
        e: 10*1向量
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    #git test
    #git test2
    #git test3