# pylint disable=C0303
"""
手写体数字识别问题及算法比"""
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
        Desc: 假设函数
        """
        x = np.mat(x)
        return np.exp(self.weights.T * x) / float(
            np.sum(np.exp(self.weights.T * x)))

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
        x = np.mat(x)
        y = np.mat(y)
        yhat = self.h(x)
        grad = ((yhat - y) * x.T).T
        return grad

    def update_mini_batch(self, mini_batch, n):
        """
        Desc: 小批量更新梯度
        Params: 
            self:
            mini_batch: 批量样本
            eta: 学习率
            lambda: 正则化系数
            n: 样本数
        Return: 
            None
        """
        nabla_w = np.mat(np.zeros((self.weights).shape))
        for x, y in mini_batch:
            delta_w = self.calGradient(x, y)
            nabla_w = nabla_w + delta_w
        self.weights = self.weights - self.eta * (
            self.lam / n) * self.weights - self.eta * nabla_w / len(mini_batch)

    def SGD(self, training_data, evaluation_data=None):
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
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size]
                for k in np.arange(0, n, self.mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, len(training_data))
                cost = self.calTotalCost(training_data)
                training_cost.append(cost)
                accuracy = self.calAccuracy(training_data)
                training_accuracy.append(accuracy)
                cost = self.calTotalCost(evaluation_data, convert=True)
                evaluation_cost.append(cost)
                accuracy = self.calAccuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
            print(("Epoch {} training complete").format(j))
        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy

    def calTotalCost(self, data, convert=False):
        """
        Desc: 计算cost
        Params: 
            data: 输入样本
            lam: 正则化参数
            convert: 标示位 False表示输入样本为训练样本 True表示输入样本为验证集或测试集样本
        Return: 
            cost: 定义的损失
        """
        cost = 0.0
        for x, y in data:
            x = np.mat(x)
            if convert:
                y = vectorized_result(y)
            y = np.mat(y)
            cost += np.multiply(y, np.log(self.h(x))) + 0.5 * self.lam * (
                np.linalg.norm(self.weights)**2)
        cost = cost / float(len(data))
        return cost

    def calAccuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.h(x)), y)
                       for (x, y) in data]  # np.argmax返回的是最大值所在的下标
        else:
            results = [(np.argmax(self.h(x)), np.argmax(y)) for (x, y) in data]
        return np.sum([int(x == y) for (x, y) in results])


def logisticTrainModel(eta, lam, mini_batch_size, epochs):
    training_data, validation_data, test_data = loadDataWrapper()
    weights = np.mat(np.random.randn(784, 10))
    logisticModel = logisticRegModel(weights, eta, lam, mini_batch_size,
                                     epochs)
    training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = logisticModel.SGD(
        training_data, validation_data)
    plt.figure(111)
    plt.plot([i for i in range(epochs)], training_cost, label="training cost")
    plt.plot(
        [i for i in range(epochs)], evaluation_cost, label="evaluation_cost")
    plt.xlabel("iterate times")
    plt.ylabel("cost")
    plt.legend(loc="best")
    plt.show()
    plt.figure(222)
    plt.plot(
        [i for i in range(epochs)], training_accuracy, label="training cost")
    plt.plot(
        [i for i in range(epochs)],
        evaluation_accuracy,
        label="evaluation_cost")
    plt.xlabel("iterate times")
    plt.ylabel("cost")
    plt.legend(loc="best")
    plt.show()


def logisticTrainModelAccelerated(eta, lam, mini_batch_size, epochs):
    tr_d, va_d, te_d = loadData()
    trainData, trainLabel = tr_d
    trainLabel = np.reshape(trainLabel,(trainLabel.shape[0], 1))
    vaData, vaLabel = va_d
    vaLabel = np.reshape(vaLabel, (vaLabel.shape[0], 1))
    testData, testLabel = te_d
    testLabel = np.reshape(testLabel, (testLabel.shape[0], 1))
    enc = preprocessing.OneHotEncoder()
    enc.fit(trainLabel)
    trainLabel = enc.transform(trainLabel).toarray()
    enc.fit(testLabel)
    testLabel = enc.transform(testLabel).toarray()
    n = trainData.shape[0]
    training_cost, training_accuracy, test_cost, test_accuracy = [], [], [], []
    weights = np.random.randn(784, 10)

    for i in range(epochs):
        random.shuffle(trainData)
        mini_batches = [
            trainData[k:k + mini_batch_size]
            for k in np.arange(0, n, mini_batch_size)
        ]
        labels = [
            trainLabel[k:k + mini_batch_size]
            for k in np.arange(0, n, mini_batch_size)
        ]
        for mini_batch, label in zip(mini_batches, labels):
            #批量更新梯度
            yhat = np.exp(np.dot(mini_batch, weights))
            sumy = np.sum(yhat, axis=1)
            yhat = yhat / np.reshape(sumy, (mini_batch.shape[0], 1))
            err = yhat - label
            weights = weights - 1.0 * eta / mini_batch.shape[0] * np.dot(
                err.T, mini_batch).T - 1.0 * eta * (lam / n) * weights
            # 计算损失
            yhat = np.exp(np.dot(trainData, weights))
            sumy = np.sum(yhat, axis=1)
            yhat = yhat / np.reshape(sumy, (trainData.shape[0], 1))
            results = [(np.argmax(y), np.argmax(label))
                       for (y, label) in zip(yhat, trainLabel)]
            accuracy = np.sum([int(x == y)
                               for (x, y) in results]) / float(n) * 100.0
            training_accuracy.append(accuracy)
            #cost = -1.0 * np.sum(np.log(yhat) * trainLabel) +　lam * np.linalg.norm(weights)
            cost = -1.0 * np.sum(
                np.log(yhat) * trainLabel) + lam * np.linalg.norm(weights)
            cost = 1.0 * cost / trainData.shape[0]
            training_cost.append(cost)
            # 计算测试损失

            yhat = np.exp(np.dot(testData, weights))
            sumy = np.sum(yhat, axis=1)
            yhat = yhat / np.reshape(sumy, (testData.shape[0], 1))
            results = [(np.argmax(y), np.argmax(label))
                       for (y, label) in zip(yhat, testLabel)]
            accuracy = np.sum([int(x == y) for (
                x, y) in results]) / float(testData.shape[0]) * 100.0
            test_accuracy.append(accuracy)
            # cost = -1.0 * np.sum (np.log(yhat) * testLabel) +　lam * np.linalg.norm(weights)
            cost = -1.0 * np.sum(
                np.log(yhat) * testLabel) + lam * np.linalg.norm(weights)
            cost = 1.0 * cost / testData.shape[0]
            test_cost.append(cost)
    plt.figure(111)
    plt.plot(
        [i for i in range(epochs)],
        training_accuracy,
        label="Accuracy on the training data")
    plt.plot(
        [i for i in range(epochs)],
        test_accuracy,
        label="Accuracy on the test data")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
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
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        predictLabel = sortedClassCount[0][0] 
        return  predictLabel  #返回票数最多的标签
def knnModelTrain():
    tr_d, va_d, te_d = loadData()
    trainData, trainLabel = tr_d
    testData, testLabel = te_d
    accuracy = 0.0 
    kList = [1, 3, 10, 20]  # 
    for k in kList:
        knn = knnModel(k)
        for data, label in zip(testData, testLabel):
            result = knn.knnClassifier(trainData, trainLabel, data)
            if result == label :
                accuracy += 1.0
        accuracy = accuracy / testData.shape[0] *100.0
        print("The k of knn model is {} and accuracy is {}".format(k, accuracy))
              

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