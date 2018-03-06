
"""
手写体数字识别问题算法"""
import gzip
import random
import _pickle as cPickle
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import operator
def loadDataSet(filename):
    """
    Desc: svm测试数据集加载函数

    """
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

def runTime(func):
    def wrapper(*args, **kwargs):
        import time
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print(("{} run time: {}  s").format(func.__name__, t2 - t1))
    return wrapper
def vectorized_result(j):
    """
    Desc: 对标签进行OneHot编码
    Params: 
        j: 输入样本标签
    Return:
        e: 编码后向量类别为k则向量下标为k的位置是1
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
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
        yhat = yhat / np.sum(yhat, axis = 0) 
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
        l1 = np.multiply(y, theta)
        thetaMax = np.max(theta, axis = 0)
        theta -= thetaMax
        l2 = np.log(np.sum(np.exp(theta), axis = 0))
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
                self.weights = self.weights - self.eta * grad  
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
                #self.eta *= 0.8
                #lr*=0.8
            print(("Epoch: {} completed").format(iter))
        w = self.weights
        return w, Loss, ev_Loss, train_acc, ev_acc
def test(weights, te_d):
    testData, testLabel = te_d
    n = testData.shape[0]
    testData = np.insert(testData, 0, values = np.ones(n), axis = 1)
    x_test = np.transpose(np.matrix(testData))
    y_test = np.array(testLabel)
    theta = np.dot(weights.T, x_test)  # 10*784  784*m
    theta -= np.max(theta, axis = 0)  # 求每一列的最大值
    mu = np.exp(theta)
    mu = mu / np.sum(mu, axis = 0)     # 10*m
    results = [(np.argmax(yhat), label) for yhat, label in zip(mu.T, testLabel)]
    accuracy = np.sum([int(x==y) for (x,y) in results])
    accuracy = accuracy / float(n) * 100.0 
    print(("Accuracy for test set is {}").format(accuracy))
    return accuracy 
@runTime
def logisticMainFun(eta, lam, mini_batch_size, epochs):
    tr_d, va_d, te_d = loadData()
   # eta = 0.5
   # lam = 0.0
   # mini_batch_size = 10000
   # epochs = 200
    weights = np.random.randn((785, 10))
    model = logisticRegModel(weights, eta, lam, mini_batch_size, epochs)
    w, Loss, ev_Loss, train_acc, ev_acc = model.SGDTrain(tr_d, te_d) 
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
    return w
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
    trainData = trainData[0:10000] #截取10000个样本
    trainLabel = trainLabel[0:10000] 
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
    def __init__(self, alphas, b):
        """
        Desc: SVM模型初始化函数
        """
        self.alphas = alphas
        self.b = b
    
    def SMO(self, C, tol, max_passes, trainData, trainLabels):
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
       # m = len(trainingData)
       # trainData = [ x for x, y in trainingData]
       # trainLabels=[ y for x, y in trainingData]
        trainData = np.mat(trainData)
        trainLabels = np.mat(trainLabels).transpose()
        m, n = trainData.shape
        passes = 0
        while (passes < max_passes):
            num_changed_alphas = 0
            for i in range(m):
                # calculate Ei
                xi = trainData[i,:]  # 第i个输入样本
                yi = float(trainLabels[i]) # 第i个输入样本的标签
                fxi = np.sum([float(alpha) * float(y) * x * xi.T for x, y, alpha in zip(trainData, trainLabels, self.alphas)])
                Ei = fxi - yi
                if ((yi * Ei < -tol) and (self.alphas[i] < C)) or ((yi * Ei > tol) and (self.alphas[i] > 0)):
                    # select j random such that j \= i
                    j = i
                    while (j == i):
                        j = int(random.uniform(0, m))
                    # calculate Ej
                    xj = trainData[j,:]
                    yj = float(trainLabels[j])
                    fxj = np.sum([float(alpha) * float(y) * x * xj.T for x, y, alpha in zip(trainData, trainLabels, self.alphas)])
                    Ej = fxj - yj
                    # save old alpha 
                    alphaiOld = float(self.alphas[i])
                    alphajOld = float(self.alphas[j])
                    # 
                    if (yi != yj):
                        L = max(0, float(self.alphas[j]) - float(self.alphas[i]))
                        H = min(C, C + float(self.alphas[j]) - float(self.alphas[i]))
                    else:
                        L = max(0, float(self.alphas[i]) + float(self.alphas[j]) - C)
                        H = min(C, float(self.alphas[i]) + float(self.alphas[j]))
                    if (L == H): print("L==H"); continue  # if L==H continue to next i
                    # calculate eta 
                    eta = 2.0 * xi * xj.T -xi * xi.T - xj * xj.T 
                    if (eta >= 0): 
                        print("eta >= 0")
                        continue
                    # Compute and clip new value for alphaj
                    self.alphas[j] -= float(yj * (Ei - Ej) / eta)
                    if (self.alphas[j] > H):
                        self.alphas[j] = H
                    if (self.alphas[j] < L):
                        self.alphas[j] = L
                    if (abs(self.alphas[j] - alphajOld) < 1e-5):
                        print("j not moving enough")
                        continue
                    # Determine value for alphai
                    self.alphas[i] += yi * yj * (alphajOld - self.alphas[j])
                    # Compute b1 and b2 and b
                    b1 = self.b - Ei - yi * (self.alphas[i] - alphaiOld) * \
                    (xi * xi.T) - yj * (self.alphas[j] - alphajOld) * (xi * xj.T)
                    b2 = self.b - Ej - yi * (self.alphas[i] - alphaiOld) * \
                    (xi * xj.T) - yj * (self.alphas[j] - alphajOld) * (xj * xj.T)
                    if (self.alphas[i] > 0 ) and (self.alphas[i] < C):
                        self.b = b1
                    elif (self.alphas[j] > 0) and (self.alphas[j] < C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2)/2.0
                    num_changed_alphas += 1
                    print(("iter: {0} i: {1}, pair changed {2}").format(passes, i, num_changed_alphas))
            if (num_changed_alphas == 0):
                passes += 1
            else:
                passes = 0
            print(("iteration number: {}").format(passes))
        return self.alphas, self.b
#@runTime
def svmTrain(C, tol, max_passes, tr_d):
    """
    Desc: 利用one vs the rest 策略训练 N个分类器 保存N个二分类器的模型参数
    优点相比one vs one策略训练的分类器N*(N-1)/2要少,缺点1、对于两种极端情况即每个分类器
    结果都输出是或者每个分类器结果都输出不是只能人为处理 2、正负样本不均衡问题 对于每个
    要训练分类的分类器负样本是其他类别样本总和其数目大于正样本

    """
    trainData, trainLabels = tr_d
    total_sample = trainData.shape[0]
    classes = list(set(trainLabels))
    cl_n= len(classes)
    trainSet = {}
    params = {}
    for cl in classes:
        index = [ind for ind, label in enumerate(trainLabels) if label == cl]
        posSample = trainData[index] # 当前分类的训练样本
        # 从剩余样本中随机选择和正样本数量差不多的负样本
        n = len(index)
        indNeg = set(list(np.arange(0, total_sample))) - set(index) # 除去正样本之后的负样本集合
        indNeg = random.sample(indNeg, n)
        negSample = trainData[indNeg]
        Samples = np.append(posSample, negSample, axis = 0) #按行拼接
        Labels = np.append(np.ones((n, 1)), -1.0 * np.ones((n, 1)), axis = 0) #按行进行拼接
        # 训练之前随机打乱训练样本
        randInd = np.arange(Samples.shape[0])
        np.random.shuffle(randInd)
        Samples = Samples[randInd]
        Labels = Labels[randInd]
        trainSet[cl] ={zip(Samples, Labels)} #保留每次训练的样本集合用于测试
        alphas = np.mat(np.zeros((Samples.shape[0], 1)))
        b = 0
        svm = SVM(alphas, b)
        svm.SMO(C, tol, max_passes, Samples, Labels)
        params[cl] = svm  #保存每个分类器参数  
    return params, trainSet  # 返回模型参数和训练样本集合
def svmTest(params, testData, testLabels, trainSet):
    """
    Desc: 测试属于哪一类输出准确率分类超平面的方程为：y=wT+x=sum_i^m alphai*yi*xi 
    Params:

    """
    accuracy = 0.0 
    votes = {}    #每个分类器的投票结果 
    for x, y in zip(testData, testLabels):
        Distance = []
        for c in params.keys():
            alphas, b = params[c].alphas, params[c].b  # 取出模型参数
            alphai = alphas[alphas > 0]       #支持向量  m * 1
            ind = [ind for ind, item in enumerate(alphas) if alphas > 0]
            samples, labels = trainSet[c]  
            supportSam, supportLab = samples[ind], labels[ind]
            w = np.sum(np.multiply(alphai, supportLab) * supportSam, axis = 0) # 对某一列求和 w
            result = np.dot(x, w) + b
            dis = abs(result) / float(np.linalg.norm(w))  # 保存测试样本到分类超平面的距离
            Distance.append(dis)
            if result >= 0:
                votes[c] = votes.get(c, 0) + 1
        # 进行决策
        if votes == {}:   #不属于任何类别  随机判一个类别还是
            label = random.choice(list(params.keys()))
        else:
            sortedVotes = sorted(votes.items(), key = operator.itemgetter(1), reverse=True)
            if sortedVotes[0][1] != sortedVotes[-1][1]:
                label = sortedVotes[0][0]
            else:       #属于所有类别 判断给距离最远的超平面
                label = list(params.keys())[Distance.index(min(Distance))] 
        if int(label) == int(y):
            accuracy += 1.0
    accuracy = accuracy / float(len(testData.shape[0]))
    print(("Accuracy is {} ").format(accuracy))
    return accuracy
def svmMainFun(C, tol, max_passe):
    tr_d, va_d, te_d = loadData()
    params, trainSet = svmTrain(C, tol, max_passe, tr_d)
    testData, testLabels = te_d
    accuracy = svmTest(params, testData, testLabels, trainSet)
        

