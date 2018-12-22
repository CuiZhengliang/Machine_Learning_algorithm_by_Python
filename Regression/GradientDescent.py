# -- encoding:utf-8/-
"""
Created by CZL on 2018/12/21
DIY Gradient Descent
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


class GD:
    """
    梯度下降
    """

    def __init__(self, inter_input=True, alpha=1e-6, tol=1e-8, max_iter=100000):
        self.__theta = None
        self.__intercept = None
        self.__iteration = None
        self.__loss = None
        self.__score = None
        self.__inter_input = inter_input
        self.__alpha = alpha
        self.__tol = tol
        self.__max_iter = max_iter

    def fit(self, x, y):
        x = np.mat(x)
        y = np.mat(y)
        a = x.shape[0]
        if self.__inter_input:
            i_ = np.array([[10] for n in range(a)])
            x = np.hstack((i_, x))
        b = x.shape[1]
        c = y.shape[0]
        loss_old = 0
        if a == c:
            theta = np.array([[1.0] for i in range(b)])
            for j in range(self.__max_iter):
                theta = theta + x.T.dot(y - x.dot(theta)) * self.__alpha
                loss_ = np.power(y - x.dot(theta), 2).mean()
                self.__loss = loss_
                self.__iteration = j
                if self.__inter_input:
                    self.__intercept = theta[0].mean()
                    self.__theta = theta[1:]
                else:
                    self.__theta = theta
                if loss_ - loss_old <= 1e-20 and loss_ - loss_old >= -1e-20:
                    print(j)
                    break
                loss_old = loss_
        else:
            print('输入参数有误!')

    def predict(self, x_test_):
        x_test__ = np.mat(x_test_)
        if self.__inter_input:
            y_hat = x_test__.dot(self.__theta) + self.__intercept
        else:
            y_hat = x_test__.dot(self.__theta)
        return y_hat

    def score(self, y_hat_, y_test_):
        y_test__ = np.mat(y_test_)
        # mse = np.sum(np.power((y_test__ - y_hat_), 2)) / y_test__.shape[0]
        # self.__score = 1 - mse / np.var(y_test__)
        # self.__score = mse
        SStot = np.sum(np.power(y_test__ - np.mean(y_test__), 2))
        SSres = np.sum(np.power(y_test__ - y_hat_, 2))
        self.__score = 1 - SSres / SStot
        # print(self.__theta)
        return self.__score

    def get_loss(self):
        return self.__loss


# 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = 'datas/boston_housing.data'

# 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再按每行数据进行处理
fd = pd.read_csv(path, header=None)


def not_empty(s):
    return s != ''


data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):  # enumerate生成一列索 引i,d为其元素
    # 根据函数结果是否为真，来过滤list中的项。
    d = map(float, filter(not_empty, d[0].split(' ')))  # filter一个函数，一个list
    data[i] = list(d)
# 分割数据
x, y = np.split(data, (13,), axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
# print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
# print("测试数据X的格式:{}".format(x_test.shape))
# print("训练数据Y的格式:{}, 以及类型:{}".format(y_train.shape, type(y_train)))

gd = GD(inter_input=False, alpha=1.53e-8, max_iter=1000000)
gd.fit(x_train, y_train)
y_p = gd.predict(x_test)
s = gd.score(y_p, y_test)
ss = gd.get_loss()
print(s)
print(ss)
