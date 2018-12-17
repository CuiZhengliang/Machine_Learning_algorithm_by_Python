# -- encoding:utf-8 --
"""
Created by CZL on 2018/12/17
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

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
y = y.ravel()  # 转换格式 拉直操作
ly = len(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
# print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
# print("测试数据X的格式:{}".format(x_test.shape))
# print("训练数据Y的格式:{}, 以及类型:{}".format(y_train.shape, type(y_train)))


# 构建LinearRegression Model(根据数据情况需要转换为numpy矩阵格式)
def lir_fit(x_train_, y_train_):
    x_mat = np.mat(x_train_)
    y_mat = np.mat(y_train_).reshape(-1, 1)
    return (x_mat.T * x_mat).I * x_mat.T * y_mat


def predict_lir(x_):
    return np.mat(x_) * theta


theta = lir_fit(x_train, y_train)
p = predict_lir(x_test)

t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'y_test真实值')
plt.plot(t, p, 'b-', label=u'x_test预测值')
plt.title('DIY LinearRegression')
plt.legend(loc='upper right')
plt.show()
