# -- encoding:utf-8 --
"""
Created by CZL on 2018/12/18
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
# y = y.ravel()  # 转换格式 拉直操作
# ly = len(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# 构建构建LinearRegression Model base on Polynomial
def lir_pl_fit(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)

    xtx = x_mat.T * x_mat
    if np.linalg.det(xtx) == 0:
        print("Error!")
        return
    theta = xtx.I * (x_mat.T * y_mat)
    return theta


def lir_pl_predict(x_arr, theta):
    x_mat = np.mat(x_arr)
    return x_mat * theta


theta_ = lir_pl_fit(x_train, y_train)
y_p = lir_pl_predict(x_test, theta_)

# 画图
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', label=u'y_test真实值')
plt.plot(t, y_p, 'b-', label=u'x_test预测值')
plt.title('DIY LinearRegressionPolynomial')
plt.legend(loc='upper right')
plt.show()


def score_reg(test_y, predict_y):
    mse = np.sum(np.power((test_y.reshape(-1, 1) - predict_y), 2)) / len(test_y)
    r2 = 1 - mse / np.var(test_y)
    print("MSE:", mse)
    print("R2:", r2)
    print("theta:", theta_)


score_reg(y_test, y_p)
