# -- encoding:utf-8 --
"""
Created by CZL on 2018/12/25
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

mpl.rcParams['font.sans-serif'] = [u'simHei']

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=12)
# print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
# print("测试数据X的格式:{}".format(x_test.shape))
# print("训练数据Y的格式:{}, 以及类型:{}".format(y_train.shape, type(y_train)))

# Pipeline组合Models
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', Ridge(alpha=1))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', Lasso(alpha=0.5))
    ])
]

parameters = {
    'poly__degree': [3, 2, 1],
    'poly__interaction_only': [True, False],
    'poly__include_bias': [True, False],
    'linear__fit_intercept': [True, False],
}

titles = ['LinearR', 'Ridge', 'Lasso']
colors = ['y-', 'g-', 'b-']
plt.figure(facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
for t in range(len(models)):
    model = GridSearchCV(models[t], param_grid=parameters, cv=5, n_jobs=1)
    model.fit(x_train, y_train)
    print("%s算法最优参数：%s" % (titles[t], model.best_params_))
    print("%s算法最优R²值：%.5f" % (titles[t], model.best_score_))
    y_predict = model.predict(x_test)
    plt.plot(ln_x_test, y_predict, colors[t], lw=t+3, label=u'%s算法估计值，$R^2$=%.5f' % (titles[t], model.best_score_))
plt.legend(loc='upper left')
plt.grid(True)
plt.title(u'波士顿房屋价格预测')
plt.show()
