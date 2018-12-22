# -- encoding:utf8 --
"""
Learning Regression base on sklearn
Created by CZL on 2012/12/22
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1.加载数据
path = 'datas/household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')
# df.info()

# 2.数据清洗
df.replace('?', np.nan, inplace=True)
df = df.dropna(axis=0, how='any')
# df.info()

# 3.获取原始特征属性矩阵X和目标属性Y
X = df.iloc[:, 2:4].astype(np.float32)
Y = df.iloc[:, 5].astype(np.float32)
# X.info()
# print(Y.shape)

# 4.数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=26)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5.特征工程
ss = StandardScaler()
"""
方式一：
# b. 模型训练(从训练数据中找转换函数)
StandardScaler.fit(x_train, y_train)
# c. 使用训练好的模型对训练数据做一个转换操作
x_train = StandardScaler.transform(x_train)
"""
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

# 6.构建模型
"""
fit_intercept=True, 模型是否训练截距项，默认为True，表示训练，False表示不训练
normalize=False, 在模型训练之前是否对数据做一个归一化的处理，默认表示不进行，该参数一般不改
copy_X=True, 对于训练数据是否copy一份再训练，默认是
n_jobs=1 指定使用多少个线程来训练模型，默认为1
NOTE: 除了fit_intercept之外，其它参数基本不修改
"""
lr = LinearRegression(fit_intercept=True)

# 7.模型训练
lr.fit(x_train, y_train)

# 8.模型评估
print(lr.coef_)
print('截距：{}'.format(lr.intercept_))
print('训练数据R²：{}'.format(lr.score(x_train, y_train)))
print('测试数据R²：{}'.format(lr.score(x_test, y_test)))

# 9.模型保存/模型持久化
joblib.dump(ss, 'model/Sscaler.pkl')
joblib.dump(lr, 'model/lr.pkl')

# 10.画图查看效果
predict_y = lr.predict(x_test)
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.title('LinearRegression_sklearn')
plt.plot(t, y_test, 'r-', label=u'真实值')
plt.plot(t, predict_y, 'b-', label=u'预测值')
plt.legend(loc='lower right')
plt.show()
