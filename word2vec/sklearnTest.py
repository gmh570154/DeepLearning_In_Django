#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
path = os.getcwd()
#从网络获取数据
column_names=['Sample code number', 'Clump Thickness', 'Uniformity of cell size', 
			  'Uniformity of cell shepe','Marginal Adhesion',
			  'Single Epithelial cell size', 'Bare Nuclei', 'Bland Chromatin',
              'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv('E://PycharmProjects//DeepLearning_In_Django//word2vec//breast-scancer.data',
                   names=column_names)
# E:\PycharmProjects\DeepLearning_In_Django\word2vec
#
# E:\PycharmProjects\DeepLearning_In_Django\word2vec\breast-scancer.data
#数据中有部分缺失，标记为？号，将这部分数据去除
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

#输出data的数据量和维度
print(data.shape)

#分割数据，选择25%为测试集
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]], 
                                                    test_size=0.25, random_state=33)
#查看训练&测试样本的数量和类别分布
print(y_train.value_counts())
print(y_test.value_counts())

#将数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#重命名LR和随机梯度下降函数
lr = LogisticRegression()
sgdc = SGDClassifier()

#用LR拟合并预测数据
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
#用随机梯度下降拟合并预测数据
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)

#输出LR的准确率和混淆矩阵
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

print('-------------------------------------------------------')

#输出随机梯度下降的准确率和混淆矩阵
print('Accravy of SGD Classifier:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))