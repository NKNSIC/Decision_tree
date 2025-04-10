import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

"""
数据采集
"""
iris = load_iris()      #加载数据集
data = iris.data        #存储特征数据
target = iris.target        #数据标签
target_names = iris.target_names        #数据标签名（鸢尾花名）
feature_names = iris.feature_names      #特征名

df = pd.DataFrame(data, columns = feature_names)        #保存为DataFrame形式
df['target'] = target       #创建标签列
df['target_names'] = [target_names[i] for i in target]      #创建标签名列，推导
#df.to_csv("KNN_Eg/Iris.csv", index = False)

"""
数据可视化
"""
fig = plt.figure(figsize = (16, 5))     #定义画板尺寸

ax1 = fig.add_subplot(1, 2, 1)      #添加子图可视化花萼长宽
ax1.set_title('Sepal Data', fontsize = 18)      #设置标题
for i in range(3):      #画散点图
    ax1.scatter(data[:, 0][target == i], data[:, 1][target == i], label = target_names[i])
    #[:,0]表示
ax1.set_xlabel(feature_names[0], fontsize = 15)
ax1.set_ylabel(feature_names[1], fontsize = 15)
ax1.legend(fontsize = 14)

ax2 = fig.add_subplot(1, 2, 2)      #添加子图可视化花瓣长宽
ax2.set_title('Petal Data', fontsize = 18)      #设置标题
for i in range(3):      #画散点图
    ax2.scatter(data[:, 2][target == i], data[:, 3][target == i], label = target_names[i])
ax2.set_xlabel(feature_names[2], fontsize = 15)
ax2.set_ylabel(feature_names[3], fontsize = 15)
ax2.legend(fontsize = 14)
plt.show()

"""
数据集划分
"""
X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size = 0.25, random_state = 0)
#（特征，标签，训练集划分占比，随机数）
"""
构建模型并评价
"""
KNN = KNeighborsClassifier(n_neighbors = 3)     #选择k近邻
KNN.fit(X_train, Y_train)       #训练模型
score = KNN.score(X_test, Y_test)       #数据精确度
print("Test score: %0.2f" %score)

"""
使用模型预测
"""
Y_pred = KNN.predict(X_test[: 10])      #数据预测
print("y_pred:", Y_pred)
print("y_true:", Y_test[: 10])

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = KNN.predict(X_new)
print("Prediction names:", target_names[prediction])