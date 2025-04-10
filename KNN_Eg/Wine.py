import torch as to
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #解决中文乱码问题

"""
数据采集
"""
wine = load_wine()
Data = wine.data
target = wine.target
target_names = wine.target_names
feature_names = wine.feature_names

df = pd.DataFrame(Data, columns = feature_names)
df['target'] = target
df['target_names'] = [target_names[i] for i in target]
#df.to_csv("KNN_Eg/wine.csv", index = False)

"""
数据可视化(PCA降维，用方差筛出影响小的特征值)
"""

pca = PCA(n_components = 2)
pca = pca.fit(Data)
data_pca_old = pca.transform(Data)

fig = plt.figure(figsize = (16, 5)) 
ax1 = fig.add_subplot(1, 2, 1)   
ax1.set_title('PCA标准化前', fontsize = 18)     
for i in range(3):   
    ax1.scatter(data_pca_old[:, 0][target == i], data_pca_old[:, 1][target == i], label = target_names[i])
 
ax1.set_xlabel("主成分1", fontsize = 15)
ax1.set_ylabel("主成分2", fontsize = 15)
ax1.legend(fontsize = 14)

# 数据标准化（重点，能够大幅度提高模型预测精确度）
scaler = StandardScaler()
data = scaler.fit_transform(Data)

pca = PCA(n_components = 2)
pca = pca.fit(data)
data_pca = pca.transform(data)

ax2 = fig.add_subplot(1, 2, 2)     
ax2.set_title('PCA标准化后', fontsize = 18)    
for i in range(3):      
    ax2.scatter(data_pca[:, 0][target == i], data_pca[:, 1][target == i], label = target_names[i])
ax2.set_xlabel("主成分1", fontsize = 15)
ax2.set_ylabel("主成分2", fontsize = 15)
ax2.legend(fontsize = 14)
plt.show()

"""
数据集划分
"""
X_train, X_test, Y_train, Y_test = train_test_split(data_pca, target, train_size = 0.25,random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.25,random_state = 0)

"""
模型构建和评测（余弦相似度KNN）
"""
CNN = KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', metric = 'cosine')
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(CNN, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Cos Test score: %0.2f" %accuracy)

"""
模型构建和评测（闵可夫斯基KNN）
"""
MNN = KNeighborsClassifier(algorithm='ball_tree', metric='minkowski', p=2)
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(MNN, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Min Test score: %0.2f" %accuracy)

"""
模型构建和评测(PCA降维后的最近邻聚类)
"""
KNN = KNeighborsClassifier(n_neighbors = 1)
KNN.fit(X_train, Y_train)
score = KNN.score(X_test, Y_test)
print("PCA Test score: %0.2f" %score)

"""
模型构建和评测(线性回归)
"""
L = LinearRegression()
L.fit(x_train, y_train)
score = L.score(x_test, y_test)
print("Linear Test score: %0.2f" %score)
# 预测与实际拟合散点图
y_pred = L.predict(x_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

"""
模型构建和评测(神经网络)
"""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = 13
hidden_dim = 64
output_dim = 3
NN = MLP(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NN.parameters(), lr = 0.01)


x_train = to.tensor(x_train, dtype=to.float32)
x_test = to.tensor(x_test, dtype=to.float32)
y_train = to.tensor(y_train, dtype=to.long)
y_test = to.tensor(y_test, dtype=to.long)

# 训练模型
epochs = 100
for epoch in range(epochs):
    NN.train()
    optimizer.zero_grad()
    outputs = NN(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

NN.eval()
with to.no_grad():
    predictions = NN(x_test)
    _, predicted_classes = to.max(predictions, 1)
    accuracy = (predicted_classes == y_test).sum().item() / y_test.size(0)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 将标签二值化
y_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# 假设模型输出是 logits
logits = predictions

# 应用 Softmax 转换为概率
probs = F.softmax(logits, dim=1)

# 获取模型的预测概率
y_score = probs  # 假设模型有 predict_proba 方法

# 计算每个类别的 ROC 曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()