import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("Decision_tree\\abalone.csv")
data.columns=["sex","length","diameter","height","whole weight","shucked weight","viscera weight","shell weight","rings"]
ablone = data.replace({"M":0, "F":1, "I":2})

data = ablone.iloc[:, :8]
target = ablone.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state = 0)

DT = DecisionTreeRegressor()
DT.fit(x_train, y_train)

prediction = DT.predict(x_test)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize = (20, 5))
plt.title("决策树")
x = np.arange(len(y_test[:100]))
plt.plot(x, prediction[:100], label = '预测值', ls = '--')
plt.plot(x, y_test[:100], label = '真实值')
plt.legend()
plt.show()
print(DT.score(x_test, y_test))

model_F = RandomForestRegressor(500)
model_F.fit(x_train, y_train)
prediction_R = model_F.predict(x_test)
plt.figure(figsize = (20, 5))
plt.title("随机森林")
x = np.arange(len(y_test[:100]))
plt.plot(x, prediction_R[:100], label = '预测值', ls = '--')
plt.plot(x, y_test[:100], label = '真实值')
plt.legend()
plt.show()

print(model_F.score(x_test, y_test))