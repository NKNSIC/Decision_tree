import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = pd.read_csv('Linear_R/boston_data.csv')
data = boston.iloc[:,:13]
target = boston.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.25, random_state = 0)

L_R = LinearRegression()
L_R.fit(x_train, y_train)

print(L_R.coef_)        #权重值
print(L_R.intercept_)       #偏置

prediction = L_R.predict(x_test)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize = (20, 5))
x = np.arange(len(y_test))
plt.plot(x, prediction, label = '预测值', ls = '--')
plt.plot(x, y_test, label = '真实值')
plt.legend()
plt.show()

features = ['RM', 'LSTAT', 'PTRATIO']  # 选择几个关键特征
for feature in features:
    sns.lmplot(x = feature, y='target', data = boston, aspect=1.5, ci=None)
    plt.title(f'Relationship between {feature} and Price')
    plt.show()