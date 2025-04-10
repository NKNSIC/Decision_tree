from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt

digits = load_digits()
data = digits.data
target = digits.target

fig = plt.figure(figsize = (8, 8))
for i in range(20):
    ax1 = fig.add_subplot(4, 5, i+1)
    ax1.matshow(digits.data[i].reshape(8, 8))
plt.show()

#建立支持向量机模型，调优
#准确率，前20行

'''
数据矩阵转置
'''
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.25, random_state = 0)
svm_1 = SVC()
svm_1.fit(x_train, y_train)
svm_2 = SVC(C = 0.1, gamma = 'scale')
svm_2.fit(x_train, y_train)
svm_3 = SVC(kernel = 'linear', gamma = 'scale')
svm_3.fit(x_train, y_train)

print("mod_1准确率：", end='')
print(svm_1.score(x_test, y_test))
print("mod_2准确率：",end='')
print(svm_2.score(x_test, y_test))
print("mod_3准确率：",end='')
print(svm_3.score(x_test, y_test))


print("mod_1预测值：")
print(svm_1.predict(x_test)[:20])
print("mod_2预测值：")
print(svm_2.predict(x_test)[:20])
print("mod_3预测值：")
print(svm_3.predict(x_test)[:20])
print("真实值：")
print(y_test[:20])