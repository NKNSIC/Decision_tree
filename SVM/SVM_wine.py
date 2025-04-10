import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

wine = load_wine()
data = wine.data
target = wine.target
feature_names = wine.feature_names
target_names = wine.target_names
df = pd.DataFrame(data, columns = feature_names)
df['target'] = target
df['target_names'] = [target_names[i] for i in target]

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.25, random_state = 42)
svm_1 = SVC()
svm_1.fit(x_train, y_train)
svm_2 = SVC(C = 0.1, gamma = 'scale')
svm_2.fit(x_train, y_train)
svm_3 = SVC(kernel = 'linear', gamma = 'scale')
svm_3.fit(x_train, y_train)

print(svm_1.score(x_test, y_test))
print(svm_2.score(x_test, y_test))
print(svm_3.score(x_test, y_test))

print(svm_1.predict(x_test)[:10])
print(svm_2.predict(x_test)[:10])
print(svm_3.predict(x_test)[:10])
print(y_train[:10])
