import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target
target_names = cancer.target_names
feature_names = cancer.feature_names

df = pd.DataFrame(data, columns = feature_names)
df['target'] = target
df['target_names'] = [target_names[i] for i in target]
#df.to_csv('Log_R/cancer.csv', index = False)

x_train, x_test, y_train, y_test = train_test_split(data, target, random_state = 0)
Log = LogisticRegression()
Log.fit(x_train, y_train)
print("score：", Log.score(x_test, y_test))
print(Log.predict(x_test[:3]))
print(y_test[:3])
predict_pro = Log.predict_proba(x_test)
print("Prob：[0_prob, 1_prob]\n", predict_pro[:3])
predict = Log.predict(x_test)

print("准确率：", accuracy_score(y_test, predict))
print("精确率：", precision_score(y_test, predict))
print("召回率：", recall_score(y_test, predict))
print("F1值：", f1_score(y_test, predict))