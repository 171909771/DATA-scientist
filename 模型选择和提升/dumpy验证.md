### 假验证，按照训练集中的结果分层进行构建
```
from sklearn.datasets import load_digits
dataset=load_digits()
x,y=dataset.data, dataset.target
y_imbalance=y.copy()
y_imbalance[y_imbalance!=1]=0
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor  ## 对应的回归dummy
## most_frequent: 预测值是出现频率最高的类别
## stratified : 根据训练集中的频率分布给出随机预测
## uniform: 使用等可能概率给出随机预测
## constant: 根据用户的要求, 给出常数预测.
dummy_m=DummyClassifier(strategy='constant',constant =0).fit(X_train, y_train)
dummy_m.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(dummy_m.predict(X_test),y_test)
```
