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
### 混淆矩阵基本评估
![image](https://user-images.githubusercontent.com/41554601/180994030-905c4795-0928-442a-a789-88f098a90e07.png)
![image](https://user-images.githubusercontent.com/41554601/180994499-156daa90-6448-4119-a650-17cbaaee62f7.png)
```
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_imbalance, random_state=0)
svm=SVC(kernel='linear',C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
### 混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(svm.predict(X_test),y_test)
### 评估accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(svm.predict(X_test),y_test))
print(precision_score(svm.predict(X_test),y_test))
print(recall_score(svm.predict(X_test),y_test))
print(f1_score(svm.predict(X_test),y_test))
##### 高阶画图
precm=confusion_matrix(y_test,svm.predict(X_test))
import seaborn as sns
precm1=pd.DataFrame(precm)
sns.heatmap(precm1,annot=True)
### 综合评估
from sklearn.metrics import classification_report
print(classification_report(svm.predict(X_test),y_test))
```
### 连续因变量的dummy比较，用r2
![image](https://user-images.githubusercontent.com/41554601/181385182-92012fee-0a53-42d2-9ec1-73a98bde1560.png)
```
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
ln1=LinearRegression().fit(X_train, y_train)
ln2=DummyRegressor(strategy='mean').fit(X_train, y_train)
from sklearn import metrics
metrics.r2_score(y_train,ln1.predict(X_train))
metrics.r2_score(y_train,ln2.predict(X_train))
```
