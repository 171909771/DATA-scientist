# 载入数据
[Social_Network_Ads.csv](https://github.com/171909771/DATA-scientist/files/8984596/Social_Network_Ads.csv)

```
# 数据前处理
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
datasets=pd.read_csv('Social_Network_Ads.csv')
X=datasets.iloc[:,2:-1].values
Y=datasets.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
# 支持向量
from sklearn.svm import SVC
## 注意调参，主要用'poly', 'rbf', 'sigmoid'
### poly就是多项式回归，可以调degree
### rbf就是高悬函数
### sigmoid就是logistic回归
### rbf:C是正则化，越大效果越小,越大越锐利
### rbf:gamma越大越锐利（只有靠近的点被认为相似）
classifier=SVC(kernel='poly',degree=3.5, C=1, random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
# 画图
from matplotlib.colors import ListedColormap
## 改变train为test就可以看测试数据的分类
X_set,y_set=X_train,y_train
## meshgrid函数让2组数据中的值互相匹配。给表格周边留白，此处留白数字1
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
## 画背景和分界线，ravel把array转成长list
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
                                                X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
## 画散点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               color=ListedColormap(('orange','blue'))(i),label=i)
```
![image](https://user-images.githubusercontent.com/41554601/175771586-14e9902f-7963-4cda-bbb5-c35b1c25247a.png)

### Precision-Recall及ROC曲线
```
decision_scores=decision_scores=svm.decision_function(X_test)  ###SVM所独有
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precisions = [] #记录精准率
recalls = [] #记录召回率
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1) #分割不同的threshold
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))
import matplotlib.pyplot as plt
plt.plot(thresholds, precisions) #绘制thresholds—精确率曲线
plt.plot(thresholds, recalls) #绘制thresholds—召回率曲线
plt.show()
plt.plot(precisions, recalls) #绘制Precision-Recall曲线
```
![image](https://user-images.githubusercontent.com/41554601/181265375-68e8277c-2fab-4dc5-be15-e0400ead290e.png)
![image](https://user-images.githubusercontent.com/41554601/181265390-09e7c236-c642-4143-81c1-94f6003eef95.png)

