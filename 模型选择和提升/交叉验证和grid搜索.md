## 载入svm数据库
https://github.com/171909771/DATA-scientist/files/8984596/Social_Network_Ads.csv
```
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
classifier=SVC()
```
### 交叉验证
```
## cv代表把数据集分成10份，验证每一成分中的准确率
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
#### 可以分开训练集和测试集评估
train,test=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
```
#### 验证曲线
###### 查看在交叉验证中不同参数的作用
```
from sklearn.model_selection import validation_curve
range1=np.arange(1,11,1)
a1,a2=validation_curve(SVC(kernel='poly',random_state=0),X_train,y_train,param_name='degree',param_range=range1,cv=5)
### 画图
y1=[x.max() for x in a1]
y2=[x.min() for x in a1]
ymean=[x.mean() for x in a1]
y21=[x.max() for x in a2]
y22=[x.min() for x in a2]
y2mean=[x.mean() for x in a2]
plt.fill_between(range1, y1, y2, alpha=.5, linewidth=0)
plt.plot(range1, ymean, linewidth=2)
plt.fill_between(range1, y21, y22, alpha=.5, linewidth=0)
plt.plot(range1, y2mean, linewidth=2)
```
![image](https://user-images.githubusercontent.com/41554601/180798696-b91e5314-eac9-4a16-bc9d-676b6e1b739e.png)

### grid search (相当于验证不同的参数)
```
from sklearn.model_selection import GridSearchCV
## C 代表是否过度拟合（是SVM的参数），最小可能过度拟合，最大可能拟合不足
parameters=[{'C':[1,2,3],'kernel':['rbf']},
           {'C':[1,2,3,4,5,6,7],'kernel':['rbf'],'gamma':[0.5,0.6,0.7,0.8]}]
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_scores_=grid_search.fit(X_train,y_train)
## 下面的几个参数需要了解
grid_scores_.cv_results_ ## 查看不同参数下的数值
grid_scores_.best_score_ ## 查看最好的准确率是多少
grid_scores_.best_params_ ## 查看最好的参数怎么设置
```


