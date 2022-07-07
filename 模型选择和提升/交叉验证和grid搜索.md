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

# 交叉验证
## cv代表把数据集分成10份，验证每一成分中的准确率
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()

# grid search (相当于验证不同的参数)
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


