### 载入数据
[Social_Network_Ads.csv](https://github.com/171909771/DATA-scientist/files/8984238/Social_Network_Ads.csv)

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
# logistic 回归
from sklearn.linear_model import LogisticRegression
## 下面C参数越小代表更强的正则化
classifier=LogisticRegression(C=1，random_state=0)
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
![image](https://user-images.githubusercontent.com/41554601/175766081-ff7d6892-fc6f-44f7-84bc-b996e6df7c2c.png)
