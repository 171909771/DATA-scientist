# 载入数据
https://github.com/171909771/DATA-scientist/files/8984238/Social_Network_Ads.csv
## 就是让logictis回归中的非线性数据变成线性可分数据

```
# step1 数据前处理
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

# step2 kernel PCA
from sklearn.decomposition import KernelPCA
##### n_components先设置为None（就等于自变量数目），PCA 查看每个成分的P值 #####
kpca=KernelPCA(n_components=2,kernel='rbf')
X_train=kpca.fit_transform(X_train)
X_test=kpca.transform(X_test)

# step3 logistic 回归
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# step4 画图
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
               color=ListedColormap(('orange','blue'))(i),label=i,s=5)
```
![image](https://user-images.githubusercontent.com/41554601/177652499-2e8c81b0-6c32-4924-b2b9-3a5e878e5b87.png)
