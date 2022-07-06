## 载入数据
[Wine.csv](https://github.com/171909771/DATA-scientist/files/9055451/Wine.csv)
```
# step1 数据预处理
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
datasets=pd.read_csv('Wine.csv')
X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# step2 PCA
from sklearn.decomposition import PCA
##### n_components先设置为None（就等于自变量数目），PCA 查看每个成分的P值 #####
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
pca.explained_variance_ratio_
##### n_components先设置为None（就等于自变量数目），PCA 查看每个成分的P值 #####
## 根据上面的数值，设置下面的n_components
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

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
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green','black')))
## 画散点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               color=ListedColormap(('orange','blue','black'))(i),label=i)
```
![image](https://user-images.githubusercontent.com/41554601/177569654-80c3b32e-7a79-44d8-a0ac-0f941585d09c.png)
