### 数据预处理
```
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

```
### MDS方法
```
from sklearn.manifold import MDS
mds=MDS(n_components=2)
X_train=mds.fit_transform(X_train)
# step4 画图
from matplotlib.colors import ListedColormap
## 改变train为test就可以看测试数据的分类
X_set,y_set=X_train,y_train
## meshgrid函数让2组数据中的值互相匹配。给表格周边留白，此处留白数字1
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
## 画散点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               color=ListedColormap(('orange','blue','black'))(i),label=i)
```
###TSNE
```
from sklearn.manifold import TSNE

tsne=TSNE(n_components=2)
X_train=tsne.fit_transform(X_train)
# step4 画图
from matplotlib.colors import ListedColormap
## 改变train为test就可以看测试数据的分类
X_set,y_set=X_train,y_train
## meshgrid函数让2组数据中的值互相匹配。给表格周边留白，此处留白数字1
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

## 画散点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               color=ListedColormap(('orange','blue','black'))(i),label=i)
```              
