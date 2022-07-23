### 载入数据
[fruit_data_with_colors.txt](https://github.com/171909771/DATA-scientist/files/9173566/fruit_data_with_colors.txt)

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
fruit=pd.read_table("fruit_data_with_colors.txt")
ref=dict(zip(fruit.fruit_label,fruit.fruit_name))
X,y=fruit[["mass","width","height","color_score"]],fruit["fruit_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 查看自变量之间的关系
from matplotlib import cm
cmap =cm.get_cmap("gnuplot")  # https://blog.csdn.net/weixin_39534395/article/details/111528965
scatter=pd.plotting.scatter_matrix(X_train,c=y_train,marker='o',s=40,cmap=cmap)
```
![image](https://user-images.githubusercontent.com/41554601/180601507-0eb12c58-36fa-4fea-8322-249fc5f4bfa2.png)
```
# 设定K值，看最近的几个数据
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
# 混淆矩阵
from sklearn.metrics import confusion_matrix
cmat=confusion_matrix(y_test.values,y_pred)
```
