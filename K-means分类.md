[Mall_Customers.csv](https://github.com/171909771/DATA-scientist/files/8986281/Mall_Customers.csv)
```
# 数据前处理
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
datasets=pd.read_csv('Mall_Customers.csv')
X=datasets.iloc[:,3:5].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    ## n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10次质心，实现算法，然后返回最好的结果。
    ## max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
```
![image](https://user-images.githubusercontent.com/41554601/175806926-33fad7ef-ea58-4d24-aae5-29d95b64148f.png)

```
## 根据上面elbow method寻找到cluster个数后，替代n_clusters的值
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_keams=kmeans.fit_predict(X)
plt.scatter(X[y_keams==0,0],X[y_keams==0,1],c="red",label="cluster0")
plt.scatter(X[y_keams==1,0],X[y_keams==1,1],c="blue",label="cluster1")
plt.scatter(X[y_keams==2,0],X[y_keams==2,1],c="black",label="cluster2")
plt.scatter(X[y_keams==3,0],X[y_keams==3,1],c="cyan",label="cluster3")
plt.scatter(X[y_keams==4,0],X[y_keams==4,1],c="magenta",label="cluster4")
### 画中心点
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="dimgrey",s=80,label="cluster4")
#### X轴的刻度变化
plt.xticks(range(0,150,20))
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/41554601/175806933-d8435db3-3bd8-44a0-8701-b60ecfb7cc85.png)
