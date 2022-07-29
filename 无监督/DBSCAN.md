### DBSCAN 
#### eps表示数据间的半径，min_samples表示聚类的最小数量
![image](https://user-images.githubusercontent.com/41554601/181861469-3fad5911-d01d-4171-a5f6-de4de587d324.png)
```
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=2,min_samples=4)
cls=db.fit_predict(X_train)
```
