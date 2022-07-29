###  数据预处理
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
### Hierarchical
```
from scipy.cluster.hierarchy import ward,dendrogram
plt.figure()
dendrogram(ward(X_train))
plt.show()
```
