### 载入数据
[fruit_data_with_colors.txt](https://github.com/171909771/DATA-scientist/files/9175891/fruit_data_with_colors.txt)
## Ridge 回归
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
fruit=pd.read_table("fruit_data_with_colors.txt")
X,y=fruit[["height",'mass','height']],fruit["color_score"]
# 归一化
from sklearn.preprocessing import MinMaxScaler
scl=MinMaxScaler().fit(X_train)
x1_scl=scl.transform(X_train)
x2_scl=scl.transform(X_test)
from sklearn.linear_model import Ridge
## alpha代表L2正则化的系数
linRid2=Ridge(alpha=1).fit(x1_scl, y_train)
print(linRid2.score(x1_scl, y_train))
print(linRid2.score(x2_scl, y_test))
```
## Lasso回归
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
fruit=pd.read_table("fruit_data_with_colors.txt")
X,y=fruit[["height",'mass','height']],fruit["color_score"]
# 归一化
from sklearn.preprocessing import MinMaxScaler
scl=MinMaxScaler().fit(X_train)
x1_scl=scl.transform(X_train)
x2_scl=scl.transform(X_test)
from sklearn.linear_model import Lasso
## 可以加大max_iter迭代的次数
linlass=Lasso(alpha=100,max_iter=10000).fit(x1_scl, y_train)
## 通过lasso 可以看到最相关的自变量
for e in sorted(list(zip(list(X),linlass.coef_)), key= lambda x: -abs(x[1])):
    print(e)
```
