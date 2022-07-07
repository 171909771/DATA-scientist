# 载入数据
[Ads_CTR_Optimisation.csv](https://github.com/171909771/DATA-scientist/files/9001080/Ads_CTR_Optimisation.csv)

```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
dataset
```
## 方法一：随机数来验证，作为上面方法的对比
```
import random
record=[]
total=0
for n in range(10000):
    k=random.randrange(0,10)
    record.append(k)
    total=total+dataset.iloc[n,k]
### 绘图
plt.hist(record)
plt.title(f"hitshot number is {total}")
plt.show()
```
![image](https://user-images.githubusercontent.com/41554601/176184088-931663cd-d54d-48cc-80a5-1cb487466163.png)

## 方法二：运行置信区间上界算法
![image](https://user-images.githubusercontent.com/41554601/176180687-64547b6f-7c8f-4eee-968d-a1987682635f.png)
## 先把每个机器运行一遍，然后算每个机器的置信区间的上限，选最高的上限的那个机器继续运行
```
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
```
![image](https://user-images.githubusercontent.com/41554601/176180732-bce3309a-6659-4d82-b497-a532c2e7fd26.png)

## 方法三：Thompson抽样算法
### 每次用概率最高的广告去进行下游实验
![image](https://user-images.githubusercontent.com/41554601/176210629-ab85f6a2-5b72-4192-9dd6-7b21c5e1668b.png)

```
import random
N=10000
d=10
property_1=[0]*d
property_0=[0]*d
total=0
ads_select=[]
for n in range(N):
    value_max=0
    ## 计算beta分布
    for k in range(d):
        value_beta=random.betavariate(property_1[k]+1,property_0[k]+1)
        if value_beta>value_max:
            value_max=value_beta
            set_value=k
    ads_select.append(set_value)
    total+=dataset.iloc[n,set_value]
    if dataset.iloc[n,set_value]==1:
        property_1[set_value]+=1
    else:
        property_0[set_value]+=1
plt.hist(ads_select)
plt.title(f"hitshot number is {total}")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
```
![image](https://user-images.githubusercontent.com/41554601/176209801-a9435092-2b23-4e24-a645-5a1600cd53e6.png)
