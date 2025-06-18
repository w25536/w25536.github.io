---
title: "062_Predicting_Breakfast_Cereal_Ratings_80_cereals"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "062_Predicting_Breakfast_Cereal_Ratings_80_cereals"
use_math: true
classes: wide
---

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("crawford/80-cereals")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/crawford/80-cereals/versions/2



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv(os.path.join(path, "cereal.csv"))
```


```python
df = df.drop(["name"], axis=1)
```


```python
# binary encode
df["type"] = df["type"].replace({"C": 0, "H": 1})
```

    /var/folders/v7/tlyx9w190ks2gfgzd_j0l5c80000gn/T/ipykernel_70863/932868930.py:2: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df['type'] = df['type'].replace({'C': 0, 'H':1})



```python
def onehot_encode(df, column):
    dummies = pd.get_dummies(df[column], dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df


df = onehot_encode(df, "mfr")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 77 entries, 0 to 76
    Data columns (total 21 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   type      77 non-null     int64  
     1   calories  77 non-null     int64  
     2   protein   77 non-null     int64  
     3   fat       77 non-null     int64  
     4   sodium    77 non-null     int64  
     5   fiber     77 non-null     float64
     6   carbo     77 non-null     float64
     7   sugars    77 non-null     int64  
     8   potass    77 non-null     int64  
     9   vitamins  77 non-null     int64  
     10  shelf     77 non-null     int64  
     11  weight    77 non-null     float64
     12  cups      77 non-null     float64
     13  rating    77 non-null     float64
     14  A         77 non-null     int64  
     15  G         77 non-null     int64  
     16  K         77 non-null     int64  
     17  N         77 non-null     int64  
     18  P         77 non-null     int64  
     19  Q         77 non-null     int64  
     20  R         77 non-null     int64  
    dtypes: float64(5), int64(16)
    memory usage: 12.8 KB



```python
y = df["rating"]
X = df.drop(["rating"], axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)
```


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import *
from sklearn.tree import *
from xgboost import XGBRegressor


LR = LinearRegression()
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()
KNR = KNeighborsRegressor()
MLP = MLPRegressor()
XGB = XGBRegressor()
```


```python
models = [LR, DTR, RFR, KNR, KNR, XGB]
result = []
d = {}
for model in models:
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    print(model, ":", r2_score)
    result.append((model, r2_score))
```

    LinearRegression() : 0.9999999999999993
    DecisionTreeRegressor() : 0.5425705963375649
    RandomForestRegressor() : 0.7859891960209286
    KNeighborsRegressor() : 0.251978691444407
    KNeighborsRegressor() : 0.251978691444407
    XGBRegressor(base_score=None, booster=None, callbacks=None,
                 colsample_bylevel=None, colsample_bynode=None,
                 colsample_bytree=None, device=None, early_stopping_rounds=None,
                 enable_categorical=False, eval_metric=None, feature_types=None,
                 gamma=None, grow_policy=None, importance_type=None,
                 interaction_constraints=None, learning_rate=None, max_bin=None,
                 max_cat_threshold=None, max_cat_to_onehot=None,
                 max_delta_step=None, max_depth=None, max_leaves=None,
                 min_child_weight=None, missing=nan, monotone_constraints=None,
                 multi_strategy=None, n_estimators=None, n_jobs=None,
                 num_parallel_tree=None, random_state=None, ...) : 0.700593949307747



```python
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
```


```python
pd.DataFrame(y_pred, columns=["rating"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.384843</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.871292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.042851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68.402973</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34.139764</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40.105965</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31.230054</td>
    </tr>
    <tr>
      <th>7</th>
      <td>41.503540</td>
    </tr>
    <tr>
      <th>8</th>
      <td>59.642837</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41.015492</td>
    </tr>
    <tr>
      <th>10</th>
      <td>59.363994</td>
    </tr>
    <tr>
      <th>11</th>
      <td>49.787445</td>
    </tr>
    <tr>
      <th>12</th>
      <td>22.396513</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19.823573</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39.259197</td>
    </tr>
    <tr>
      <th>15</th>
      <td>53.371007</td>
    </tr>
    <tr>
      <th>16</th>
      <td>53.313813</td>
    </tr>
    <tr>
      <th>17</th>
      <td>29.509541</td>
    </tr>
    <tr>
      <th>18</th>
      <td>45.811716</td>
    </tr>
    <tr>
      <th>19</th>
      <td>36.176196</td>
    </tr>
    <tr>
      <th>20</th>
      <td>35.252444</td>
    </tr>
    <tr>
      <th>21</th>
      <td>39.241114</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36.471512</td>
    </tr>
    <tr>
      <th>23</th>
      <td>45.863325</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
