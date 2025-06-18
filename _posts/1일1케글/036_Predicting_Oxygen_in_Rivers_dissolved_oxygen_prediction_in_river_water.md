---
title: "036_Predicting_Oxygen_in_Rivers_dissolved_oxygen_prediction_in_river_water"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "036_Predicting_Oxygen_in_Rivers_dissolved_oxygen_prediction_in_river_water"
use_math: true
classes: wide
---

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vbmokin/dissolved-oxygen-prediction-in-river-water")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/vbmokin/dissolved-oxygen-prediction-in-river-water/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv(os.path.join(path, "train.csv"))
```


```python
df.isna().sum()
```




    Id          0
    target      0
    O2_1        2
    O2_2        2
    O2_3      115
    O2_4      116
    O2_5      114
    O2_6      110
    O2_7      110
    NH4_1       2
    NH4_2       2
    NH4_3     115
    NH4_4     116
    NH4_5     114
    NH4_6     110
    NH4_7     110
    NO2_1       2
    NO2_2       2
    NO2_3     115
    NO2_4     116
    NO2_5     114
    NO2_6     110
    NO2_7     110
    NO3_1       2
    NO3_2       2
    NO3_3     115
    NO3_4     116
    NO3_5     114
    NO3_6     110
    NO3_7     110
    BOD5_1      2
    BOD5_2      2
    BOD5_3    115
    BOD5_4    116
    BOD5_5    114
    BOD5_6    110
    BOD5_7    110
    dtype: int64




```python
null_columns = list(df.columns[(df.isna().sum() > 100)])

df1 = df.drop(null_columns, axis=1)
```


```python
df1.isna().sum()
```




    Id        0
    target    0
    O2_1      2
    O2_2      2
    NH4_1     2
    NH4_2     2
    NO2_1     2
    NO2_2     2
    NO3_1     2
    NO3_2     2
    BOD5_1    2
    BOD5_2    2
    dtype: int64




```python
(df1.isna().sum(axis=0) != 0).sum()

"""
   Col1  Col2  Col3  Col4
0   1    NaN   3    NaN    <- Row 1 has nulls
1   4     5    6     7
2   8    NaN   10   NaN    <- Row 2 has nulls
3   11   NaN   13   NaN    <- Row 3 has nulls

"""
```




    '\n   Col1  Col2  Col3  Col4\n0   1    NaN   3    NaN    <- Row 1 has nulls\n1   4     5    6     7\n2   8    NaN   10   NaN    <- Row 2 has nulls\n3   11   NaN   13   NaN    <- Row 3 has nulls\n\n'




```python
(df1.isna().sum(axis=1) != 0).sum()
```




    3




```python
null_columns = list(df1.columns[df1.isna().sum() > 100])
```


```python
df2 = df1.drop(null_columns, axis=1)
```


```python
df2.isna().sum()
```




    Id        0
    target    0
    O2_1      2
    O2_2      2
    NH4_1     2
    NH4_2     2
    NO2_1     2
    NO2_2     2
    NO3_1     2
    NO3_2     2
    BOD5_1    2
    BOD5_2    2
    dtype: int64




```python
(df2.isna().sum(axis=1) != 0).sum()
(df2.isna().sum(axis=0) != 0).sum()
```




    10




```python
print(f"Number of rows containing null values: {(df2.isna().sum(axis=1) != 0).sum()}")
print(
    f"Number of columns containing null values: {(df2.isna().sum(axis=0) != 0).sum()}"
)
```

    Number of rows containing null values: 3
    Number of columns containing null values: 10



```python
df3 = df2.dropna(axis=0)
```


```python
df3.isna().sum()
```




    Id        0
    target    0
    O2_1      0
    O2_2      0
    NH4_1     0
    NH4_2     0
    NO2_1     0
    NO2_2     0
    NO3_1     0
    NO3_2     0
    BOD5_1    0
    BOD5_2    0
    dtype: int64




```python
df4 = df3.drop(["Id"], axis=1)
```


```python
y = df4["target"]
X = df4.drop(["target"], axis=1)
```


```python
X
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
      <th>O2_1</th>
      <th>O2_2</th>
      <th>NH4_1</th>
      <th>NH4_2</th>
      <th>NO2_1</th>
      <th>NO2_2</th>
      <th>NO3_1</th>
      <th>NO3_2</th>
      <th>BOD5_1</th>
      <th>BOD5_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.875</td>
      <td>9.20</td>
      <td>0.690</td>
      <td>1.040</td>
      <td>0.0940</td>
      <td>0.0990</td>
      <td>1.58</td>
      <td>1.825</td>
      <td>4.80</td>
      <td>5.850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.300</td>
      <td>10.75</td>
      <td>0.710</td>
      <td>0.725</td>
      <td>0.0585</td>
      <td>0.0515</td>
      <td>1.21</td>
      <td>0.905</td>
      <td>5.88</td>
      <td>6.835</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.290</td>
      <td>7.90</td>
      <td>2.210</td>
      <td>2.210</td>
      <td>0.1000</td>
      <td>0.1100</td>
      <td>1.34</td>
      <td>1.250</td>
      <td>3.20</td>
      <td>2.700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.820</td>
      <td>6.80</td>
      <td>0.595</td>
      <td>0.675</td>
      <td>0.0460</td>
      <td>0.0535</td>
      <td>0.59</td>
      <td>0.790</td>
      <td>7.70</td>
      <td>7.055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.000</td>
      <td>6.50</td>
      <td>0.600</td>
      <td>0.900</td>
      <td>0.1800</td>
      <td>0.3400</td>
      <td>1.36</td>
      <td>1.820</td>
      <td>5.50</td>
      <td>5.300</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>142</th>
      <td>7.700</td>
      <td>7.50</td>
      <td>0.380</td>
      <td>1.900</td>
      <td>0.6200</td>
      <td>0.0640</td>
      <td>2.80</td>
      <td>3.330</td>
      <td>5.00</td>
      <td>5.800</td>
    </tr>
    <tr>
      <th>143</th>
      <td>6.300</td>
      <td>5.65</td>
      <td>0.370</td>
      <td>0.500</td>
      <td>0.6900</td>
      <td>0.9500</td>
      <td>4.37</td>
      <td>3.160</td>
      <td>8.00</td>
      <td>8.000</td>
    </tr>
    <tr>
      <th>144</th>
      <td>8.600</td>
      <td>11.00</td>
      <td>2.400</td>
      <td>3.600</td>
      <td>0.1500</td>
      <td>0.1400</td>
      <td>0.53</td>
      <td>3.000</td>
      <td>6.80</td>
      <td>7.200</td>
    </tr>
    <tr>
      <th>145</th>
      <td>9.600</td>
      <td>14.10</td>
      <td>0.310</td>
      <td>0.500</td>
      <td>0.2100</td>
      <td>0.0800</td>
      <td>3.10</td>
      <td>3.500</td>
      <td>5.20</td>
      <td>7.800</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.500</td>
      <td>7.70</td>
      <td>0.190</td>
      <td>0.260</td>
      <td>0.1300</td>
      <td>0.0720</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>3.40</td>
      <td>4.100</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 10 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

scaler = StandardScaler()
q_scaler = PowerTransformer()
scaled_X = scaler.fit_transform(X)
q_scaled_X = q_scaler.fit_transform(X)

pd.DataFrame(q_scaled_X, columns=X.columns)
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
      <th>O2_1</th>
      <th>O2_2</th>
      <th>NH4_1</th>
      <th>NH4_2</th>
      <th>NO2_1</th>
      <th>NO2_2</th>
      <th>NO3_1</th>
      <th>NO3_2</th>
      <th>BOD5_1</th>
      <th>BOD5_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.425462</td>
      <td>0.197201</td>
      <td>0.582635</td>
      <td>0.971205</td>
      <td>0.269592</td>
      <td>0.319885</td>
      <td>-0.440337</td>
      <td>-0.428021</td>
      <td>0.006209</td>
      <td>0.522629</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.544547</td>
      <td>0.663001</td>
      <td>0.623522</td>
      <td>0.501471</td>
      <td>-0.401798</td>
      <td>-0.598370</td>
      <td>-0.696229</td>
      <td>-1.095671</td>
      <td>0.513850</td>
      <td>0.967304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.072976</td>
      <td>-0.262286</td>
      <td>1.997515</td>
      <td>1.810066</td>
      <td>0.362592</td>
      <td>0.483744</td>
      <td>-0.601366</td>
      <td>-0.816648</td>
      <td>-0.847650</td>
      <td>-1.148793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.104262</td>
      <td>-0.717024</td>
      <td>0.370143</td>
      <td>0.407296</td>
      <td>-0.697743</td>
      <td>-0.551446</td>
      <td>-1.251361</td>
      <td>-1.198934</td>
      <td>1.282423</td>
      <td>1.063045</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.006411</td>
      <td>-0.854164</td>
      <td>0.382142</td>
      <td>0.785270</td>
      <td>1.221232</td>
      <td>1.972354</td>
      <td>-0.587279</td>
      <td>-0.431078</td>
      <td>0.340347</td>
      <td>0.261508</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>139</th>
      <td>-0.284981</td>
      <td>-0.419650</td>
      <td>-0.255041</td>
      <td>1.664881</td>
      <td>2.120797</td>
      <td>-0.318377</td>
      <td>0.185812</td>
      <td>0.324063</td>
      <td>0.103738</td>
      <td>0.499299</td>
    </tr>
    <tr>
      <th>140</th>
      <td>-0.864859</td>
      <td>-1.280426</td>
      <td>-0.290372</td>
      <td>0.017883</td>
      <td>2.137578</td>
      <td>2.338861</td>
      <td>0.728598</td>
      <td>0.252238</td>
      <td>1.400704</td>
      <td>1.461286</td>
    </tr>
    <tr>
      <th>141</th>
      <td>0.032105</td>
      <td>0.731384</td>
      <td>2.067546</td>
      <td>2.187739</td>
      <td>0.968061</td>
      <td>0.860436</td>
      <td>-1.317192</td>
      <td>0.182167</td>
      <td>0.914148</td>
      <td>1.125493</td>
    </tr>
    <tr>
      <th>142</th>
      <td>0.345421</td>
      <td>1.462264</td>
      <td>-0.517102</td>
      <td>0.017883</td>
      <td>1.416746</td>
      <td>-0.002360</td>
      <td>0.306388</td>
      <td>0.393376</td>
      <td>0.199578</td>
      <td>1.378668</td>
    </tr>
    <tr>
      <th>143</th>
      <td>-0.774277</td>
      <td>-0.339912</td>
      <td>-1.061115</td>
      <td>-0.732008</td>
      <td>0.758022</td>
      <td>-0.154812</td>
      <td>-2.058888</td>
      <td>-2.135809</td>
      <td>-0.732525</td>
      <td>-0.347439</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 10 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, train_size=0.7)
```


```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

LR = LinearRegression()
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()
KNR = KNeighborsRegressor()
MLP = MLPRegressor()
XGB = XGBRegressor()


models = [LR, DTR, RFR, KNR, KNR, XGB]
result = []

for model in models:
    model.fit(X_train, y_train)
    r2score = model.score(X_test, y_test)
    print(model, ":", r2score)
    #    y_pred = model.predict(X_test)
    #    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #    print(model, ":", rmse)

    result.append((model, r2score))
```

    LinearRegression() : 0.4689967944676089
    DecisionTreeRegressor() : 0.32545186191777353
    RandomForestRegressor() : 0.639000593988786
    KNeighborsRegressor() : 0.3773443619402673
    KNeighborsRegressor() : 0.3773443619402673
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
                 num_parallel_tree=None, random_state=None, ...) : 0.5136872794653518



```python
y.mean()
```




    9.145763888888888




```python
y_new = pd.qcut(y, q=2, labels=[0, 1])
```


```python
sum(y_new) / len(y_new)  # balance so it's okay to use accuracy
```




    0.5




```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y_new, train_size=0.7)
```


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

DTR = DecisionTreeClassifier()
RFR = RandomForestClassifier()
KNR = KNeighborsClassifier()
MLP = MLPClassifier()
XGB = XGBClassifier()

models = [DTR, RFR, KNR, KNR, XGB]
result = []
d = {}

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    print(model, "=> Accuracy: ", accuracy, "F1: ", f1, "BA:", "ba")
    result.append((model, accuracy))
```

    DecisionTreeClassifier() => Accuracy:  0.6818181818181818 F1:  0.6818181818181818 BA: ba
    RandomForestClassifier() => Accuracy:  0.7727272727272727 F1:  0.7619047619047619 BA: ba
    KNeighborsClassifier() => Accuracy:  0.7045454545454546 F1:  0.6976744186046512 BA: ba
    KNeighborsClassifier() => Accuracy:  0.7045454545454546 F1:  0.6976744186046512 BA: ba
    XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, device=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric=None, feature_types=None,
                  gamma=None, grow_policy=None, importance_type=None,
                  interaction_constraints=None, learning_rate=None, max_bin=None,
                  max_cat_threshold=None, max_cat_to_onehot=None,
                  max_delta_step=None, max_depth=None, max_leaves=None,
                  min_child_weight=None, missing=nan, monotone_constraints=None,
                  multi_strategy=None, n_estimators=None, n_jobs=None,
                  num_parallel_tree=None, random_state=None, ...) => Accuracy:  0.8409090909090909 F1:  0.8444444444444444 BA: ba

