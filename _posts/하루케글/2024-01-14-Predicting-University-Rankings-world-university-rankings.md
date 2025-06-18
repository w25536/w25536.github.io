---
title: "Predicting University Rankings world university rankings"
date: 2024-01-14
last_modified_at: 2024-01-14
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Predicting University Rankings world university rankings 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mylesoneill/world-university-rankings")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/mylesoneill/world-university-rankings/versions/2



```python
!/Users/jeongho/.cache/kagglehub/datasets/mylesoneill/world-university-rankings/versions/2
```

    zsh:1: permission denied: /Users/jeongho/.cache/kagglehub/datasets/mylesoneill/world-university-rankings/versions/2



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


import os

df = pd.read_csv(os.path.join(path, "cwurData.csv"))
```


```python
df
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
      <th>world_rank</th>
      <th>institution</th>
      <th>country</th>
      <th>national_rank</th>
      <th>quality_of_education</th>
      <th>alumni_employment</th>
      <th>quality_of_faculty</th>
      <th>publications</th>
      <th>influence</th>
      <th>citations</th>
      <th>broad_impact</th>
      <th>patents</th>
      <th>score</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Harvard University</td>
      <td>USA</td>
      <td>1</td>
      <td>7</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>5</td>
      <td>100.00</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Massachusetts Institute of Technology</td>
      <td>USA</td>
      <td>2</td>
      <td>9</td>
      <td>17</td>
      <td>3</td>
      <td>12</td>
      <td>4</td>
      <td>4</td>
      <td>NaN</td>
      <td>1</td>
      <td>91.67</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Stanford University</td>
      <td>USA</td>
      <td>3</td>
      <td>17</td>
      <td>11</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>15</td>
      <td>89.50</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>University of Cambridge</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>10</td>
      <td>24</td>
      <td>4</td>
      <td>16</td>
      <td>16</td>
      <td>11</td>
      <td>NaN</td>
      <td>50</td>
      <td>86.17</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>California Institute of Technology</td>
      <td>USA</td>
      <td>4</td>
      <td>2</td>
      <td>29</td>
      <td>7</td>
      <td>37</td>
      <td>22</td>
      <td>22</td>
      <td>NaN</td>
      <td>18</td>
      <td>85.21</td>
      <td>2012</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2195</th>
      <td>996</td>
      <td>University of the Algarve</td>
      <td>Portugal</td>
      <td>7</td>
      <td>367</td>
      <td>567</td>
      <td>218</td>
      <td>926</td>
      <td>845</td>
      <td>812</td>
      <td>969.0</td>
      <td>816</td>
      <td>44.03</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>997</td>
      <td>Alexandria University</td>
      <td>Egypt</td>
      <td>4</td>
      <td>236</td>
      <td>566</td>
      <td>218</td>
      <td>997</td>
      <td>908</td>
      <td>645</td>
      <td>981.0</td>
      <td>871</td>
      <td>44.03</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>998</td>
      <td>Federal University of Ceará</td>
      <td>Brazil</td>
      <td>18</td>
      <td>367</td>
      <td>549</td>
      <td>218</td>
      <td>830</td>
      <td>823</td>
      <td>812</td>
      <td>975.0</td>
      <td>824</td>
      <td>44.03</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>999</td>
      <td>University of A Coruña</td>
      <td>Spain</td>
      <td>40</td>
      <td>367</td>
      <td>567</td>
      <td>218</td>
      <td>886</td>
      <td>974</td>
      <td>812</td>
      <td>975.0</td>
      <td>651</td>
      <td>44.02</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>1000</td>
      <td>China Pharmaceutical University</td>
      <td>China</td>
      <td>83</td>
      <td>367</td>
      <td>567</td>
      <td>218</td>
      <td>861</td>
      <td>991</td>
      <td>812</td>
      <td>981.0</td>
      <td>547</td>
      <td>44.02</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
<p>2200 rows × 14 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2200 entries, 0 to 2199
    Data columns (total 14 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   world_rank            2200 non-null   int64  
     1   institution           2200 non-null   object 
     2   country               2200 non-null   object 
     3   national_rank         2200 non-null   int64  
     4   quality_of_education  2200 non-null   int64  
     5   alumni_employment     2200 non-null   int64  
     6   quality_of_faculty    2200 non-null   int64  
     7   publications          2200 non-null   int64  
     8   influence             2200 non-null   int64  
     9   citations             2200 non-null   int64  
     10  broad_impact          2000 non-null   float64
     11  patents               2200 non-null   int64  
     12  score                 2200 non-null   float64
     13  year                  2200 non-null   int64  
    dtypes: float64(2), int64(10), object(2)
    memory usage: 240.8+ KB



```python
np.sum(df.isnull())
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/numpy/core/fromnumeric.py:86: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
      return reduction(axis=axis, out=out, **passkwargs)





    world_rank                0
    institution               0
    country                   0
    national_rank             0
    quality_of_education      0
    alumni_employment         0
    quality_of_faculty        0
    publications              0
    influence                 0
    citations                 0
    broad_impact            200
    patents                   0
    score                     0
    year                      0
    dtype: int64




```python
df = df.drop(["institution", "year", "broad_impact"], axis=1)
```


```python
encoder = LabelEncoder()

mappings = list()

for col in df.select_dtypes("object"):
    df[col] = encoder.fit_transform(df[col])
    mapping_dict = {index: label for index, label in enumerate(encoder.classes_)}
    mappings.append(mapping_dict)
```


```python
mappings
```




    [{0: 'Argentina',
      1: 'Australia',
      2: 'Austria',
      3: 'Belgium',
      4: 'Brazil',
      5: 'Bulgaria',
      6: 'Canada',
      7: 'Chile',
      8: 'China',
      9: 'Colombia',
      10: 'Croatia',
      11: 'Cyprus',
      12: 'Czech Republic',
      13: 'Denmark',
      14: 'Egypt',
      15: 'Estonia',
      16: 'Finland',
      17: 'France',
      18: 'Germany',
      19: 'Greece',
      20: 'Hong Kong',
      21: 'Hungary',
      22: 'Iceland',
      23: 'India',
      24: 'Iran',
      25: 'Ireland',
      26: 'Israel',
      27: 'Italy',
      28: 'Japan',
      29: 'Lebanon',
      30: 'Lithuania',
      31: 'Malaysia',
      32: 'Mexico',
      33: 'Netherlands',
      34: 'New Zealand',
      35: 'Norway',
      36: 'Poland',
      37: 'Portugal',
      38: 'Puerto Rico',
      39: 'Romania',
      40: 'Russia',
      41: 'Saudi Arabia',
      42: 'Serbia',
      43: 'Singapore',
      44: 'Slovak Republic',
      45: 'Slovenia',
      46: 'South Africa',
      47: 'South Korea',
      48: 'Spain',
      49: 'Sweden',
      50: 'Switzerland',
      51: 'Taiwan',
      52: 'Thailand',
      53: 'Turkey',
      54: 'USA',
      55: 'Uganda',
      56: 'United Arab Emirates',
      57: 'United Kingdom',
      58: 'Uruguay'}]




```python
X = df.drop(["world_rank"], axis=1)
y = df["world_rank"]
```


```python
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
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
      <th>country</th>
      <th>national_rank</th>
      <th>quality_of_education</th>
      <th>alumni_employment</th>
      <th>quality_of_faculty</th>
      <th>publications</th>
      <th>influence</th>
      <th>citations</th>
      <th>patents</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.035555</td>
      <td>-0.759305</td>
      <td>-2.199214</td>
      <td>-1.864211</td>
      <td>-2.777926</td>
      <td>-1.511102</td>
      <td>-1.512871</td>
      <td>-1.560375</td>
      <td>-1.563683</td>
      <td>6.727841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.035555</td>
      <td>-0.739974</td>
      <td>-2.182808</td>
      <td>-1.821370</td>
      <td>-2.746694</td>
      <td>-1.474881</td>
      <td>-1.502979</td>
      <td>-1.549025</td>
      <td>-1.578285</td>
      <td>5.654255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.035555</td>
      <td>-0.720642</td>
      <td>-2.117185</td>
      <td>-1.853501</td>
      <td>-2.715462</td>
      <td>-1.501224</td>
      <td>-1.509574</td>
      <td>-1.556592</td>
      <td>-1.527178</td>
      <td>5.374581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.191751</td>
      <td>-0.759305</td>
      <td>-2.174605</td>
      <td>-1.783884</td>
      <td>-2.731078</td>
      <td>-1.461710</td>
      <td>-1.463409</td>
      <td>-1.522540</td>
      <td>-1.399410</td>
      <td>4.945405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.035555</td>
      <td>-0.701311</td>
      <td>-2.240229</td>
      <td>-1.757109</td>
      <td>-2.684229</td>
      <td>-1.392561</td>
      <td>-1.443625</td>
      <td>-1.480922</td>
      <td>-1.516227</td>
      <td>4.821678</td>
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
      <th>2195</th>
      <td>0.150445</td>
      <td>-0.643316</td>
      <td>0.753847</td>
      <td>1.123952</td>
      <td>0.610775</td>
      <td>1.534754</td>
      <td>1.270193</td>
      <td>1.508032</td>
      <td>1.396881</td>
      <td>-0.485678</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>-1.047057</td>
      <td>-0.701311</td>
      <td>-0.320739</td>
      <td>1.118597</td>
      <td>0.610775</td>
      <td>1.768544</td>
      <td>1.477933</td>
      <td>0.876190</td>
      <td>1.597659</td>
      <td>-0.485678</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>-1.567710</td>
      <td>-0.430670</td>
      <td>0.753847</td>
      <td>1.027560</td>
      <td>0.610775</td>
      <td>1.218643</td>
      <td>1.197648</td>
      <td>1.508032</td>
      <td>1.426085</td>
      <td>-0.485678</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>0.723163</td>
      <td>-0.005378</td>
      <td>0.753847</td>
      <td>1.123952</td>
      <td>0.610775</td>
      <td>1.403041</td>
      <td>1.695566</td>
      <td>1.508032</td>
      <td>0.794547</td>
      <td>-0.486967</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>-1.359448</td>
      <td>0.825876</td>
      <td>0.753847</td>
      <td>1.123952</td>
      <td>0.610775</td>
      <td>1.320721</td>
      <td>1.751623</td>
      <td>1.508032</td>
      <td>0.414894</td>
      <td>-0.486967</td>
    </tr>
  </tbody>
</table>
<p>2200 rows × 10 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
```


```python
model = LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)  # R^2 value
```




    0.9220688121916525




```python

```
