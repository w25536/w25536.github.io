---
title: "Predicting FIFA Man of the Match predict fifa 2018 man of the match"
date: 2024-01-15
last_modified_at: 2024-01-15
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Predicting FIFA Man of the Match predict fifa 2018 man of the match 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mathan/fifa-2018-match-statistics")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/mathan/fifa-2018-match-statistics/versions/20



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier
import tensorflow as tf

import os

df = pd.read_csv(os.path.join(path, "FIFA 2018 Statistics.csv"))
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
      <th>Date</th>
      <th>Team</th>
      <th>Opponent</th>
      <th>Goal Scored</th>
      <th>Ball Possession %</th>
      <th>Attempts</th>
      <th>On-Target</th>
      <th>Off-Target</th>
      <th>Blocked</th>
      <th>Corners</th>
      <th>...</th>
      <th>Yellow Card</th>
      <th>Yellow &amp; Red</th>
      <th>Red</th>
      <th>Man of the Match</th>
      <th>1st Goal</th>
      <th>Round</th>
      <th>PSO</th>
      <th>Goals in PSO</th>
      <th>Own goals</th>
      <th>Own goal Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14-06-2018</td>
      <td>Russia</td>
      <td>Saudi Arabia</td>
      <td>5</td>
      <td>40</td>
      <td>13</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>12.0</td>
      <td>Group Stage</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14-06-2018</td>
      <td>Saudi Arabia</td>
      <td>Russia</td>
      <td>0</td>
      <td>60</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>NaN</td>
      <td>Group Stage</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15-06-2018</td>
      <td>Egypt</td>
      <td>Uruguay</td>
      <td>0</td>
      <td>43</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>NaN</td>
      <td>Group Stage</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15-06-2018</td>
      <td>Uruguay</td>
      <td>Egypt</td>
      <td>1</td>
      <td>57</td>
      <td>14</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>89.0</td>
      <td>Group Stage</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15-06-2018</td>
      <td>Morocco</td>
      <td>Iran</td>
      <td>0</td>
      <td>64</td>
      <td>13</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>NaN</td>
      <td>Group Stage</td>
      <td>No</td>
      <td>0</td>
      <td>1.0</td>
      <td>90.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>11-07-2018</td>
      <td>England</td>
      <td>Croatia</td>
      <td>1</td>
      <td>46</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>5.0</td>
      <td>Semi- Finals</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>124</th>
      <td>14-07-2018</td>
      <td>Belgium</td>
      <td>England</td>
      <td>2</td>
      <td>43</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>4.0</td>
      <td>3rd Place</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>125</th>
      <td>14-07-2018</td>
      <td>England</td>
      <td>Belgium</td>
      <td>0</td>
      <td>57</td>
      <td>15</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>NaN</td>
      <td>3rd Place</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>126</th>
      <td>15-07-2018</td>
      <td>France</td>
      <td>Croatia</td>
      <td>4</td>
      <td>39</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>18.0</td>
      <td>Final</td>
      <td>No</td>
      <td>0</td>
      <td>1.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>15-07-2018</td>
      <td>Croatia</td>
      <td>France</td>
      <td>2</td>
      <td>61</td>
      <td>15</td>
      <td>3</td>
      <td>8</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>28.0</td>
      <td>Final</td>
      <td>No</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 27 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 128 entries, 0 to 127
    Data columns (total 27 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Date                    128 non-null    object 
     1   Team                    128 non-null    object 
     2   Opponent                128 non-null    object 
     3   Goal Scored             128 non-null    int64  
     4   Ball Possession %       128 non-null    int64  
     5   Attempts                128 non-null    int64  
     6   On-Target               128 non-null    int64  
     7   Off-Target              128 non-null    int64  
     8   Blocked                 128 non-null    int64  
     9   Corners                 128 non-null    int64  
     10  Offsides                128 non-null    int64  
     11  Free Kicks              128 non-null    int64  
     12  Saves                   128 non-null    int64  
     13  Pass Accuracy %         128 non-null    int64  
     14  Passes                  128 non-null    int64  
     15  Distance Covered (Kms)  128 non-null    int64  
     16  Fouls Committed         128 non-null    int64  
     17  Yellow Card             128 non-null    int64  
     18  Yellow & Red            128 non-null    int64  
     19  Red                     128 non-null    int64  
     20  Man of the Match        128 non-null    object 
     21  1st Goal                94 non-null     float64
     22  Round                   128 non-null    object 
     23  PSO                     128 non-null    object 
     24  Goals in PSO            128 non-null    int64  
     25  Own goals               12 non-null     float64
     26  Own goal Time           12 non-null     float64
    dtypes: float64(3), int64(18), object(6)
    memory usage: 27.1+ KB



```python
np.sum(df.isnull())
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/numpy/core/fromnumeric.py:86: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
      return reduction(axis=axis, out=out, **passkwargs)





    Date                        0
    Team                        0
    Opponent                    0
    Goal Scored                 0
    Ball Possession %           0
    Attempts                    0
    On-Target                   0
    Off-Target                  0
    Blocked                     0
    Corners                     0
    Offsides                    0
    Free Kicks                  0
    Saves                       0
    Pass Accuracy %             0
    Passes                      0
    Distance Covered (Kms)      0
    Fouls Committed             0
    Yellow Card                 0
    Yellow & Red                0
    Red                         0
    Man of the Match            0
    1st Goal                   34
    Round                       0
    PSO                         0
    Goals in PSO                0
    Own goals                 116
    Own goal Time             116
    dtype: int64




```python
df = df.drop(["Own goals", "Own goal Time", "Date"], axis=1)
```


```python
np.sum(df["1st Goal"]) / len(df["1st Goal"])
```




    28.9765625




```python
df["1st Goal"] = df["1st Goal"].fillna(df["1st Goal"].mean())
```


```python
for col in df.select_dtypes("object").columns:
    print(f"{col}", df[col].unique())
```

    Team ['Russia' 'Saudi Arabia' 'Egypt' 'Uruguay' 'Morocco' 'Iran' 'Portugal'
     'Spain' 'France' 'Australia' 'Argentina' 'Iceland' 'Peru' 'Denmark'
     'Croatia' 'Nigeria' 'Costa Rica' 'Serbia' 'Germany' 'Mexico' 'Brazil'
     'Switzerland' 'Sweden' 'Korea Republic' 'Belgium' 'Panama' 'Tunisia'
     'England' 'Colombia' 'Japan' 'Poland' 'Senegal']
    Opponent ['Saudi Arabia' 'Russia' 'Uruguay' 'Egypt' 'Iran' 'Morocco' 'Spain'
     'Portugal' 'Australia' 'France' 'Iceland' 'Argentina' 'Denmark' 'Peru'
     'Nigeria' 'Croatia' 'Serbia' 'Costa Rica' 'Mexico' 'Germany'
     'Switzerland' 'Brazil' 'Korea Republic' 'Sweden' 'Panama' 'Belgium'
     'England' 'Tunisia' 'Japan' 'Colombia' 'Senegal' 'Poland']
    Man of the Match ['Yes' 'No']
    Round ['Group Stage' 'Round of 16' 'Quarter Finals' 'Semi- Finals' '3rd Place'
     'Final']
    PSO ['No' 'Yes']



```python
round_values = list(df["Round"].unique())

print(round_values)

round_mappings = {label: index for index, label in enumerate(round_values)}
print(round_mappings)

df["Round"] = df["Round"].apply(lambda x: round_mappings[x])
```

    ['Group Stage', 'Round of 16', 'Quarter Finals', 'Semi- Finals', '3rd Place', 'Final']
    {'Group Stage': 0, 'Round of 16': 1, 'Quarter Finals': 2, 'Semi- Finals': 3, '3rd Place': 4, 'Final': 5}



```python
encoder = LabelEncoder()

df["PSO"] = encoder.fit_transform(df["PSO"])
pso_mappings = {label: index for index, label in enumerate(encoder.classes_)}
```


```python
df["Man of the Match"] = encoder.fit_transform(df["Man of the Match"])
motm_mappings = {label: index for index, label in enumerate(encoder.classes_)}
```


```python
print(pso_mappings)
print(motm_mappings)
```

    {'No': 0, 'Yes': 1}
    {'No': 0, 'Yes': 1}



```python
pd.get_dummies(df["Team"], dtype=int)
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
      <th>Argentina</th>
      <th>Australia</th>
      <th>Belgium</th>
      <th>Brazil</th>
      <th>Colombia</th>
      <th>Costa Rica</th>
      <th>Croatia</th>
      <th>Denmark</th>
      <th>Egypt</th>
      <th>England</th>
      <th>...</th>
      <th>Portugal</th>
      <th>Russia</th>
      <th>Saudi Arabia</th>
      <th>Senegal</th>
      <th>Serbia</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>Switzerland</th>
      <th>Tunisia</th>
      <th>Uruguay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 32 columns</p>
</div>




```python
pd.get_dummies(df["Opponent"], dtype=int)
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
      <th>Argentina</th>
      <th>Australia</th>
      <th>Belgium</th>
      <th>Brazil</th>
      <th>Colombia</th>
      <th>Costa Rica</th>
      <th>Croatia</th>
      <th>Denmark</th>
      <th>Egypt</th>
      <th>England</th>
      <th>...</th>
      <th>Portugal</th>
      <th>Russia</th>
      <th>Saudi Arabia</th>
      <th>Senegal</th>
      <th>Serbia</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>Switzerland</th>
      <th>Tunisia</th>
      <th>Uruguay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 32 columns</p>
</div>




```python
pd.get_dummies(df["Opponent"].apply(lambda x: "opp_" + x), dtype=int)
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
      <th>opp_Argentina</th>
      <th>opp_Australia</th>
      <th>opp_Belgium</th>
      <th>opp_Brazil</th>
      <th>opp_Colombia</th>
      <th>opp_Costa Rica</th>
      <th>opp_Croatia</th>
      <th>opp_Denmark</th>
      <th>opp_Egypt</th>
      <th>opp_England</th>
      <th>...</th>
      <th>opp_Portugal</th>
      <th>opp_Russia</th>
      <th>opp_Saudi Arabia</th>
      <th>opp_Senegal</th>
      <th>opp_Serbia</th>
      <th>opp_Spain</th>
      <th>opp_Sweden</th>
      <th>opp_Switzerland</th>
      <th>opp_Tunisia</th>
      <th>opp_Uruguay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 32 columns</p>
</div>




```python
df["Opponent"] = df["Opponent"].apply(lambda x: "opp_" + x)
```


```python
df_concat = pd.concat(
    [df, pd.get_dummies(df["Team"]), pd.get_dummies(df["Opponent"])], axis=1
)
```


```python
df_concat.drop(["Team", "Opponent"], axis=1, inplace=True)
```


```python
df_concat
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
      <th>Goal Scored</th>
      <th>Ball Possession %</th>
      <th>Attempts</th>
      <th>On-Target</th>
      <th>Off-Target</th>
      <th>Blocked</th>
      <th>Corners</th>
      <th>Offsides</th>
      <th>Free Kicks</th>
      <th>Saves</th>
      <th>...</th>
      <th>opp_Portugal</th>
      <th>opp_Russia</th>
      <th>opp_Saudi Arabia</th>
      <th>opp_Senegal</th>
      <th>opp_Serbia</th>
      <th>opp_Spain</th>
      <th>opp_Sweden</th>
      <th>opp_Switzerland</th>
      <th>opp_Tunisia</th>
      <th>opp_Uruguay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>40</td>
      <td>13</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>60</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>25</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>43</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>57</td>
      <td>14</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>64</td>
      <td>13</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>14</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>1</td>
      <td>46</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>24</td>
      <td>5</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2</td>
      <td>43</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0</td>
      <td>57</td>
      <td>15</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>12</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>126</th>
      <td>4</td>
      <td>39</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>127</th>
      <td>2</td>
      <td>61</td>
      <td>15</td>
      <td>3</td>
      <td>8</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>15</td>
      <td>3</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 86 columns</p>
</div>




```python
np.sum(df_concat.dtypes == "object")
```




    0




```python
y = df_concat["Man of the Match"]
X = df_concat.drop(["Man of the Match"], axis=1)
```


```python
y
```




    0      1
    1      0
    2      0
    3      1
    4      0
          ..
    123    0
    124    1
    125    0
    126    1
    127    0
    Name: Man of the Match, Length: 128, dtype: int64




```python
scaler = RobustScaler()

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
      <th>Goal Scored</th>
      <th>Ball Possession %</th>
      <th>Attempts</th>
      <th>On-Target</th>
      <th>Off-Target</th>
      <th>Blocked</th>
      <th>Corners</th>
      <th>Offsides</th>
      <th>Free Kicks</th>
      <th>Saves</th>
      <th>...</th>
      <th>opp_Portugal</th>
      <th>opp_Russia</th>
      <th>opp_Saudi Arabia</th>
      <th>opp_Senegal</th>
      <th>opp_Serbia</th>
      <th>opp_Spain</th>
      <th>opp_Sweden</th>
      <th>opp_Switzerland</th>
      <th>opp_Tunisia</th>
      <th>opp_Uruguay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>-0.6250</td>
      <td>0.166667</td>
      <td>1.166667</td>
      <td>-0.666667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>-0.571429</td>
      <td>-0.666667</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.5</td>
      <td>0.6250</td>
      <td>-1.000000</td>
      <td>-1.166667</td>
      <td>-0.666667</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.0</td>
      <td>1.428571</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5</td>
      <td>-0.4375</td>
      <td>-0.666667</td>
      <td>-0.166667</td>
      <td>-0.666667</td>
      <td>-0.444444</td>
      <td>-1.666667</td>
      <td>0.0</td>
      <td>-1.142857</td>
      <td>0.333333</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.4375</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>0.333333</td>
      <td>0.444444</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-0.285714</td>
      <td>0.333333</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.5</td>
      <td>0.8750</td>
      <td>0.166667</td>
      <td>-0.166667</td>
      <td>0.333333</td>
      <td>0.444444</td>
      <td>0.000000</td>
      <td>-0.5</td>
      <td>-0.142857</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0.0</td>
      <td>-0.2500</td>
      <td>-0.166667</td>
      <td>-0.833333</td>
      <td>0.333333</td>
      <td>0.444444</td>
      <td>-0.333333</td>
      <td>1.0</td>
      <td>1.285714</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0.5</td>
      <td>-0.4375</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>-0.666667</td>
      <td>0.888889</td>
      <td>-0.333333</td>
      <td>0.0</td>
      <td>-1.428571</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>-0.5</td>
      <td>0.4375</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.5</td>
      <td>-0.428571</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>1.5</td>
      <td>-0.6875</td>
      <td>-0.666667</td>
      <td>0.833333</td>
      <td>-1.333333</td>
      <td>-0.888889</td>
      <td>-1.000000</td>
      <td>0.0</td>
      <td>-0.142857</td>
      <td>-0.333333</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0.5</td>
      <td>0.6875</td>
      <td>0.500000</td>
      <td>-0.166667</td>
      <td>1.000000</td>
      <td>0.444444</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 85 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```


```python
sk_model = MLPClassifier(hidden_layer_sizes=(32, 32))
sk_model.fit(X_train, y_train)
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/sklearn/neural_network/_multilayer_perceptron.py:690: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(





<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MLPClassifier(hidden_layer_sizes=(32, 32))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;MLPClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html">?<span>Documentation for MLPClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>MLPClassifier(hidden_layer_sizes=(32, 32))</pre></div> </div></div></div></div>




```python
inputs = tf.keras.Input(shape=(85,))
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)

tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)

tf_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
```


```python
tf_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=16,
    epochs=200,
)
```

    Epoch 1/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.4789 - loss: 0.7420 - val_accuracy: 0.3889 - val_loss: 0.7718
    Epoch 2/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.4678 - loss: 0.7302 - val_accuracy: 0.3889 - val_loss: 0.7464
    Epoch 3/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.6395 - loss: 0.6754 - val_accuracy: 0.3889 - val_loss: 0.7272
    Epoch 4/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.5362 - loss: 0.6756 - val_accuracy: 0.4444 - val_loss: 0.7115
    Epoch 5/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.6203 - loss: 0.6320 - val_accuracy: 0.5556 - val_loss: 0.7005
    Epoch 6/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.7030 - loss: 0.6211 - val_accuracy: 0.5556 - val_loss: 0.6913
    Epoch 7/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - accuracy: 0.6578 - loss: 0.6265 - val_accuracy: 0.6111 - val_loss: 0.6812
    Epoch 8/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - accuracy: 0.7575 - loss: 0.5880 - val_accuracy: 0.6111 - val_loss: 0.6741
    Epoch 9/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.7752 - loss: 0.5691 - val_accuracy: 0.6111 - val_loss: 0.6672
    Epoch 10/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.7344 - loss: 0.5772 - val_accuracy: 0.6111 - val_loss: 0.6610
    Epoch 11/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.7924 - loss: 0.5431 - val_accuracy: 0.6111 - val_loss: 0.6534
    Epoch 12/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.8011 - loss: 0.5357 - val_accuracy: 0.6111 - val_loss: 0.6440
    Epoch 13/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.7592 - loss: 0.5396 - val_accuracy: 0.6111 - val_loss: 0.6359
    Epoch 14/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.7615 - loss: 0.5267 - val_accuracy: 0.6667 - val_loss: 0.6282
    Epoch 15/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.7815 - loss: 0.4969 - val_accuracy: 0.6111 - val_loss: 0.6201
    Epoch 16/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.7940 - loss: 0.4930 - val_accuracy: 0.6111 - val_loss: 0.6133
    Epoch 17/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8494 - loss: 0.4678 - val_accuracy: 0.6111 - val_loss: 0.6054
    Epoch 18/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8437 - loss: 0.4764 - val_accuracy: 0.6667 - val_loss: 0.5989
    Epoch 19/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.8576 - loss: 0.4282 - val_accuracy: 0.6667 - val_loss: 0.5933
    Epoch 20/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8263 - loss: 0.4309 - val_accuracy: 0.6667 - val_loss: 0.5874
    Epoch 21/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.8952 - loss: 0.3789 - val_accuracy: 0.6667 - val_loss: 0.5808
    Epoch 22/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.8874 - loss: 0.3700 - val_accuracy: 0.6667 - val_loss: 0.5756
    Epoch 23/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8925 - loss: 0.3615 - val_accuracy: 0.6667 - val_loss: 0.5705
    Epoch 24/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8769 - loss: 0.3558 - val_accuracy: 0.6667 - val_loss: 0.5661
    Epoch 25/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9479 - loss: 0.2997 - val_accuracy: 0.6667 - val_loss: 0.5604
    Epoch 26/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9508 - loss: 0.2932 - val_accuracy: 0.6667 - val_loss: 0.5551
    Epoch 27/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9422 - loss: 0.2739 - val_accuracy: 0.6667 - val_loss: 0.5488
    Epoch 28/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9590 - loss: 0.2658 - val_accuracy: 0.6667 - val_loss: 0.5450
    Epoch 29/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9555 - loss: 0.2427 - val_accuracy: 0.6667 - val_loss: 0.5432
    Epoch 30/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9611 - loss: 0.2351 - val_accuracy: 0.6667 - val_loss: 0.5438
    Epoch 31/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9677 - loss: 0.2054 - val_accuracy: 0.7222 - val_loss: 0.5485
    Epoch 32/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9521 - loss: 0.2221 - val_accuracy: 0.7222 - val_loss: 0.5527
    Epoch 33/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9607 - loss: 0.1978 - val_accuracy: 0.7222 - val_loss: 0.5579
    Epoch 34/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9628 - loss: 0.1958 - val_accuracy: 0.6667 - val_loss: 0.5623
    Epoch 35/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9663 - loss: 0.1884 - val_accuracy: 0.6667 - val_loss: 0.5606
    Epoch 36/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9732 - loss: 0.1579 - val_accuracy: 0.6667 - val_loss: 0.5614
    Epoch 37/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9892 - loss: 0.1489 - val_accuracy: 0.6667 - val_loss: 0.5652
    Epoch 38/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.9892 - loss: 0.1422 - val_accuracy: 0.6667 - val_loss: 0.5702
    Epoch 39/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9892 - loss: 0.1324 - val_accuracy: 0.6667 - val_loss: 0.5736
    Epoch 40/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - accuracy: 0.9840 - loss: 0.1221 - val_accuracy: 0.6667 - val_loss: 0.5776
    Epoch 41/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9892 - loss: 0.1292 - val_accuracy: 0.6667 - val_loss: 0.5826
    Epoch 42/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9736 - loss: 0.1163 - val_accuracy: 0.6667 - val_loss: 0.5931
    Epoch 43/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9736 - loss: 0.1085 - val_accuracy: 0.6667 - val_loss: 0.6073
    Epoch 44/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0941 - val_accuracy: 0.6667 - val_loss: 0.6143
    Epoch 45/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0937 - val_accuracy: 0.6667 - val_loss: 0.6174
    Epoch 46/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0848 - val_accuracy: 0.6667 - val_loss: 0.6248
    Epoch 47/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0818 - val_accuracy: 0.6667 - val_loss: 0.6305
    Epoch 48/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0800 - val_accuracy: 0.6667 - val_loss: 0.6384
    Epoch 49/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0706 - val_accuracy: 0.6667 - val_loss: 0.6475
    Epoch 50/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0601 - val_accuracy: 0.6667 - val_loss: 0.6586
    Epoch 51/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0562 - val_accuracy: 0.6667 - val_loss: 0.6691
    Epoch 52/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0530 - val_accuracy: 0.6667 - val_loss: 0.6820
    Epoch 53/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step - accuracy: 1.0000 - loss: 0.0461 - val_accuracy: 0.6667 - val_loss: 0.6881
    Epoch 54/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0471 - val_accuracy: 0.6111 - val_loss: 0.7041
    Epoch 55/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0376 - val_accuracy: 0.6111 - val_loss: 0.7202
    Epoch 56/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0379 - val_accuracy: 0.6111 - val_loss: 0.7283
    Epoch 57/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0427 - val_accuracy: 0.6111 - val_loss: 0.7343
    Epoch 58/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0372 - val_accuracy: 0.6111 - val_loss: 0.7389
    Epoch 59/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0315 - val_accuracy: 0.6111 - val_loss: 0.7449
    Epoch 60/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0303 - val_accuracy: 0.6111 - val_loss: 0.7465
    Epoch 61/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0265 - val_accuracy: 0.6111 - val_loss: 0.7518
    Epoch 62/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0249 - val_accuracy: 0.6111 - val_loss: 0.7549
    Epoch 63/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0281 - val_accuracy: 0.6111 - val_loss: 0.7600
    Epoch 64/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0235 - val_accuracy: 0.6111 - val_loss: 0.7650
    Epoch 65/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0199 - val_accuracy: 0.6111 - val_loss: 0.7677
    Epoch 66/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0202 - val_accuracy: 0.6111 - val_loss: 0.7723
    Epoch 67/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0188 - val_accuracy: 0.6111 - val_loss: 0.7778
    Epoch 68/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0204 - val_accuracy: 0.6111 - val_loss: 0.7821
    Epoch 69/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0174 - val_accuracy: 0.6111 - val_loss: 0.7883
    Epoch 70/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0184 - val_accuracy: 0.6111 - val_loss: 0.7920
    Epoch 71/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0167 - val_accuracy: 0.6111 - val_loss: 0.7959
    Epoch 72/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0138 - val_accuracy: 0.6111 - val_loss: 0.8019
    Epoch 73/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0124 - val_accuracy: 0.6111 - val_loss: 0.8060
    Epoch 74/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0163 - val_accuracy: 0.6111 - val_loss: 0.8106
    Epoch 75/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0153 - val_accuracy: 0.6111 - val_loss: 0.8116
    Epoch 76/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0150 - val_accuracy: 0.6111 - val_loss: 0.8159
    Epoch 77/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0129 - val_accuracy: 0.6111 - val_loss: 0.8223
    Epoch 78/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0104 - val_accuracy: 0.6111 - val_loss: 0.8261
    Epoch 79/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0105 - val_accuracy: 0.6111 - val_loss: 0.8305
    Epoch 80/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0108 - val_accuracy: 0.6111 - val_loss: 0.8343
    Epoch 81/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0092 - val_accuracy: 0.6111 - val_loss: 0.8432
    Epoch 82/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0084 - val_accuracy: 0.6111 - val_loss: 0.8481
    Epoch 83/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0106 - val_accuracy: 0.6111 - val_loss: 0.8491
    Epoch 84/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0083 - val_accuracy: 0.6111 - val_loss: 0.8514
    Epoch 85/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0087 - val_accuracy: 0.6111 - val_loss: 0.8533
    Epoch 86/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0092 - val_accuracy: 0.6111 - val_loss: 0.8559
    Epoch 87/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0069 - val_accuracy: 0.6111 - val_loss: 0.8594
    Epoch 88/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0065 - val_accuracy: 0.6111 - val_loss: 0.8642
    Epoch 89/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0085 - val_accuracy: 0.6111 - val_loss: 0.8690
    Epoch 90/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0063 - val_accuracy: 0.6111 - val_loss: 0.8734
    Epoch 91/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0061 - val_accuracy: 0.6111 - val_loss: 0.8771
    Epoch 92/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0062 - val_accuracy: 0.6111 - val_loss: 0.8805
    Epoch 93/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0055 - val_accuracy: 0.6111 - val_loss: 0.8820
    Epoch 94/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0058 - val_accuracy: 0.6111 - val_loss: 0.8842
    Epoch 95/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0061 - val_accuracy: 0.6111 - val_loss: 0.8873
    Epoch 96/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - accuracy: 1.0000 - loss: 0.0066 - val_accuracy: 0.6111 - val_loss: 0.8894
    Epoch 97/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0047 - val_accuracy: 0.6111 - val_loss: 0.8907
    Epoch 98/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0053 - val_accuracy: 0.6111 - val_loss: 0.8933
    Epoch 99/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0040 - val_accuracy: 0.6111 - val_loss: 0.8939
    Epoch 100/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0047 - val_accuracy: 0.6111 - val_loss: 0.8941
    Epoch 101/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0048 - val_accuracy: 0.6111 - val_loss: 0.8963
    Epoch 102/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0048 - val_accuracy: 0.6111 - val_loss: 0.8972
    Epoch 103/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0047 - val_accuracy: 0.6111 - val_loss: 0.8996
    Epoch 104/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0042 - val_accuracy: 0.6111 - val_loss: 0.9032
    Epoch 105/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0041 - val_accuracy: 0.6111 - val_loss: 0.9050
    Epoch 106/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0047 - val_accuracy: 0.6111 - val_loss: 0.9077
    Epoch 107/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0038 - val_accuracy: 0.6111 - val_loss: 0.9127
    Epoch 108/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0034 - val_accuracy: 0.6111 - val_loss: 0.9156
    Epoch 109/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0035 - val_accuracy: 0.6111 - val_loss: 0.9175
    Epoch 110/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0036 - val_accuracy: 0.6111 - val_loss: 0.9208
    Epoch 111/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0038 - val_accuracy: 0.6111 - val_loss: 0.9231
    Epoch 112/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0033 - val_accuracy: 0.6111 - val_loss: 0.9257
    Epoch 113/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0031 - val_accuracy: 0.6111 - val_loss: 0.9281
    Epoch 114/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0031 - val_accuracy: 0.6111 - val_loss: 0.9326
    Epoch 115/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0037 - val_accuracy: 0.6111 - val_loss: 0.9351
    Epoch 116/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0033 - val_accuracy: 0.6111 - val_loss: 0.9380
    Epoch 117/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0033 - val_accuracy: 0.6111 - val_loss: 0.9400
    Epoch 118/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0033 - val_accuracy: 0.6111 - val_loss: 0.9417
    Epoch 119/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0029 - val_accuracy: 0.6111 - val_loss: 0.9438
    Epoch 120/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0029 - val_accuracy: 0.6111 - val_loss: 0.9454
    Epoch 121/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 1.0000 - loss: 0.0027 - val_accuracy: 0.6111 - val_loss: 0.9475
    Epoch 122/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0031 - val_accuracy: 0.6111 - val_loss: 0.9498
    Epoch 123/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0027 - val_accuracy: 0.6111 - val_loss: 0.9528
    Epoch 124/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9549
    Epoch 125/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0026 - val_accuracy: 0.6111 - val_loss: 0.9555
    Epoch 126/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0025 - val_accuracy: 0.6111 - val_loss: 0.9573
    Epoch 127/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0025 - val_accuracy: 0.6111 - val_loss: 0.9599
    Epoch 128/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9608
    Epoch 129/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9633
    Epoch 130/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0026 - val_accuracy: 0.6111 - val_loss: 0.9640
    Epoch 131/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9653
    Epoch 132/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9663
    Epoch 133/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 0.6111 - val_loss: 0.9681
    Epoch 134/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0024 - val_accuracy: 0.6111 - val_loss: 0.9707
    Epoch 135/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0020 - val_accuracy: 0.6111 - val_loss: 0.9731
    Epoch 136/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.6111 - val_loss: 0.9739
    Epoch 137/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0020 - val_accuracy: 0.6111 - val_loss: 0.9755
    Epoch 138/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 0.6111 - val_loss: 0.9766
    Epoch 139/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0020 - val_accuracy: 0.6111 - val_loss: 0.9777
    Epoch 140/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0020 - val_accuracy: 0.6111 - val_loss: 0.9792
    Epoch 141/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 0.6111 - val_loss: 0.9812
    Epoch 142/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 0.6111 - val_loss: 0.9825
    Epoch 143/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 0.6111 - val_loss: 0.9843
    Epoch 144/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - accuracy: 1.0000 - loss: 0.0019 - val_accuracy: 0.6111 - val_loss: 0.9860
    Epoch 145/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 0.6111 - val_loss: 0.9876
    Epoch 146/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0016 - val_accuracy: 0.6111 - val_loss: 0.9889
    Epoch 147/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 0.6111 - val_loss: 0.9892
    Epoch 148/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.6111 - val_loss: 0.9903
    Epoch 149/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 0.9918
    Epoch 150/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 0.6111 - val_loss: 0.9932
    Epoch 151/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.6111 - val_loss: 0.9946
    Epoch 152/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 0.6111 - val_loss: 0.9962
    Epoch 153/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 0.9982
    Epoch 154/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 0.9998
    Epoch 155/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 1.0014
    Epoch 156/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 1.0027
    Epoch 157/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - accuracy: 1.0000 - loss: 0.0016 - val_accuracy: 0.6111 - val_loss: 1.0035
    Epoch 158/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0016 - val_accuracy: 0.6111 - val_loss: 1.0041
    Epoch 159/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 1.0048
    Epoch 160/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.6111 - val_loss: 1.0062
    Epoch 161/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 0.6111 - val_loss: 1.0075
    Epoch 162/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.6111 - val_loss: 1.0089
    Epoch 163/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0013 - val_accuracy: 0.6111 - val_loss: 1.0097
    Epoch 164/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.6111 - val_loss: 1.0121
    Epoch 165/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 0.6111 - val_loss: 1.0140
    Epoch 166/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 0.6111 - val_loss: 1.0156
    Epoch 167/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.6111 - val_loss: 1.0163
    Epoch 168/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0169
    Epoch 169/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0184
    Epoch 170/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 0.6111 - val_loss: 1.0199
    Epoch 171/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 0.6111 - val_loss: 1.0222
    Epoch 172/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0233
    Epoch 173/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0246
    Epoch 174/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0256
    Epoch 175/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0273
    Epoch 176/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 0.6111 - val_loss: 1.0289
    Epoch 177/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 0.6111 - val_loss: 1.0303
    Epoch 178/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 0.6111 - val_loss: 1.0317
    Epoch 179/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 9.6512e-04 - val_accuracy: 0.6111 - val_loss: 1.0326
    Epoch 180/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 9.3576e-04 - val_accuracy: 0.6111 - val_loss: 1.0345
    Epoch 181/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 9.5367e-04 - val_accuracy: 0.6111 - val_loss: 1.0360
    Epoch 182/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 9.1145e-04 - val_accuracy: 0.6111 - val_loss: 1.0377
    Epoch 183/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 8.9403e-04 - val_accuracy: 0.6111 - val_loss: 1.0394
    Epoch 184/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 8.9942e-04 - val_accuracy: 0.6111 - val_loss: 1.0397
    Epoch 185/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 9.0985e-04 - val_accuracy: 0.6111 - val_loss: 1.0410
    Epoch 186/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.7400e-04 - val_accuracy: 0.6111 - val_loss: 1.0422
    Epoch 187/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step - accuracy: 1.0000 - loss: 8.9465e-04 - val_accuracy: 0.6111 - val_loss: 1.0435
    Epoch 188/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.5975e-04 - val_accuracy: 0.6111 - val_loss: 1.0442
    Epoch 189/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 9.8913e-04 - val_accuracy: 0.6111 - val_loss: 1.0449
    Epoch 190/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.7707e-04 - val_accuracy: 0.6111 - val_loss: 1.0462
    Epoch 191/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 0.6111 - val_loss: 1.0469
    Epoch 192/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.0689e-04 - val_accuracy: 0.6111 - val_loss: 1.0481
    Epoch 193/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.0898e-04 - val_accuracy: 0.6111 - val_loss: 1.0489
    Epoch 194/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.0958e-04 - val_accuracy: 0.6111 - val_loss: 1.0495
    Epoch 195/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 9.2997e-04 - val_accuracy: 0.6111 - val_loss: 1.0509
    Epoch 196/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 7.9158e-04 - val_accuracy: 0.6111 - val_loss: 1.0515
    Epoch 197/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 1.0000 - loss: 7.0981e-04 - val_accuracy: 0.6111 - val_loss: 1.0529
    Epoch 198/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - accuracy: 1.0000 - loss: 8.5940e-04 - val_accuracy: 0.6111 - val_loss: 1.0541
    Epoch 199/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 1.0000 - loss: 6.7073e-04 - val_accuracy: 0.6111 - val_loss: 1.0555
    Epoch 200/200
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 1.0000 - loss: 8.3621e-04 - val_accuracy: 0.6111 - val_loss: 1.0562





    <keras.src.callbacks.history.History at 0x32e3149a0>




```python
print(sk_model.score(X_test, y_test))
print(tf_model.evaluate(X_test, y_test, verbose=False)[1])
```

    0.5897435897435898
    0.5641025900840759



```python
from pycaret.classification import *

setup(df_concat, target=df_concat["Man of the Match"], train_size=0.7, session_id=42)
```


<style type="text/css">
#T_68f19_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_68f19">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_68f19_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_68f19_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_68f19_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_68f19_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_68f19_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_68f19_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_68f19_row1_col1" class="data row1 col1" >Man of the Match_y</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_68f19_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_68f19_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_68f19_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_68f19_row3_col1" class="data row3 col1" >(128, 87)</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_68f19_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_68f19_row4_col1" class="data row4 col1" >(128, 87)</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_68f19_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_68f19_row5_col1" class="data row5 col1" >(89, 87)</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_68f19_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_68f19_row6_col1" class="data row6 col1" >(39, 87)</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_68f19_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_68f19_row7_col1" class="data row7 col1" >22</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_68f19_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_68f19_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_68f19_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_68f19_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_68f19_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_68f19_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_68f19_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_68f19_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_68f19_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_68f19_row12_col1" class="data row12 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_68f19_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_68f19_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_68f19_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_68f19_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_68f19_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_68f19_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_68f19_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_68f19_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_68f19_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_68f19_row17_col1" class="data row17 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_68f19_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_68f19_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_68f19_row18_col1" class="data row18 col1" >6dd6</td>
    </tr>
  </tbody>
</table>






    <pycaret.classification.oop.ClassificationExperiment at 0x331ef2fd0>




```python
compare_models()
```






<style type="text/css">
#T_c09a4 th {
  text-align: left;
}
#T_c09a4_row0_col0, #T_c09a4_row0_col2, #T_c09a4_row0_col3, #T_c09a4_row0_col4, #T_c09a4_row0_col5, #T_c09a4_row1_col0, #T_c09a4_row1_col1, #T_c09a4_row1_col2, #T_c09a4_row1_col4, #T_c09a4_row1_col6, #T_c09a4_row1_col7, #T_c09a4_row2_col0, #T_c09a4_row2_col1, #T_c09a4_row2_col3, #T_c09a4_row2_col5, #T_c09a4_row2_col6, #T_c09a4_row2_col7, #T_c09a4_row3_col0, #T_c09a4_row3_col1, #T_c09a4_row3_col2, #T_c09a4_row3_col3, #T_c09a4_row3_col4, #T_c09a4_row3_col5, #T_c09a4_row3_col6, #T_c09a4_row3_col7, #T_c09a4_row4_col0, #T_c09a4_row4_col1, #T_c09a4_row4_col2, #T_c09a4_row4_col3, #T_c09a4_row4_col4, #T_c09a4_row4_col5, #T_c09a4_row4_col6, #T_c09a4_row4_col7, #T_c09a4_row5_col0, #T_c09a4_row5_col1, #T_c09a4_row5_col2, #T_c09a4_row5_col3, #T_c09a4_row5_col4, #T_c09a4_row5_col5, #T_c09a4_row5_col6, #T_c09a4_row5_col7, #T_c09a4_row6_col0, #T_c09a4_row6_col1, #T_c09a4_row6_col2, #T_c09a4_row6_col3, #T_c09a4_row6_col4, #T_c09a4_row6_col5, #T_c09a4_row6_col6, #T_c09a4_row6_col7, #T_c09a4_row7_col0, #T_c09a4_row7_col1, #T_c09a4_row7_col2, #T_c09a4_row7_col3, #T_c09a4_row7_col4, #T_c09a4_row7_col5, #T_c09a4_row7_col6, #T_c09a4_row7_col7, #T_c09a4_row8_col0, #T_c09a4_row8_col1, #T_c09a4_row8_col2, #T_c09a4_row8_col3, #T_c09a4_row8_col4, #T_c09a4_row8_col5, #T_c09a4_row8_col6, #T_c09a4_row8_col7, #T_c09a4_row9_col0, #T_c09a4_row9_col1, #T_c09a4_row9_col2, #T_c09a4_row9_col3, #T_c09a4_row9_col4, #T_c09a4_row9_col5, #T_c09a4_row9_col6, #T_c09a4_row9_col7, #T_c09a4_row10_col0, #T_c09a4_row10_col1, #T_c09a4_row10_col2, #T_c09a4_row10_col3, #T_c09a4_row10_col4, #T_c09a4_row10_col5, #T_c09a4_row10_col6, #T_c09a4_row10_col7, #T_c09a4_row11_col0, #T_c09a4_row11_col1, #T_c09a4_row11_col2, #T_c09a4_row11_col3, #T_c09a4_row11_col4, #T_c09a4_row11_col5, #T_c09a4_row11_col6, #T_c09a4_row11_col7, #T_c09a4_row12_col0, #T_c09a4_row12_col1, #T_c09a4_row12_col2, #T_c09a4_row12_col3, #T_c09a4_row12_col4, #T_c09a4_row12_col5, #T_c09a4_row12_col6, #T_c09a4_row12_col7, #T_c09a4_row13_col0, #T_c09a4_row13_col1, #T_c09a4_row13_col2, #T_c09a4_row13_col3, #T_c09a4_row13_col4, #T_c09a4_row13_col5, #T_c09a4_row13_col6, #T_c09a4_row13_col7, #T_c09a4_row14_col0, #T_c09a4_row14_col1, #T_c09a4_row14_col2, #T_c09a4_row14_col3, #T_c09a4_row14_col4, #T_c09a4_row14_col5, #T_c09a4_row14_col6, #T_c09a4_row14_col7, #T_c09a4_row15_col0, #T_c09a4_row15_col1, #T_c09a4_row15_col2, #T_c09a4_row15_col3, #T_c09a4_row15_col4, #T_c09a4_row15_col5, #T_c09a4_row15_col6, #T_c09a4_row15_col7 {
  text-align: left;
}
#T_c09a4_row0_col1, #T_c09a4_row0_col6, #T_c09a4_row0_col7, #T_c09a4_row1_col3, #T_c09a4_row1_col5, #T_c09a4_row2_col2, #T_c09a4_row2_col4 {
  text-align: left;
  background-color: yellow;
}
#T_c09a4_row0_col8, #T_c09a4_row1_col8, #T_c09a4_row2_col8, #T_c09a4_row3_col8, #T_c09a4_row4_col8, #T_c09a4_row5_col8, #T_c09a4_row6_col8, #T_c09a4_row7_col8, #T_c09a4_row10_col8, #T_c09a4_row11_col8, #T_c09a4_row12_col8, #T_c09a4_row13_col8, #T_c09a4_row15_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_c09a4_row8_col8, #T_c09a4_row9_col8, #T_c09a4_row14_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_c09a4">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c09a4_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_c09a4_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_c09a4_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_c09a4_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_c09a4_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_c09a4_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_c09a4_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_c09a4_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_c09a4_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c09a4_level0_row0" class="row_heading level0 row0" >lightgbm</th>
      <td id="T_c09a4_row0_col0" class="data row0 col0" >Light Gradient Boosting Machine</td>
      <td id="T_c09a4_row0_col1" class="data row0 col1" >0.5750</td>
      <td id="T_c09a4_row0_col2" class="data row0 col2" >0.5875</td>
      <td id="T_c09a4_row0_col3" class="data row0 col3" >0.5800</td>
      <td id="T_c09a4_row0_col4" class="data row0 col4" >0.5267</td>
      <td id="T_c09a4_row0_col5" class="data row0 col5" >0.5474</td>
      <td id="T_c09a4_row0_col6" class="data row0 col6" >0.1297</td>
      <td id="T_c09a4_row0_col7" class="data row0 col7" >0.1366</td>
      <td id="T_c09a4_row0_col8" class="data row0 col8" >0.0430</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row1" class="row_heading level0 row1" >catboost</th>
      <td id="T_c09a4_row1_col0" class="data row1 col0" >CatBoost Classifier</td>
      <td id="T_c09a4_row1_col1" class="data row1 col1" >0.5625</td>
      <td id="T_c09a4_row1_col2" class="data row1 col2" >0.6025</td>
      <td id="T_c09a4_row1_col3" class="data row1 col3" >0.6800</td>
      <td id="T_c09a4_row1_col4" class="data row1 col4" >0.5879</td>
      <td id="T_c09a4_row1_col5" class="data row1 col5" >0.6156</td>
      <td id="T_c09a4_row1_col6" class="data row1 col6" >0.1248</td>
      <td id="T_c09a4_row1_col7" class="data row1 col7" >0.1331</td>
      <td id="T_c09a4_row1_col8" class="data row1 col8" >0.1460</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row2" class="row_heading level0 row2" >rf</th>
      <td id="T_c09a4_row2_col0" class="data row2 col0" >Random Forest Classifier</td>
      <td id="T_c09a4_row2_col1" class="data row2 col1" >0.5514</td>
      <td id="T_c09a4_row2_col2" class="data row2 col2" >0.6138</td>
      <td id="T_c09a4_row2_col3" class="data row2 col3" >0.6150</td>
      <td id="T_c09a4_row2_col4" class="data row2 col4" >0.6012</td>
      <td id="T_c09a4_row2_col5" class="data row2 col5" >0.5803</td>
      <td id="T_c09a4_row2_col6" class="data row2 col6" >0.1025</td>
      <td id="T_c09a4_row2_col7" class="data row2 col7" >0.1162</td>
      <td id="T_c09a4_row2_col8" class="data row2 col8" >0.0230</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row3" class="row_heading level0 row3" >qda</th>
      <td id="T_c09a4_row3_col0" class="data row3 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_c09a4_row3_col1" class="data row3 col1" >0.5500</td>
      <td id="T_c09a4_row3_col2" class="data row3 col2" >0.5450</td>
      <td id="T_c09a4_row3_col3" class="data row3 col3" >0.5950</td>
      <td id="T_c09a4_row3_col4" class="data row3 col4" >0.5614</td>
      <td id="T_c09a4_row3_col5" class="data row3 col5" >0.5655</td>
      <td id="T_c09a4_row3_col6" class="data row3 col6" >0.0972</td>
      <td id="T_c09a4_row3_col7" class="data row3 col7" >0.1080</td>
      <td id="T_c09a4_row3_col8" class="data row3 col8" >0.0070</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row4" class="row_heading level0 row4" >xgboost</th>
      <td id="T_c09a4_row4_col0" class="data row4 col0" >Extreme Gradient Boosting</td>
      <td id="T_c09a4_row4_col1" class="data row4 col1" >0.5292</td>
      <td id="T_c09a4_row4_col2" class="data row4 col2" >0.5738</td>
      <td id="T_c09a4_row4_col3" class="data row4 col3" >0.5750</td>
      <td id="T_c09a4_row4_col4" class="data row4 col4" >0.5705</td>
      <td id="T_c09a4_row4_col5" class="data row4 col5" >0.5445</td>
      <td id="T_c09a4_row4_col6" class="data row4 col6" >0.0425</td>
      <td id="T_c09a4_row4_col7" class="data row4 col7" >0.0539</td>
      <td id="T_c09a4_row4_col8" class="data row4 col8" >0.0150</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row5" class="row_heading level0 row5" >et</th>
      <td id="T_c09a4_row5_col0" class="data row5 col0" >Extra Trees Classifier</td>
      <td id="T_c09a4_row5_col1" class="data row5 col1" >0.5278</td>
      <td id="T_c09a4_row5_col2" class="data row5 col2" >0.5163</td>
      <td id="T_c09a4_row5_col3" class="data row5 col3" >0.5600</td>
      <td id="T_c09a4_row5_col4" class="data row5 col4" >0.5875</td>
      <td id="T_c09a4_row5_col5" class="data row5 col5" >0.5418</td>
      <td id="T_c09a4_row5_col6" class="data row5 col6" >0.0625</td>
      <td id="T_c09a4_row5_col7" class="data row5 col7" >0.0583</td>
      <td id="T_c09a4_row5_col8" class="data row5 col8" >0.0290</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row6" class="row_heading level0 row6" >lr</th>
      <td id="T_c09a4_row6_col0" class="data row6 col0" >Logistic Regression</td>
      <td id="T_c09a4_row6_col1" class="data row6 col1" >0.5194</td>
      <td id="T_c09a4_row6_col2" class="data row6 col2" >0.5850</td>
      <td id="T_c09a4_row6_col3" class="data row6 col3" >0.5550</td>
      <td id="T_c09a4_row6_col4" class="data row6 col4" >0.5656</td>
      <td id="T_c09a4_row6_col5" class="data row6 col5" >0.5176</td>
      <td id="T_c09a4_row6_col6" class="data row6 col6" >0.0338</td>
      <td id="T_c09a4_row6_col7" class="data row6 col7" >0.0361</td>
      <td id="T_c09a4_row6_col8" class="data row6 col8" >0.2640</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row7" class="row_heading level0 row7" >dt</th>
      <td id="T_c09a4_row7_col0" class="data row7 col0" >Decision Tree Classifier</td>
      <td id="T_c09a4_row7_col1" class="data row7 col1" >0.4958</td>
      <td id="T_c09a4_row7_col2" class="data row7 col2" >0.4925</td>
      <td id="T_c09a4_row7_col3" class="data row7 col3" >0.6150</td>
      <td id="T_c09a4_row7_col4" class="data row7 col4" >0.5112</td>
      <td id="T_c09a4_row7_col5" class="data row7 col5" >0.5497</td>
      <td id="T_c09a4_row7_col6" class="data row7 col6" >-0.0189</td>
      <td id="T_c09a4_row7_col7" class="data row7 col7" >-0.0142</td>
      <td id="T_c09a4_row7_col8" class="data row7 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row8" class="row_heading level0 row8" >nb</th>
      <td id="T_c09a4_row8_col0" class="data row8 col0" >Naive Bayes</td>
      <td id="T_c09a4_row8_col1" class="data row8 col1" >0.4944</td>
      <td id="T_c09a4_row8_col2" class="data row8 col2" >0.4750</td>
      <td id="T_c09a4_row8_col3" class="data row8 col3" >0.4550</td>
      <td id="T_c09a4_row8_col4" class="data row8 col4" >0.5352</td>
      <td id="T_c09a4_row8_col5" class="data row8 col5" >0.4635</td>
      <td id="T_c09a4_row8_col6" class="data row8 col6" >-0.0008</td>
      <td id="T_c09a4_row8_col7" class="data row8 col7" >-0.0027</td>
      <td id="T_c09a4_row8_col8" class="data row8 col8" >0.0050</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row9" class="row_heading level0 row9" >svm</th>
      <td id="T_c09a4_row9_col0" class="data row9 col0" >SVM - Linear Kernel</td>
      <td id="T_c09a4_row9_col1" class="data row9 col1" >0.4944</td>
      <td id="T_c09a4_row9_col2" class="data row9 col2" >0.4400</td>
      <td id="T_c09a4_row9_col3" class="data row9 col3" >0.5200</td>
      <td id="T_c09a4_row9_col4" class="data row9 col4" >0.3444</td>
      <td id="T_c09a4_row9_col5" class="data row9 col5" >0.3608</td>
      <td id="T_c09a4_row9_col6" class="data row9 col6" >0.0182</td>
      <td id="T_c09a4_row9_col7" class="data row9 col7" >0.0316</td>
      <td id="T_c09a4_row9_col8" class="data row9 col8" >0.0050</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row10" class="row_heading level0 row10" >gbc</th>
      <td id="T_c09a4_row10_col0" class="data row10 col0" >Gradient Boosting Classifier</td>
      <td id="T_c09a4_row10_col1" class="data row10 col1" >0.4847</td>
      <td id="T_c09a4_row10_col2" class="data row10 col2" >0.5175</td>
      <td id="T_c09a4_row10_col3" class="data row10 col3" >0.6100</td>
      <td id="T_c09a4_row10_col4" class="data row10 col4" >0.5398</td>
      <td id="T_c09a4_row10_col5" class="data row10 col5" >0.5507</td>
      <td id="T_c09a4_row10_col6" class="data row10 col6" >-0.0373</td>
      <td id="T_c09a4_row10_col7" class="data row10 col7" >-0.0400</td>
      <td id="T_c09a4_row10_col8" class="data row10 col8" >0.0130</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row11" class="row_heading level0 row11" >knn</th>
      <td id="T_c09a4_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_c09a4_row11_col1" class="data row11 col1" >0.4708</td>
      <td id="T_c09a4_row11_col2" class="data row11 col2" >0.5019</td>
      <td id="T_c09a4_row11_col3" class="data row11 col3" >0.5550</td>
      <td id="T_c09a4_row11_col4" class="data row11 col4" >0.4405</td>
      <td id="T_c09a4_row11_col5" class="data row11 col5" >0.4824</td>
      <td id="T_c09a4_row11_col6" class="data row11 col6" >-0.0681</td>
      <td id="T_c09a4_row11_col7" class="data row11 col7" >-0.0793</td>
      <td id="T_c09a4_row11_col8" class="data row11 col8" >0.0090</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row12" class="row_heading level0 row12" >ridge</th>
      <td id="T_c09a4_row12_col0" class="data row12 col0" >Ridge Classifier</td>
      <td id="T_c09a4_row12_col1" class="data row12 col1" >0.4597</td>
      <td id="T_c09a4_row12_col2" class="data row12 col2" >0.5625</td>
      <td id="T_c09a4_row12_col3" class="data row12 col3" >0.5100</td>
      <td id="T_c09a4_row12_col4" class="data row12 col4" >0.4844</td>
      <td id="T_c09a4_row12_col5" class="data row12 col5" >0.4557</td>
      <td id="T_c09a4_row12_col6" class="data row12 col6" >-0.0832</td>
      <td id="T_c09a4_row12_col7" class="data row12 col7" >-0.0875</td>
      <td id="T_c09a4_row12_col8" class="data row12 col8" >0.0130</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row13" class="row_heading level0 row13" >ada</th>
      <td id="T_c09a4_row13_col0" class="data row13 col0" >Ada Boost Classifier</td>
      <td id="T_c09a4_row13_col1" class="data row13 col1" >0.4500</td>
      <td id="T_c09a4_row13_col2" class="data row13 col2" >0.4162</td>
      <td id="T_c09a4_row13_col3" class="data row13 col3" >0.5050</td>
      <td id="T_c09a4_row13_col4" class="data row13 col4" >0.4202</td>
      <td id="T_c09a4_row13_col5" class="data row13 col5" >0.4424</td>
      <td id="T_c09a4_row13_col6" class="data row13 col6" >-0.1091</td>
      <td id="T_c09a4_row13_col7" class="data row13 col7" >-0.1393</td>
      <td id="T_c09a4_row13_col8" class="data row13 col8" >0.0140</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row14" class="row_heading level0 row14" >dummy</th>
      <td id="T_c09a4_row14_col0" class="data row14 col0" >Dummy Classifier</td>
      <td id="T_c09a4_row14_col1" class="data row14 col1" >0.4500</td>
      <td id="T_c09a4_row14_col2" class="data row14 col2" >0.5000</td>
      <td id="T_c09a4_row14_col3" class="data row14 col3" >0.5000</td>
      <td id="T_c09a4_row14_col4" class="data row14 col4" >0.2278</td>
      <td id="T_c09a4_row14_col5" class="data row14 col5" >0.3128</td>
      <td id="T_c09a4_row14_col6" class="data row14 col6" >0.0000</td>
      <td id="T_c09a4_row14_col7" class="data row14 col7" >0.0000</td>
      <td id="T_c09a4_row14_col8" class="data row14 col8" >0.0050</td>
    </tr>
    <tr>
      <th id="T_c09a4_level0_row15" class="row_heading level0 row15" >lda</th>
      <td id="T_c09a4_row15_col0" class="data row15 col0" >Linear Discriminant Analysis</td>
      <td id="T_c09a4_row15_col1" class="data row15 col1" >0.4472</td>
      <td id="T_c09a4_row15_col2" class="data row15 col2" >0.4688</td>
      <td id="T_c09a4_row15_col3" class="data row15 col3" >0.4800</td>
      <td id="T_c09a4_row15_col4" class="data row15 col4" >0.4283</td>
      <td id="T_c09a4_row15_col5" class="data row15 col5" >0.4496</td>
      <td id="T_c09a4_row15_col6" class="data row15 col6" >-0.1232</td>
      <td id="T_c09a4_row15_col7" class="data row15 col7" >-0.1372</td>
      <td id="T_c09a4_row15_col8" class="data row15 col8" >0.0060</td>
    </tr>
  </tbody>
</table>










<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMClassifier(boosting_type=&#x27;gbdt&#x27;, class_weight=None, colsample_bytree=1.0,
               importance_type=&#x27;split&#x27;, learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
               random_state=42, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LGBMClassifier<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LGBMClassifier(boosting_type=&#x27;gbdt&#x27;, class_weight=None, colsample_bytree=1.0,
               importance_type=&#x27;split&#x27;, learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
               random_state=42, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)</pre></div> </div></div></div></div>




```python
setup(df, target=df["Man of the Match"], train_size=0.7, session_id=42)
```


<style type="text/css">
#T_cc80c_row9_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_cc80c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cc80c_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_cc80c_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cc80c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cc80c_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_cc80c_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cc80c_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_cc80c_row1_col1" class="data row1 col1" >Man of the Match_y</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cc80c_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_cc80c_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cc80c_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_cc80c_row3_col1" class="data row3 col1" >(128, 25)</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cc80c_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_cc80c_row4_col1" class="data row4 col1" >(128, 25)</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cc80c_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_cc80c_row5_col1" class="data row5 col1" >(89, 25)</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cc80c_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_cc80c_row6_col1" class="data row6 col1" >(39, 25)</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cc80c_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_cc80c_row7_col1" class="data row7 col1" >22</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cc80c_row8_col0" class="data row8 col0" >Categorical features</td>
      <td id="T_cc80c_row8_col1" class="data row8 col1" >2</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cc80c_row9_col0" class="data row9 col0" >Preprocess</td>
      <td id="T_cc80c_row9_col1" class="data row9 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_cc80c_row10_col0" class="data row10 col0" >Imputation type</td>
      <td id="T_cc80c_row10_col1" class="data row10 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_cc80c_row11_col0" class="data row11 col0" >Numeric imputation</td>
      <td id="T_cc80c_row11_col1" class="data row11 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_cc80c_row12_col0" class="data row12 col0" >Categorical imputation</td>
      <td id="T_cc80c_row12_col1" class="data row12 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_cc80c_row13_col0" class="data row13 col0" >Maximum one-hot encoding</td>
      <td id="T_cc80c_row13_col1" class="data row13 col1" >25</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_cc80c_row14_col0" class="data row14 col0" >Encoding method</td>
      <td id="T_cc80c_row14_col1" class="data row14 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_cc80c_row15_col0" class="data row15 col0" >Fold Generator</td>
      <td id="T_cc80c_row15_col1" class="data row15 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_cc80c_row16_col0" class="data row16 col0" >Fold Number</td>
      <td id="T_cc80c_row16_col1" class="data row16 col1" >10</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_cc80c_row17_col0" class="data row17 col0" >CPU Jobs</td>
      <td id="T_cc80c_row17_col1" class="data row17 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_cc80c_row18_col0" class="data row18 col0" >Use GPU</td>
      <td id="T_cc80c_row18_col1" class="data row18 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_cc80c_row19_col0" class="data row19 col0" >Log Experiment</td>
      <td id="T_cc80c_row19_col1" class="data row19 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_cc80c_row20_col0" class="data row20 col0" >Experiment Name</td>
      <td id="T_cc80c_row20_col1" class="data row20 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_cc80c_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_cc80c_row21_col0" class="data row21 col0" >USI</td>
      <td id="T_cc80c_row21_col1" class="data row21 col1" >bd26</td>
    </tr>
  </tbody>
</table>






    <pycaret.classification.oop.ClassificationExperiment at 0x3341d8610>




```python
compare_models()
```






<style type="text/css">
#T_8327a th {
  text-align: left;
}
#T_8327a_row0_col0, #T_8327a_row0_col4, #T_8327a_row1_col0, #T_8327a_row1_col1, #T_8327a_row1_col2, #T_8327a_row1_col3, #T_8327a_row1_col5, #T_8327a_row1_col6, #T_8327a_row1_col7, #T_8327a_row2_col0, #T_8327a_row2_col1, #T_8327a_row2_col2, #T_8327a_row2_col3, #T_8327a_row2_col4, #T_8327a_row2_col5, #T_8327a_row2_col6, #T_8327a_row2_col7, #T_8327a_row3_col0, #T_8327a_row3_col1, #T_8327a_row3_col2, #T_8327a_row3_col3, #T_8327a_row3_col4, #T_8327a_row3_col5, #T_8327a_row3_col6, #T_8327a_row3_col7, #T_8327a_row4_col0, #T_8327a_row4_col1, #T_8327a_row4_col2, #T_8327a_row4_col3, #T_8327a_row4_col4, #T_8327a_row4_col5, #T_8327a_row4_col6, #T_8327a_row4_col7, #T_8327a_row5_col0, #T_8327a_row5_col1, #T_8327a_row5_col2, #T_8327a_row5_col3, #T_8327a_row5_col4, #T_8327a_row5_col5, #T_8327a_row5_col6, #T_8327a_row5_col7, #T_8327a_row6_col0, #T_8327a_row6_col1, #T_8327a_row6_col2, #T_8327a_row6_col3, #T_8327a_row6_col4, #T_8327a_row6_col5, #T_8327a_row6_col6, #T_8327a_row6_col7, #T_8327a_row7_col0, #T_8327a_row7_col1, #T_8327a_row7_col2, #T_8327a_row7_col3, #T_8327a_row7_col4, #T_8327a_row7_col5, #T_8327a_row7_col6, #T_8327a_row7_col7, #T_8327a_row8_col0, #T_8327a_row8_col1, #T_8327a_row8_col2, #T_8327a_row8_col3, #T_8327a_row8_col4, #T_8327a_row8_col5, #T_8327a_row8_col6, #T_8327a_row8_col7, #T_8327a_row9_col0, #T_8327a_row9_col1, #T_8327a_row9_col2, #T_8327a_row9_col3, #T_8327a_row9_col4, #T_8327a_row9_col5, #T_8327a_row9_col6, #T_8327a_row9_col7, #T_8327a_row10_col0, #T_8327a_row10_col1, #T_8327a_row10_col2, #T_8327a_row10_col3, #T_8327a_row10_col4, #T_8327a_row10_col5, #T_8327a_row10_col6, #T_8327a_row10_col7, #T_8327a_row11_col0, #T_8327a_row11_col1, #T_8327a_row11_col2, #T_8327a_row11_col3, #T_8327a_row11_col4, #T_8327a_row11_col5, #T_8327a_row11_col6, #T_8327a_row11_col7, #T_8327a_row12_col0, #T_8327a_row12_col1, #T_8327a_row12_col2, #T_8327a_row12_col3, #T_8327a_row12_col4, #T_8327a_row12_col5, #T_8327a_row12_col6, #T_8327a_row12_col7, #T_8327a_row13_col0, #T_8327a_row13_col1, #T_8327a_row13_col2, #T_8327a_row13_col3, #T_8327a_row13_col4, #T_8327a_row13_col5, #T_8327a_row13_col6, #T_8327a_row13_col7, #T_8327a_row14_col0, #T_8327a_row14_col1, #T_8327a_row14_col2, #T_8327a_row14_col3, #T_8327a_row14_col4, #T_8327a_row14_col5, #T_8327a_row14_col6, #T_8327a_row14_col7, #T_8327a_row15_col0, #T_8327a_row15_col1, #T_8327a_row15_col2, #T_8327a_row15_col3, #T_8327a_row15_col4, #T_8327a_row15_col5, #T_8327a_row15_col6, #T_8327a_row15_col7 {
  text-align: left;
}
#T_8327a_row0_col1, #T_8327a_row0_col2, #T_8327a_row0_col3, #T_8327a_row0_col5, #T_8327a_row0_col6, #T_8327a_row0_col7, #T_8327a_row1_col4 {
  text-align: left;
  background-color: yellow;
}
#T_8327a_row0_col8, #T_8327a_row1_col8, #T_8327a_row3_col8, #T_8327a_row4_col8, #T_8327a_row5_col8, #T_8327a_row6_col8, #T_8327a_row7_col8, #T_8327a_row8_col8, #T_8327a_row9_col8, #T_8327a_row10_col8, #T_8327a_row11_col8, #T_8327a_row12_col8, #T_8327a_row13_col8, #T_8327a_row14_col8, #T_8327a_row15_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_8327a_row2_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_8327a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8327a_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_8327a_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_8327a_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_8327a_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_8327a_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_8327a_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_8327a_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_8327a_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_8327a_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8327a_level0_row0" class="row_heading level0 row0" >nb</th>
      <td id="T_8327a_row0_col0" class="data row0 col0" >Naive Bayes</td>
      <td id="T_8327a_row0_col1" class="data row0 col1" >0.5500</td>
      <td id="T_8327a_row0_col2" class="data row0 col2" >0.6175</td>
      <td id="T_8327a_row0_col3" class="data row0 col3" >0.6250</td>
      <td id="T_8327a_row0_col4" class="data row0 col4" >0.5667</td>
      <td id="T_8327a_row0_col5" class="data row0 col5" >0.5411</td>
      <td id="T_8327a_row0_col6" class="data row0 col6" >0.1284</td>
      <td id="T_8327a_row0_col7" class="data row0 col7" >0.1510</td>
      <td id="T_8327a_row0_col8" class="data row0 col8" >0.0080</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row1" class="row_heading level0 row1" >dt</th>
      <td id="T_8327a_row1_col0" class="data row1 col0" >Decision Tree Classifier</td>
      <td id="T_8327a_row1_col1" class="data row1 col1" >0.5264</td>
      <td id="T_8327a_row1_col2" class="data row1 col2" >0.5325</td>
      <td id="T_8327a_row1_col3" class="data row1 col3" >0.4100</td>
      <td id="T_8327a_row1_col4" class="data row1 col4" >0.6167</td>
      <td id="T_8327a_row1_col5" class="data row1 col5" >0.4399</td>
      <td id="T_8327a_row1_col6" class="data row1 col6" >0.0704</td>
      <td id="T_8327a_row1_col7" class="data row1 col7" >0.0991</td>
      <td id="T_8327a_row1_col8" class="data row1 col8" >0.0100</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row2" class="row_heading level0 row2" >lda</th>
      <td id="T_8327a_row2_col0" class="data row2 col0" >Linear Discriminant Analysis</td>
      <td id="T_8327a_row2_col1" class="data row2 col1" >0.5264</td>
      <td id="T_8327a_row2_col2" class="data row2 col2" >0.5100</td>
      <td id="T_8327a_row2_col3" class="data row2 col3" >0.5550</td>
      <td id="T_8327a_row2_col4" class="data row2 col4" >0.5548</td>
      <td id="T_8327a_row2_col5" class="data row2 col5" >0.5240</td>
      <td id="T_8327a_row2_col6" class="data row2 col6" >0.0555</td>
      <td id="T_8327a_row2_col7" class="data row2 col7" >0.0781</td>
      <td id="T_8327a_row2_col8" class="data row2 col8" >0.0070</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row3" class="row_heading level0 row3" >lr</th>
      <td id="T_8327a_row3_col0" class="data row3 col0" >Logistic Regression</td>
      <td id="T_8327a_row3_col1" class="data row3 col1" >0.5208</td>
      <td id="T_8327a_row3_col2" class="data row3 col2" >0.5950</td>
      <td id="T_8327a_row3_col3" class="data row3 col3" >0.5200</td>
      <td id="T_8327a_row3_col4" class="data row3 col4" >0.5262</td>
      <td id="T_8327a_row3_col5" class="data row3 col5" >0.5020</td>
      <td id="T_8327a_row3_col6" class="data row3 col6" >0.0492</td>
      <td id="T_8327a_row3_col7" class="data row3 col7" >0.0440</td>
      <td id="T_8327a_row3_col8" class="data row3 col8" >0.0190</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row4" class="row_heading level0 row4" >ridge</th>
      <td id="T_8327a_row4_col0" class="data row4 col0" >Ridge Classifier</td>
      <td id="T_8327a_row4_col1" class="data row4 col1" >0.5194</td>
      <td id="T_8327a_row4_col2" class="data row4 col2" >0.6000</td>
      <td id="T_8327a_row4_col3" class="data row4 col3" >0.5850</td>
      <td id="T_8327a_row4_col4" class="data row4 col4" >0.5187</td>
      <td id="T_8327a_row4_col5" class="data row4 col5" >0.5354</td>
      <td id="T_8327a_row4_col6" class="data row4 col6" >0.0460</td>
      <td id="T_8327a_row4_col7" class="data row4 col7" >0.0064</td>
      <td id="T_8327a_row4_col8" class="data row4 col8" >0.0090</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row5" class="row_heading level0 row5" >ada</th>
      <td id="T_8327a_row5_col0" class="data row5 col0" >Ada Boost Classifier</td>
      <td id="T_8327a_row5_col1" class="data row5 col1" >0.5167</td>
      <td id="T_8327a_row5_col2" class="data row5 col2" >0.5450</td>
      <td id="T_8327a_row5_col3" class="data row5 col3" >0.5550</td>
      <td id="T_8327a_row5_col4" class="data row5 col4" >0.5333</td>
      <td id="T_8327a_row5_col5" class="data row5 col5" >0.5244</td>
      <td id="T_8327a_row5_col6" class="data row5 col6" >0.0303</td>
      <td id="T_8327a_row5_col7" class="data row5 col7" >0.0442</td>
      <td id="T_8327a_row5_col8" class="data row5 col8" >0.0130</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row6" class="row_heading level0 row6" >et</th>
      <td id="T_8327a_row6_col0" class="data row6 col0" >Extra Trees Classifier</td>
      <td id="T_8327a_row6_col1" class="data row6 col1" >0.5167</td>
      <td id="T_8327a_row6_col2" class="data row6 col2" >0.4975</td>
      <td id="T_8327a_row6_col3" class="data row6 col3" >0.5550</td>
      <td id="T_8327a_row6_col4" class="data row6 col4" >0.5567</td>
      <td id="T_8327a_row6_col5" class="data row6 col5" >0.5264</td>
      <td id="T_8327a_row6_col6" class="data row6 col6" >0.0365</td>
      <td id="T_8327a_row6_col7" class="data row6 col7" >0.0532</td>
      <td id="T_8327a_row6_col8" class="data row6 col8" >0.0190</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row7" class="row_heading level0 row7" >svm</th>
      <td id="T_8327a_row7_col0" class="data row7 col0" >SVM - Linear Kernel</td>
      <td id="T_8327a_row7_col1" class="data row7 col1" >0.4944</td>
      <td id="T_8327a_row7_col2" class="data row7 col2" >0.4450</td>
      <td id="T_8327a_row7_col3" class="data row7 col3" >0.5200</td>
      <td id="T_8327a_row7_col4" class="data row7 col4" >0.3444</td>
      <td id="T_8327a_row7_col5" class="data row7 col5" >0.3608</td>
      <td id="T_8327a_row7_col6" class="data row7 col6" >0.0182</td>
      <td id="T_8327a_row7_col7" class="data row7 col7" >0.0316</td>
      <td id="T_8327a_row7_col8" class="data row7 col8" >0.0190</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row8" class="row_heading level0 row8" >gbc</th>
      <td id="T_8327a_row8_col0" class="data row8 col0" >Gradient Boosting Classifier</td>
      <td id="T_8327a_row8_col1" class="data row8 col1" >0.4944</td>
      <td id="T_8327a_row8_col2" class="data row8 col2" >0.4638</td>
      <td id="T_8327a_row8_col3" class="data row8 col3" >0.5350</td>
      <td id="T_8327a_row8_col4" class="data row8 col4" >0.5516</td>
      <td id="T_8327a_row8_col5" class="data row8 col5" >0.4985</td>
      <td id="T_8327a_row8_col6" class="data row8 col6" >-0.0105</td>
      <td id="T_8327a_row8_col7" class="data row8 col7" >-0.0013</td>
      <td id="T_8327a_row8_col8" class="data row8 col8" >0.0130</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row9" class="row_heading level0 row9" >rf</th>
      <td id="T_8327a_row9_col0" class="data row9 col0" >Random Forest Classifier</td>
      <td id="T_8327a_row9_col1" class="data row9 col1" >0.4833</td>
      <td id="T_8327a_row9_col2" class="data row9 col2" >0.5062</td>
      <td id="T_8327a_row9_col3" class="data row9 col3" >0.4950</td>
      <td id="T_8327a_row9_col4" class="data row9 col4" >0.5469</td>
      <td id="T_8327a_row9_col5" class="data row9 col5" >0.4696</td>
      <td id="T_8327a_row9_col6" class="data row9 col6" >-0.0238</td>
      <td id="T_8327a_row9_col7" class="data row9 col7" >-0.0010</td>
      <td id="T_8327a_row9_col8" class="data row9 col8" >0.0230</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row10" class="row_heading level0 row10" >qda</th>
      <td id="T_8327a_row10_col0" class="data row10 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_8327a_row10_col1" class="data row10 col1" >0.4722</td>
      <td id="T_8327a_row10_col2" class="data row10 col2" >0.5388</td>
      <td id="T_8327a_row10_col3" class="data row10 col3" >0.5950</td>
      <td id="T_8327a_row10_col4" class="data row10 col4" >0.4298</td>
      <td id="T_8327a_row10_col5" class="data row10 col5" >0.4781</td>
      <td id="T_8327a_row10_col6" class="data row10 col6" >-0.0320</td>
      <td id="T_8327a_row10_col7" class="data row10 col7" >-0.0118</td>
      <td id="T_8327a_row10_col8" class="data row10 col8" >0.0080</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row11" class="row_heading level0 row11" >knn</th>
      <td id="T_8327a_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_8327a_row11_col1" class="data row11 col1" >0.4708</td>
      <td id="T_8327a_row11_col2" class="data row11 col2" >0.5019</td>
      <td id="T_8327a_row11_col3" class="data row11 col3" >0.5550</td>
      <td id="T_8327a_row11_col4" class="data row11 col4" >0.4405</td>
      <td id="T_8327a_row11_col5" class="data row11 col5" >0.4824</td>
      <td id="T_8327a_row11_col6" class="data row11 col6" >-0.0681</td>
      <td id="T_8327a_row11_col7" class="data row11 col7" >-0.0793</td>
      <td id="T_8327a_row11_col8" class="data row11 col8" >0.0080</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row12" class="row_heading level0 row12" >catboost</th>
      <td id="T_8327a_row12_col0" class="data row12 col0" >CatBoost Classifier</td>
      <td id="T_8327a_row12_col1" class="data row12 col1" >0.4708</td>
      <td id="T_8327a_row12_col2" class="data row12 col2" >0.4862</td>
      <td id="T_8327a_row12_col3" class="data row12 col3" >0.5600</td>
      <td id="T_8327a_row12_col4" class="data row12 col4" >0.5161</td>
      <td id="T_8327a_row12_col5" class="data row12 col5" >0.5035</td>
      <td id="T_8327a_row12_col6" class="data row12 col6" >-0.0433</td>
      <td id="T_8327a_row12_col7" class="data row12 col7" >-0.0402</td>
      <td id="T_8327a_row12_col8" class="data row12 col8" >0.0860</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row13" class="row_heading level0 row13" >lightgbm</th>
      <td id="T_8327a_row13_col0" class="data row13 col0" >Light Gradient Boosting Machine</td>
      <td id="T_8327a_row13_col1" class="data row13 col1" >0.4611</td>
      <td id="T_8327a_row13_col2" class="data row13 col2" >0.4200</td>
      <td id="T_8327a_row13_col3" class="data row13 col3" >0.5600</td>
      <td id="T_8327a_row13_col4" class="data row13 col4" >0.4633</td>
      <td id="T_8327a_row13_col5" class="data row13 col5" >0.4888</td>
      <td id="T_8327a_row13_col6" class="data row13 col6" >-0.0761</td>
      <td id="T_8327a_row13_col7" class="data row13 col7" >-0.0700</td>
      <td id="T_8327a_row13_col8" class="data row13 col8" >0.0440</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row14" class="row_heading level0 row14" >dummy</th>
      <td id="T_8327a_row14_col0" class="data row14 col0" >Dummy Classifier</td>
      <td id="T_8327a_row14_col1" class="data row14 col1" >0.4500</td>
      <td id="T_8327a_row14_col2" class="data row14 col2" >0.5000</td>
      <td id="T_8327a_row14_col3" class="data row14 col3" >0.5000</td>
      <td id="T_8327a_row14_col4" class="data row14 col4" >0.2278</td>
      <td id="T_8327a_row14_col5" class="data row14 col5" >0.3128</td>
      <td id="T_8327a_row14_col6" class="data row14 col6" >0.0000</td>
      <td id="T_8327a_row14_col7" class="data row14 col7" >0.0000</td>
      <td id="T_8327a_row14_col8" class="data row14 col8" >0.0080</td>
    </tr>
    <tr>
      <th id="T_8327a_level0_row15" class="row_heading level0 row15" >xgboost</th>
      <td id="T_8327a_row15_col0" class="data row15 col0" >Extreme Gradient Boosting</td>
      <td id="T_8327a_row15_col1" class="data row15 col1" >0.4486</td>
      <td id="T_8327a_row15_col2" class="data row15 col2" >0.4800</td>
      <td id="T_8327a_row15_col3" class="data row15 col3" >0.5300</td>
      <td id="T_8327a_row15_col4" class="data row15 col4" >0.4724</td>
      <td id="T_8327a_row15_col5" class="data row15 col5" >0.4734</td>
      <td id="T_8327a_row15_col6" class="data row15 col6" >-0.1006</td>
      <td id="T_8327a_row15_col7" class="data row15 col7" >-0.1180</td>
      <td id="T_8327a_row15_col8" class="data row15 col8" >0.0150</td>
    </tr>
  </tbody>
</table>










<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB(priors=None, var_smoothing=1e-09)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GaussianNB<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.GaussianNB.html">?<span>Documentation for GaussianNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GaussianNB(priors=None, var_smoothing=1e-09)</pre></div> </div></div></div></div>




```python

```
