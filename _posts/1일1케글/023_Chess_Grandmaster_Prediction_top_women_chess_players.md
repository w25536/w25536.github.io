---
title: "023_Chess_Grandmaster_Prediction_top_women_chess_players"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "023_Chess_Grandmaster_Prediction_top_women_chess_players"
use_math: true
classes: wide
---

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikasojha98/top-women-chess-players")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/vikasojha98/top-women-chess-players/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


df = pd.read_csv(os.path.join(path, "top_women_chess_players_aug_2020.csv"))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8553 entries, 0 to 8552
    Data columns (total 10 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Fide id          8553 non-null   int64  
     1   Name             8553 non-null   object 
     2   Federation       8553 non-null   object 
     3   Gender           8553 non-null   object 
     4   Year_of_birth    8261 non-null   float64
     5   Title            3118 non-null   object 
     6   Standard_Rating  8553 non-null   int64  
     7   Rapid_rating     3608 non-null   float64
     8   Blitz_rating     3472 non-null   float64
     9   Inactive_flag    5852 non-null   object 
    dtypes: float64(3), int64(2), object(5)
    memory usage: 668.3+ KB



```python
encoder = LabelEncoder()


def preprocess_df(df):
    df = df.drop(["Fide id", "Name", "Gender", "Federation"], axis=1).copy()

    numeric_df = df.select_dtypes(np.number)
    numeric_cols = numeric_df.columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    df["Inactive_flag"] = df["Inactive_flag"].fillna("wa").copy()

    title_dummies = pd.get_dummies(df["Title"], dtype=int).copy()

    df = pd.concat([df, title_dummies["GM"]], axis=1)
    df = df.drop(["Title"], axis=1)

    df["Inactive_flag"] = encoder.fit_transform(df["Inactive_flag"])
    mappings = {label: index for index, label in enumerate(encoder.classes_)}

    print(mappings)

    y = df["GM"].copy()
    X = df.drop(["GM"], axis=1).copy()

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    X = pd.DataFrame(scaled_X)

    return X, y
```


```python
X, y = preprocess_df(df)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

model = LogisticRegression()
```

    {'wa': 0, 'wi': 1}



```python
from pycaret.classification import *

df = pd.concat([X, y], axis=1)

setup
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.622222</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.822222</td>
      <td>0.980549</td>
      <td>0.982419</td>
      <td>0.914394</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.744444</td>
      <td>0.898169</td>
      <td>0.885373</td>
      <td>0.839569</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.866667</td>
      <td>0.893593</td>
      <td>0.898734</td>
      <td>0.812936</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.544444</td>
      <td>0.887872</td>
      <td>0.497665</td>
      <td>0.485831</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8548</th>
      <td>0.725464</td>
      <td>0.000000</td>
      <td>0.497665</td>
      <td>0.485831</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8549</th>
      <td>0.822222</td>
      <td>0.000000</td>
      <td>0.398734</td>
      <td>0.384274</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8550</th>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.464135</td>
      <td>0.421687</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8551</th>
      <td>0.900000</td>
      <td>0.000000</td>
      <td>0.497665</td>
      <td>0.485831</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8552</th>
      <td>0.688889</td>
      <td>0.000000</td>
      <td>0.497665</td>
      <td>0.485831</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8553 rows × 5 columns</p>
</div>




```python
df.isnull().sum()
```




    Fide id               0
    Name                  0
    Federation            0
    Gender                0
    Year_of_birth       292
    Title              5435
    Standard_Rating       0
    Rapid_rating       4945
    Blitz_rating       5081
    Inactive_flag      2701
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8553 entries, 0 to 8552
    Data columns (total 10 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Fide id          8553 non-null   int64  
     1   Name             8553 non-null   object 
     2   Federation       8553 non-null   object 
     3   Gender           8553 non-null   object 
     4   Year_of_birth    8261 non-null   float64
     5   Title            3118 non-null   object 
     6   Standard_Rating  8553 non-null   int64  
     7   Rapid_rating     3608 non-null   float64
     8   Blitz_rating     3472 non-null   float64
     9   Inactive_flag    5852 non-null   object 
    dtypes: float64(3), int64(2), object(5)
    memory usage: 668.3+ KB



```python
df["Inactive_flag"].unique()
```




    array(['wi', nan], dtype=object)




```python
df["Federation"].unique()
```




    array(['HUN', 'CHN', 'IND', 'RUS', 'UKR', 'LTU', 'GEO', 'KAZ', 'IRI',
           'GER', 'SWE', 'BUL', 'TUR', 'GRE', 'AZE', 'FRA', 'ROU', 'USA',
           'MGL', 'POL', 'BLR', 'QAT', 'ESP', 'ENG', 'INA', 'ARM', 'CZE',
           'PER', 'SRB', 'NED', 'SCO', 'UZB', 'ITA', 'CUB', 'VIE', 'ECU',
           'AUS', 'ARG', 'CRO', 'SVK', 'SGP', 'ISR', 'LUX', 'SLO', 'EST',
           'CAN', 'LAT', 'AUT', 'SUI', 'MNC', 'MDA', 'BRA', 'BEL', 'COL',
           'PHI', 'PAR', 'BRU', 'MEX', 'BIH', 'MAS', 'NOR', 'MNE', 'TKM',
           'IRL', 'VEN', 'EGY', 'IRQ', 'FIN', 'BOL', 'DEN', 'MKD', 'KGZ',
           'ESA', 'CHI', 'RSA', 'FID', 'UAE', 'LBN', 'MYA', 'ISL', 'BAN',
           'POR', 'KSA', 'NAM', 'URU', 'ALG', 'WLS', 'PUR', 'ALB', 'KOR',
           'TJK', 'SRI', 'JAM', 'ANG', 'NGR', 'BAR', 'BER', 'ZIM', 'BOT',
           'JPN', 'DOM', 'CRC', 'SYR', 'GUA', 'SEY', 'JOR', 'NZL', 'MAR',
           'MAC', 'TTO', 'NCA', 'ZAM', 'PAN', 'THA', 'GCI', 'AHO', 'HKG',
           'MLT', 'HON', 'LBA', 'SUR', 'UGA', 'CPV', 'MAD'], dtype=object)




```python

```
