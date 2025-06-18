---
title: "100% Accuracy Mushroom Classification mushroom classification"
date: 2024-01-13
last_modified_at: 2024-01-13
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "100% Accuracy Mushroom Classification mushroom classification 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/mushroom-classification")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/uciml/mushroom-classification/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# LogisticRegression, SVC, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import os

df = pd.read_csv(os.path.join(path, "mushrooms.csv"))
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>stalk-root</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>e</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>e</td>
      <td>c</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>e</td>
      <td>c</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>e</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>t</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>e</td>
      <td>k</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>e</td>
      <td>f</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>p</td>
      <td>k</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>y</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>b</td>
      <td>t</td>
      <td>?</td>
      <td>s</td>
      <td>k</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>w</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8123</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>c</td>
      <td>l</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 23 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 23 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   class                     8124 non-null   object
     1   cap-shape                 8124 non-null   object
     2   cap-surface               8124 non-null   object
     3   cap-color                 8124 non-null   object
     4   bruises                   8124 non-null   object
     5   odor                      8124 non-null   object
     6   gill-attachment           8124 non-null   object
     7   gill-spacing              8124 non-null   object
     8   gill-size                 8124 non-null   object
     9   gill-color                8124 non-null   object
     10  stalk-shape               8124 non-null   object
     11  stalk-root                8124 non-null   object
     12  stalk-surface-above-ring  8124 non-null   object
     13  stalk-surface-below-ring  8124 non-null   object
     14  stalk-color-above-ring    8124 non-null   object
     15  stalk-color-below-ring    8124 non-null   object
     16  veil-type                 8124 non-null   object
     17  veil-color                8124 non-null   object
     18  ring-number               8124 non-null   object
     19  ring-type                 8124 non-null   object
     20  spore-print-color         8124 non-null   object
     21  population                8124 non-null   object
     22  habitat                   8124 non-null   object
    dtypes: object(23)
    memory usage: 1.4+ MB



```python
col_list = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
```


```python
df.select_dtypes("object")
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>stalk-root</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>e</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>e</td>
      <td>c</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>e</td>
      <td>c</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>e</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>t</td>
      <td>e</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>e</td>
      <td>k</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>e</td>
      <td>f</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>p</td>
      <td>k</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>y</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>b</td>
      <td>t</td>
      <td>?</td>
      <td>s</td>
      <td>k</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>w</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8123</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>e</td>
      <td>?</td>
      <td>s</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>c</td>
      <td>l</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 23 columns</p>
</div>




```python
encoder = LabelEncoder()

mappings = list()

for col in df.select_dtypes("object").columns:
    df[col] = encoder.fit_transform(df[col])
    mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
    mappings.append(mappings_dict)
```


```python
X = df.drop("class", axis=1)
y = df["class"]
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
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>stalk-root</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8123</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 22 columns</p>
</div>




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
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>stalk-root</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.029712</td>
      <td>0.140128</td>
      <td>-0.198250</td>
      <td>1.185917</td>
      <td>0.881938</td>
      <td>0.162896</td>
      <td>-0.438864</td>
      <td>1.494683</td>
      <td>-0.228998</td>
      <td>-1.144806</td>
      <td>1.781460</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-0.670195</td>
      <td>-0.514389</td>
      <td>2.030028</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.029712</td>
      <td>0.140128</td>
      <td>1.765874</td>
      <td>1.185917</td>
      <td>-1.970316</td>
      <td>0.162896</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>-0.228998</td>
      <td>-1.144806</td>
      <td>0.838989</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-0.250471</td>
      <td>-1.313108</td>
      <td>-0.295730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.087047</td>
      <td>0.140128</td>
      <td>1.373049</td>
      <td>1.185917</td>
      <td>-0.544189</td>
      <td>0.162896</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>0.053477</td>
      <td>-1.144806</td>
      <td>0.838989</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-0.250471</td>
      <td>-1.313108</td>
      <td>0.867149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.029712</td>
      <td>0.953270</td>
      <td>1.373049</td>
      <td>1.185917</td>
      <td>0.881938</td>
      <td>0.162896</td>
      <td>-0.438864</td>
      <td>1.494683</td>
      <td>0.053477</td>
      <td>-1.144806</td>
      <td>1.781460</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-0.670195</td>
      <td>-0.514389</td>
      <td>2.030028</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.029712</td>
      <td>0.140128</td>
      <td>-0.591075</td>
      <td>-0.843230</td>
      <td>0.406562</td>
      <td>0.162896</td>
      <td>2.278612</td>
      <td>-0.669038</td>
      <td>-0.228998</td>
      <td>0.873511</td>
      <td>1.781460</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>-1.272216</td>
      <td>-0.250471</td>
      <td>-2.910546</td>
      <td>-0.295730</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>-0.216992</td>
      <td>0.140128</td>
      <td>-0.198250</td>
      <td>-0.843230</td>
      <td>0.406562</td>
      <td>-6.138869</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>1.748325</td>
      <td>-1.144806</td>
      <td>-1.045952</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>-0.429288</td>
      <td>-0.416681</td>
      <td>0.0</td>
      <td>-3.979055</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-1.509643</td>
      <td>-2.111827</td>
      <td>0.285710</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>1.029712</td>
      <td>0.140128</td>
      <td>-0.198250</td>
      <td>-0.843230</td>
      <td>0.406562</td>
      <td>-6.138869</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>1.748325</td>
      <td>-1.144806</td>
      <td>-1.045952</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>-0.429288</td>
      <td>-0.416681</td>
      <td>0.0</td>
      <td>-8.100146</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-1.509643</td>
      <td>0.284330</td>
      <td>0.285710</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>-0.840343</td>
      <td>0.140128</td>
      <td>-0.198250</td>
      <td>-0.843230</td>
      <td>0.406562</td>
      <td>-6.138869</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>0.053477</td>
      <td>-1.144806</td>
      <td>-1.045952</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>-0.429288</td>
      <td>-0.416681</td>
      <td>0.0</td>
      <td>-3.979055</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>-1.509643</td>
      <td>-2.111827</td>
      <td>0.285710</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>-0.216992</td>
      <td>0.953270</td>
      <td>-0.198250</td>
      <td>-0.843230</td>
      <td>1.832689</td>
      <td>0.162896</td>
      <td>-0.438864</td>
      <td>1.494683</td>
      <td>-1.358896</td>
      <td>0.873511</td>
      <td>-1.045952</td>
      <td>0.683778</td>
      <td>-0.893053</td>
      <td>0.622441</td>
      <td>0.631991</td>
      <td>0.0</td>
      <td>0.142037</td>
      <td>-0.256132</td>
      <td>-1.272216</td>
      <td>1.428426</td>
      <td>0.284330</td>
      <td>0.285710</td>
    </tr>
    <tr>
      <th>8123</th>
      <td>1.029712</td>
      <td>0.140128</td>
      <td>-0.198250</td>
      <td>-0.843230</td>
      <td>0.406562</td>
      <td>-6.138869</td>
      <td>-0.438864</td>
      <td>-0.669038</td>
      <td>1.748325</td>
      <td>-1.144806</td>
      <td>-1.045952</td>
      <td>0.683778</td>
      <td>0.586385</td>
      <td>-0.429288</td>
      <td>-0.416681</td>
      <td>0.0</td>
      <td>-3.979055</td>
      <td>-0.256132</td>
      <td>0.948081</td>
      <td>0.169254</td>
      <td>-2.111827</td>
      <td>0.285710</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 22 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
```


```python
np.sum(y) / len(y)  # quite balaced so using accuracy is fine
```




    0.48202855736090594




```python
log_model = LogisticRegression()
svc_model = SVC(C=1.0, kernel="rbf")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 128))

log_model.fit(X_train, y_train)
svc_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)


print(log_model.score(X_test, y_test))
print(svc_model.score(X_test, y_test))
print(nn_model.score(X_test, y_test))
```

    0.9495384615384616
    1.0
    1.0



```python
X_test.shape
```




    (1625, 22)




```python
corr = df.corr()
```


```python
sns.heatmap(corr)
```




    <Axes: >




    
![png](013_100%25_Accuracy_Mushroom_Classification_mushroom_classification_files/013_100%25_Accuracy_Mushroom_Classification_mushroom_classification_16_1.png)
    



```python
from pycaret.classification import *

setup(df, target=df["class"], train_size=0.8, session_id=42)
```


<style type="text/css">
#T_f4070_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_f4070">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f4070_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_f4070_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f4070_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f4070_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_f4070_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f4070_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_f4070_row1_col1" class="data row1 col1" >class_y</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f4070_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_f4070_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f4070_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_f4070_row3_col1" class="data row3 col1" >(8124, 24)</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f4070_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_f4070_row4_col1" class="data row4 col1" >(8124, 24)</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f4070_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_f4070_row5_col1" class="data row5 col1" >(6499, 24)</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f4070_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_f4070_row6_col1" class="data row6 col1" >(1625, 24)</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f4070_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_f4070_row7_col1" class="data row7 col1" >23</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f4070_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_f4070_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f4070_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_f4070_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_f4070_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_f4070_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_f4070_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_f4070_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_f4070_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_f4070_row12_col1" class="data row12 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_f4070_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_f4070_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_f4070_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_f4070_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_f4070_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_f4070_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_f4070_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_f4070_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_f4070_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_f4070_row17_col1" class="data row17 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_f4070_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_f4070_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_f4070_row18_col1" class="data row18 col1" >81de</td>
    </tr>
  </tbody>
</table>






    <pycaret.classification.oop.ClassificationExperiment at 0x316e349a0>




```python
compare_models()
```






<style type="text/css">
#T_42355 th {
  text-align: left;
}
#T_42355_row0_col0, #T_42355_row1_col0, #T_42355_row2_col0, #T_42355_row3_col0, #T_42355_row4_col0, #T_42355_row5_col0, #T_42355_row6_col0, #T_42355_row7_col0, #T_42355_row8_col0, #T_42355_row9_col0, #T_42355_row10_col0, #T_42355_row11_col0, #T_42355_row12_col0, #T_42355_row12_col1, #T_42355_row12_col3, #T_42355_row12_col4, #T_42355_row12_col5, #T_42355_row12_col6, #T_42355_row12_col7, #T_42355_row13_col0, #T_42355_row13_col1, #T_42355_row13_col2, #T_42355_row13_col3, #T_42355_row13_col4, #T_42355_row13_col5, #T_42355_row13_col6, #T_42355_row13_col7, #T_42355_row14_col0, #T_42355_row14_col1, #T_42355_row14_col2, #T_42355_row14_col3, #T_42355_row14_col4, #T_42355_row14_col5, #T_42355_row14_col6, #T_42355_row14_col7, #T_42355_row15_col0, #T_42355_row15_col1, #T_42355_row15_col2, #T_42355_row15_col3, #T_42355_row15_col4, #T_42355_row15_col5, #T_42355_row15_col6, #T_42355_row15_col7 {
  text-align: left;
}
#T_42355_row0_col1, #T_42355_row0_col2, #T_42355_row0_col3, #T_42355_row0_col4, #T_42355_row0_col5, #T_42355_row0_col6, #T_42355_row0_col7, #T_42355_row1_col1, #T_42355_row1_col2, #T_42355_row1_col3, #T_42355_row1_col4, #T_42355_row1_col5, #T_42355_row1_col6, #T_42355_row1_col7, #T_42355_row2_col1, #T_42355_row2_col2, #T_42355_row2_col3, #T_42355_row2_col4, #T_42355_row2_col5, #T_42355_row2_col6, #T_42355_row2_col7, #T_42355_row3_col1, #T_42355_row3_col2, #T_42355_row3_col3, #T_42355_row3_col4, #T_42355_row3_col5, #T_42355_row3_col6, #T_42355_row3_col7, #T_42355_row4_col1, #T_42355_row4_col2, #T_42355_row4_col3, #T_42355_row4_col4, #T_42355_row4_col5, #T_42355_row4_col6, #T_42355_row4_col7, #T_42355_row5_col1, #T_42355_row5_col2, #T_42355_row5_col3, #T_42355_row5_col4, #T_42355_row5_col5, #T_42355_row5_col6, #T_42355_row5_col7, #T_42355_row6_col1, #T_42355_row6_col2, #T_42355_row6_col3, #T_42355_row6_col4, #T_42355_row6_col5, #T_42355_row6_col6, #T_42355_row6_col7, #T_42355_row7_col1, #T_42355_row7_col2, #T_42355_row7_col3, #T_42355_row7_col4, #T_42355_row7_col5, #T_42355_row7_col6, #T_42355_row7_col7, #T_42355_row8_col1, #T_42355_row8_col2, #T_42355_row8_col3, #T_42355_row8_col4, #T_42355_row8_col5, #T_42355_row8_col6, #T_42355_row8_col7, #T_42355_row9_col1, #T_42355_row9_col2, #T_42355_row9_col3, #T_42355_row9_col4, #T_42355_row9_col5, #T_42355_row9_col6, #T_42355_row9_col7, #T_42355_row10_col1, #T_42355_row10_col2, #T_42355_row10_col3, #T_42355_row10_col4, #T_42355_row10_col5, #T_42355_row10_col6, #T_42355_row10_col7, #T_42355_row11_col1, #T_42355_row11_col2, #T_42355_row11_col3, #T_42355_row11_col4, #T_42355_row11_col5, #T_42355_row11_col6, #T_42355_row11_col7, #T_42355_row12_col2 {
  text-align: left;
  background-color: yellow;
}
#T_42355_row0_col8, #T_42355_row1_col8, #T_42355_row2_col8, #T_42355_row3_col8, #T_42355_row5_col8, #T_42355_row6_col8, #T_42355_row7_col8, #T_42355_row8_col8, #T_42355_row9_col8, #T_42355_row10_col8, #T_42355_row11_col8, #T_42355_row12_col8, #T_42355_row13_col8, #T_42355_row14_col8, #T_42355_row15_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_42355_row4_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_42355">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_42355_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_42355_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_42355_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_42355_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_42355_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_42355_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_42355_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_42355_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_42355_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_42355_level0_row0" class="row_heading level0 row0" >lr</th>
      <td id="T_42355_row0_col0" class="data row0 col0" >Logistic Regression</td>
      <td id="T_42355_row0_col1" class="data row0 col1" >1.0000</td>
      <td id="T_42355_row0_col2" class="data row0 col2" >1.0000</td>
      <td id="T_42355_row0_col3" class="data row0 col3" >1.0000</td>
      <td id="T_42355_row0_col4" class="data row0 col4" >1.0000</td>
      <td id="T_42355_row0_col5" class="data row0 col5" >1.0000</td>
      <td id="T_42355_row0_col6" class="data row0 col6" >1.0000</td>
      <td id="T_42355_row0_col7" class="data row0 col7" >1.0000</td>
      <td id="T_42355_row0_col8" class="data row0 col8" >0.2350</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row1" class="row_heading level0 row1" >nb</th>
      <td id="T_42355_row1_col0" class="data row1 col0" >Naive Bayes</td>
      <td id="T_42355_row1_col1" class="data row1 col1" >1.0000</td>
      <td id="T_42355_row1_col2" class="data row1 col2" >1.0000</td>
      <td id="T_42355_row1_col3" class="data row1 col3" >1.0000</td>
      <td id="T_42355_row1_col4" class="data row1 col4" >1.0000</td>
      <td id="T_42355_row1_col5" class="data row1 col5" >1.0000</td>
      <td id="T_42355_row1_col6" class="data row1 col6" >1.0000</td>
      <td id="T_42355_row1_col7" class="data row1 col7" >1.0000</td>
      <td id="T_42355_row1_col8" class="data row1 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row2" class="row_heading level0 row2" >dt</th>
      <td id="T_42355_row2_col0" class="data row2 col0" >Decision Tree Classifier</td>
      <td id="T_42355_row2_col1" class="data row2 col1" >1.0000</td>
      <td id="T_42355_row2_col2" class="data row2 col2" >1.0000</td>
      <td id="T_42355_row2_col3" class="data row2 col3" >1.0000</td>
      <td id="T_42355_row2_col4" class="data row2 col4" >1.0000</td>
      <td id="T_42355_row2_col5" class="data row2 col5" >1.0000</td>
      <td id="T_42355_row2_col6" class="data row2 col6" >1.0000</td>
      <td id="T_42355_row2_col7" class="data row2 col7" >1.0000</td>
      <td id="T_42355_row2_col8" class="data row2 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row3" class="row_heading level0 row3" >svm</th>
      <td id="T_42355_row3_col0" class="data row3 col0" >SVM - Linear Kernel</td>
      <td id="T_42355_row3_col1" class="data row3 col1" >1.0000</td>
      <td id="T_42355_row3_col2" class="data row3 col2" >1.0000</td>
      <td id="T_42355_row3_col3" class="data row3 col3" >1.0000</td>
      <td id="T_42355_row3_col4" class="data row3 col4" >1.0000</td>
      <td id="T_42355_row3_col5" class="data row3 col5" >1.0000</td>
      <td id="T_42355_row3_col6" class="data row3 col6" >1.0000</td>
      <td id="T_42355_row3_col7" class="data row3 col7" >1.0000</td>
      <td id="T_42355_row3_col8" class="data row3 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row4" class="row_heading level0 row4" >ridge</th>
      <td id="T_42355_row4_col0" class="data row4 col0" >Ridge Classifier</td>
      <td id="T_42355_row4_col1" class="data row4 col1" >1.0000</td>
      <td id="T_42355_row4_col2" class="data row4 col2" >1.0000</td>
      <td id="T_42355_row4_col3" class="data row4 col3" >1.0000</td>
      <td id="T_42355_row4_col4" class="data row4 col4" >1.0000</td>
      <td id="T_42355_row4_col5" class="data row4 col5" >1.0000</td>
      <td id="T_42355_row4_col6" class="data row4 col6" >1.0000</td>
      <td id="T_42355_row4_col7" class="data row4 col7" >1.0000</td>
      <td id="T_42355_row4_col8" class="data row4 col8" >0.0050</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row5" class="row_heading level0 row5" >rf</th>
      <td id="T_42355_row5_col0" class="data row5 col0" >Random Forest Classifier</td>
      <td id="T_42355_row5_col1" class="data row5 col1" >1.0000</td>
      <td id="T_42355_row5_col2" class="data row5 col2" >1.0000</td>
      <td id="T_42355_row5_col3" class="data row5 col3" >1.0000</td>
      <td id="T_42355_row5_col4" class="data row5 col4" >1.0000</td>
      <td id="T_42355_row5_col5" class="data row5 col5" >1.0000</td>
      <td id="T_42355_row5_col6" class="data row5 col6" >1.0000</td>
      <td id="T_42355_row5_col7" class="data row5 col7" >1.0000</td>
      <td id="T_42355_row5_col8" class="data row5 col8" >0.0370</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row6" class="row_heading level0 row6" >ada</th>
      <td id="T_42355_row6_col0" class="data row6 col0" >Ada Boost Classifier</td>
      <td id="T_42355_row6_col1" class="data row6 col1" >1.0000</td>
      <td id="T_42355_row6_col2" class="data row6 col2" >1.0000</td>
      <td id="T_42355_row6_col3" class="data row6 col3" >1.0000</td>
      <td id="T_42355_row6_col4" class="data row6 col4" >1.0000</td>
      <td id="T_42355_row6_col5" class="data row6 col5" >1.0000</td>
      <td id="T_42355_row6_col6" class="data row6 col6" >1.0000</td>
      <td id="T_42355_row6_col7" class="data row6 col7" >1.0000</td>
      <td id="T_42355_row6_col8" class="data row6 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row7" class="row_heading level0 row7" >gbc</th>
      <td id="T_42355_row7_col0" class="data row7 col0" >Gradient Boosting Classifier</td>
      <td id="T_42355_row7_col1" class="data row7 col1" >1.0000</td>
      <td id="T_42355_row7_col2" class="data row7 col2" >1.0000</td>
      <td id="T_42355_row7_col3" class="data row7 col3" >1.0000</td>
      <td id="T_42355_row7_col4" class="data row7 col4" >1.0000</td>
      <td id="T_42355_row7_col5" class="data row7 col5" >1.0000</td>
      <td id="T_42355_row7_col6" class="data row7 col6" >1.0000</td>
      <td id="T_42355_row7_col7" class="data row7 col7" >1.0000</td>
      <td id="T_42355_row7_col8" class="data row7 col8" >0.0320</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row8" class="row_heading level0 row8" >et</th>
      <td id="T_42355_row8_col0" class="data row8 col0" >Extra Trees Classifier</td>
      <td id="T_42355_row8_col1" class="data row8 col1" >1.0000</td>
      <td id="T_42355_row8_col2" class="data row8 col2" >1.0000</td>
      <td id="T_42355_row8_col3" class="data row8 col3" >1.0000</td>
      <td id="T_42355_row8_col4" class="data row8 col4" >1.0000</td>
      <td id="T_42355_row8_col5" class="data row8 col5" >1.0000</td>
      <td id="T_42355_row8_col6" class="data row8 col6" >1.0000</td>
      <td id="T_42355_row8_col7" class="data row8 col7" >1.0000</td>
      <td id="T_42355_row8_col8" class="data row8 col8" >0.0300</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row9" class="row_heading level0 row9" >xgboost</th>
      <td id="T_42355_row9_col0" class="data row9 col0" >Extreme Gradient Boosting</td>
      <td id="T_42355_row9_col1" class="data row9 col1" >1.0000</td>
      <td id="T_42355_row9_col2" class="data row9 col2" >1.0000</td>
      <td id="T_42355_row9_col3" class="data row9 col3" >1.0000</td>
      <td id="T_42355_row9_col4" class="data row9 col4" >1.0000</td>
      <td id="T_42355_row9_col5" class="data row9 col5" >1.0000</td>
      <td id="T_42355_row9_col6" class="data row9 col6" >1.0000</td>
      <td id="T_42355_row9_col7" class="data row9 col7" >1.0000</td>
      <td id="T_42355_row9_col8" class="data row9 col8" >0.0120</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row10" class="row_heading level0 row10" >lightgbm</th>
      <td id="T_42355_row10_col0" class="data row10 col0" >Light Gradient Boosting Machine</td>
      <td id="T_42355_row10_col1" class="data row10 col1" >1.0000</td>
      <td id="T_42355_row10_col2" class="data row10 col2" >1.0000</td>
      <td id="T_42355_row10_col3" class="data row10 col3" >1.0000</td>
      <td id="T_42355_row10_col4" class="data row10 col4" >1.0000</td>
      <td id="T_42355_row10_col5" class="data row10 col5" >1.0000</td>
      <td id="T_42355_row10_col6" class="data row10 col6" >1.0000</td>
      <td id="T_42355_row10_col7" class="data row10 col7" >1.0000</td>
      <td id="T_42355_row10_col8" class="data row10 col8" >0.1040</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row11" class="row_heading level0 row11" >catboost</th>
      <td id="T_42355_row11_col0" class="data row11 col0" >CatBoost Classifier</td>
      <td id="T_42355_row11_col1" class="data row11 col1" >1.0000</td>
      <td id="T_42355_row11_col2" class="data row11 col2" >1.0000</td>
      <td id="T_42355_row11_col3" class="data row11 col3" >1.0000</td>
      <td id="T_42355_row11_col4" class="data row11 col4" >1.0000</td>
      <td id="T_42355_row11_col5" class="data row11 col5" >1.0000</td>
      <td id="T_42355_row11_col6" class="data row11 col6" >1.0000</td>
      <td id="T_42355_row11_col7" class="data row11 col7" >1.0000</td>
      <td id="T_42355_row11_col8" class="data row11 col8" >0.3640</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row12" class="row_heading level0 row12" >knn</th>
      <td id="T_42355_row12_col0" class="data row12 col0" >K Neighbors Classifier</td>
      <td id="T_42355_row12_col1" class="data row12 col1" >0.9992</td>
      <td id="T_42355_row12_col2" class="data row12 col2" >1.0000</td>
      <td id="T_42355_row12_col3" class="data row12 col3" >0.9987</td>
      <td id="T_42355_row12_col4" class="data row12 col4" >0.9997</td>
      <td id="T_42355_row12_col5" class="data row12 col5" >0.9992</td>
      <td id="T_42355_row12_col6" class="data row12 col6" >0.9985</td>
      <td id="T_42355_row12_col7" class="data row12 col7" >0.9985</td>
      <td id="T_42355_row12_col8" class="data row12 col8" >0.0130</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row13" class="row_heading level0 row13" >lda</th>
      <td id="T_42355_row13_col0" class="data row13 col0" >Linear Discriminant Analysis</td>
      <td id="T_42355_row13_col1" class="data row13 col1" >0.9408</td>
      <td id="T_42355_row13_col2" class="data row13 col2" >0.9642</td>
      <td id="T_42355_row13_col3" class="data row13 col3" >0.9212</td>
      <td id="T_42355_row13_col4" class="data row13 col4" >0.9545</td>
      <td id="T_42355_row13_col5" class="data row13 col5" >0.9375</td>
      <td id="T_42355_row13_col6" class="data row13 col6" >0.8812</td>
      <td id="T_42355_row13_col7" class="data row13 col7" >0.8819</td>
      <td id="T_42355_row13_col8" class="data row13 col8" >0.0070</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row14" class="row_heading level0 row14" >qda</th>
      <td id="T_42355_row14_col0" class="data row14 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_42355_row14_col1" class="data row14 col1" >0.5179</td>
      <td id="T_42355_row14_col2" class="data row14 col2" >0.0000</td>
      <td id="T_42355_row14_col3" class="data row14 col3" >0.0000</td>
      <td id="T_42355_row14_col4" class="data row14 col4" >0.0000</td>
      <td id="T_42355_row14_col5" class="data row14 col5" >0.0000</td>
      <td id="T_42355_row14_col6" class="data row14 col6" >0.0000</td>
      <td id="T_42355_row14_col7" class="data row14 col7" >0.0000</td>
      <td id="T_42355_row14_col8" class="data row14 col8" >0.0080</td>
    </tr>
    <tr>
      <th id="T_42355_level0_row15" class="row_heading level0 row15" >dummy</th>
      <td id="T_42355_row15_col0" class="data row15 col0" >Dummy Classifier</td>
      <td id="T_42355_row15_col1" class="data row15 col1" >0.5179</td>
      <td id="T_42355_row15_col2" class="data row15 col2" >0.5000</td>
      <td id="T_42355_row15_col3" class="data row15 col3" >0.0000</td>
      <td id="T_42355_row15_col4" class="data row15 col4" >0.0000</td>
      <td id="T_42355_row15_col5" class="data row15 col5" >0.0000</td>
      <td id="T_42355_row15_col6" class="data row15 col6" >0.0000</td>
      <td id="T_42355_row15_col7" class="data row15 col7" >0.0000</td>
      <td id="T_42355_row15_col8" class="data row15 col8" >0.0060</td>
    </tr>
  </tbody>
</table>










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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class=&#x27;deprecated&#x27;, n_jobs=None, penalty=&#x27;l2&#x27;,
                   random_state=42, solver=&#x27;lbfgs&#x27;, tol=0.0001, verbose=0,
                   warm_start=False)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class=&#x27;deprecated&#x27;, n_jobs=None, penalty=&#x27;l2&#x27;,
                   random_state=42, solver=&#x27;lbfgs&#x27;, tol=0.0001, verbose=0,
                   warm_start=False)</pre></div> </div></div></div></div>




```python

```
