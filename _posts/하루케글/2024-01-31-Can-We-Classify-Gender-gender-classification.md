---
title: "Can We Classify Gender gender classification"
date: 2024-01-31
last_modified_at: 2024-01-31
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Can We Classify Gender gender classification 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("hb20007/gender-classification")

print("Path to dataset files:", path)
```

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.4)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/hb20007/gender-classification/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf

import os
```


```python
df = pd.read_csv(os.path.join(path, "Transformed Data Set - Sheet1.csv"))
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
      <th>Favorite Color</th>
      <th>Favorite Music Genre</th>
      <th>Favorite Beverage</th>
      <th>Favorite Soft Drink</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cool</td>
      <td>Rock</td>
      <td>Vodka</td>
      <td>7UP/Sprite</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Neutral</td>
      <td>Hip hop</td>
      <td>Vodka</td>
      <td>Coca Cola/Pepsi</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Warm</td>
      <td>Rock</td>
      <td>Wine</td>
      <td>Coca Cola/Pepsi</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Warm</td>
      <td>Folk/Traditional</td>
      <td>Whiskey</td>
      <td>Fanta</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cool</td>
      <td>Rock</td>
      <td>Vodka</td>
      <td>Coca Cola/Pepsi</td>
      <td>F</td>
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
      <th>61</th>
      <td>Cool</td>
      <td>Rock</td>
      <td>Vodka</td>
      <td>Coca Cola/Pepsi</td>
      <td>M</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Cool</td>
      <td>Hip hop</td>
      <td>Beer</td>
      <td>Coca Cola/Pepsi</td>
      <td>M</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Neutral</td>
      <td>Hip hop</td>
      <td>Doesn't drink</td>
      <td>Fanta</td>
      <td>M</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Cool</td>
      <td>Rock</td>
      <td>Wine</td>
      <td>Coca Cola/Pepsi</td>
      <td>M</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Cool</td>
      <td>Electronic</td>
      <td>Beer</td>
      <td>Coca Cola/Pepsi</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 5 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 66 entries, 0 to 65
    Data columns (total 5 columns):
     #   Column                Non-Null Count  Dtype 
    ---  ------                --------------  ----- 
     0   Favorite Color        66 non-null     object
     1   Favorite Music Genre  66 non-null     object
     2   Favorite Beverage     66 non-null     object
     3   Favorite Soft Drink   66 non-null     object
     4   Gender                66 non-null     object
    dtypes: object(5)
    memory usage: 2.7+ KB



```python
df.isna().sum()
```




    Favorite Color          0
    Favorite Music Genre    0
    Favorite Beverage       0
    Favorite Soft Drink     0
    Gender                  0
    dtype: int64




```python
color_ordering = list(df["Favorite Color"].unique())
```


```python
{column: list(df[column].unique()) for column in df.columns}
```




    {'Favorite Color': ['Cool', 'Neutral', 'Warm'],
     'Favorite Music Genre': ['Rock',
      'Hip hop',
      'Folk/Traditional',
      'Jazz/Blues',
      'Pop',
      'Electronic',
      'R&B and soul'],
     'Favorite Beverage': ['Vodka',
      'Wine',
      'Whiskey',
      "Doesn't drink",
      'Beer',
      'Other'],
     'Favorite Soft Drink': ['7UP/Sprite', 'Coca Cola/Pepsi', 'Fanta', 'Other'],
     'Gender': ['F', 'M']}




```python
def add_prefixes(df, column, prefix):
    return df[column].apply(lambda x: prefix + x)
```


```python
df["Favorite Soft Drink"] = add_prefixes(df, "Favorite Soft Drink", "s_")
df["Favorite Beverage"] = add_prefixes(df, "Favorite Beverage", "b_")
```


```python
pd.get_dummies(df["Favorite Color"], dtype=int)
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
      <th>Cool</th>
      <th>Neutral</th>
      <th>Warm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 3 columns</p>
</div>




```python
def onehot_encode(df, columns):
    for column in columns:
        dummies = pd.get_dummies(df[column], dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([column], axis=1)
    return df
```


```python
df = onehot_encode(
    df, ["Favorite Music Genre", "Favorite Beverage", "Favorite Soft Drink"]
)
```


```python
df["Favorite Color"] = df["Favorite Color"].apply(lambda x: color_ordering.index(x))
```


```python
df["Favorite Color"].unique()
```




    array([0, 1, 2])




```python
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
gender_mappings = {index: value for index, value in enumerate(label_encoder.classes_)}
```


```python
gender_mappings
```




    {0: 'F', 1: 'M'}




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
      <th>Favorite Color</th>
      <th>Gender</th>
      <th>Electronic</th>
      <th>Folk/Traditional</th>
      <th>Hip hop</th>
      <th>Jazz/Blues</th>
      <th>Pop</th>
      <th>R&amp;B and soul</th>
      <th>Rock</th>
      <th>b_Beer</th>
      <th>b_Doesn't drink</th>
      <th>b_Other</th>
      <th>b_Vodka</th>
      <th>b_Whiskey</th>
      <th>b_Wine</th>
      <th>s_7UP/Sprite</th>
      <th>s_Coca Cola/Pepsi</th>
      <th>s_Fanta</th>
      <th>s_Other</th>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>61</th>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 19 columns</p>
</div>




```python
y = df["Gender"]
X = df.drop(["Gender"], axis=1)
```


```python
scaler = MinMaxScaler()
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
      <th>Favorite Color</th>
      <th>Electronic</th>
      <th>Folk/Traditional</th>
      <th>Hip hop</th>
      <th>Jazz/Blues</th>
      <th>Pop</th>
      <th>R&amp;B and soul</th>
      <th>Rock</th>
      <th>b_Beer</th>
      <th>b_Doesn't drink</th>
      <th>b_Other</th>
      <th>b_Vodka</th>
      <th>b_Whiskey</th>
      <th>b_Wine</th>
      <th>s_7UP/Sprite</th>
      <th>s_Coca Cola/Pepsi</th>
      <th>s_Fanta</th>
      <th>s_Other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
    </tr>
    <tr>
      <th>61</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 18 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
```


```python
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, vmin=1, vmax=1)
plt.show()
```


    
![png](031_Can_We_Classify_Gender_gender_classification_files/031_Can_We_Classify_Gender_gender_classification_22_0.png)
    



```python
inputs = tf.keras.Input(shape=(18,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],  # no skew so it's fine
)

batch_size = 32
epochs = 22

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
)
```


```python
# 1. Add regularization and dropout to prevent overfitting
inputs = tf.keras.Input(shape=(18,))
x = tf.keras.layers.Dense(
    128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
)(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(
    64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
)(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 2. Modify compilation with learning rate and better metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()],
)

# 3. Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

# 4. Adjust training parameters
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=16,  # Smaller batch size
    epochs=29,  # More epochs (EarlyStopping will prevent overfitting)
    callbacks=callbacks,
    verbose=0,
)
```


```python
# plt.figure(figsize=(10,10))
# epochs_range = range(1, epochs+1)
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.plot(epochs_range, train_loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')

# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

plt.figure(figsize=(10, 10))
# Use the actual number of epochs from the history
epochs_range = range(1, len(history.history["loss"]) + 1)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```


    
![png](031_Can_We_Classify_Gender_gender_classification_files/031_Can_We_Classify_Gender_gender_classification_25_0.png)
    



```python
np.argmin(val_loss)
```




    28




```python
model.evaluate(X_test, y_test)
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - accuracy: 0.3500 - auc_2: 0.3242 - loss: 1.4164





    [1.416383981704712, 0.3499999940395355, 0.32417580485343933]




```python

```
