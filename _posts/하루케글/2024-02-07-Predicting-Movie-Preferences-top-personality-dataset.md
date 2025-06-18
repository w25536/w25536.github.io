---
title: "Predicting Movie Preferences top personality dataset"
date: 2024-02-07
last_modified_at: 2024-02-07
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Predicting Movie Preferences top personality dataset 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("arslanali4343/top-personality-dataset")

print("Path to dataset files:", path)
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
      warnings.warn(


    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.6)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/arslanali4343/top-personality-dataset/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv(os.path.join(path, "2018-personality-data.csv"))
```


```python
df.head()
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
      <th>userid</th>
      <th>openness</th>
      <th>agreeableness</th>
      <th>emotional_stability</th>
      <th>conscientiousness</th>
      <th>extraversion</th>
      <th>assigned metric</th>
      <th>assigned condition</th>
      <th>movie_1</th>
      <th>predicted_rating_1</th>
      <th>...</th>
      <th>movie_9</th>
      <th>predicted_rating_9</th>
      <th>movie_10</th>
      <th>predicted_rating_10</th>
      <th>movie_11</th>
      <th>predicted_rating_11</th>
      <th>movie_12</th>
      <th>predicted_rating_12</th>
      <th>is_personalized</th>
      <th>enjoy_watching</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8e7cebf9a234c064b75016249f2ac65e</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.5</td>
      <td>6.5</td>
      <td>serendipity</td>
      <td>high</td>
      <td>77658</td>
      <td>4.410466</td>
      <td>...</td>
      <td>120138</td>
      <td>4.244817</td>
      <td>121372</td>
      <td>4.396004</td>
      <td>127152</td>
      <td>4.120456</td>
      <td>95311</td>
      <td>4.053847</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77c7d756a093150d4377720abeaeef76</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>5.5</td>
      <td>4.0</td>
      <td>all</td>
      <td>default</td>
      <td>94959</td>
      <td>4.207280</td>
      <td>...</td>
      <td>56782</td>
      <td>4.019599</td>
      <td>5618</td>
      <td>3.963953</td>
      <td>969</td>
      <td>4.174188</td>
      <td>1232</td>
      <td>4.334877</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b7e8a92987a530cc368719a0e60e26a3</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>serendipity</td>
      <td>medium</td>
      <td>110501</td>
      <td>4.868064</td>
      <td>...</td>
      <td>2288</td>
      <td>4.823212</td>
      <td>3307</td>
      <td>4.676756</td>
      <td>1172</td>
      <td>4.649281</td>
      <td>1212</td>
      <td>4.744990</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92561f21446e017dd6b68b94b23ad5b7</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>popularity</td>
      <td>medium</td>
      <td>2905</td>
      <td>4.526371</td>
      <td>...</td>
      <td>3030</td>
      <td>4.425689</td>
      <td>1281</td>
      <td>4.479921</td>
      <td>940</td>
      <td>4.355061</td>
      <td>905</td>
      <td>4.317927</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>030001ac2145a938b07e686a35a2d638</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>2.5</td>
      <td>popularity</td>
      <td>medium</td>
      <td>2905</td>
      <td>4.526371</td>
      <td>...</td>
      <td>3030</td>
      <td>4.425689</td>
      <td>1281</td>
      <td>4.479921</td>
      <td>940</td>
      <td>4.355061</td>
      <td>905</td>
      <td>4.317927</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1834 entries, 0 to 1833
    Data columns (total 9 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0    openness             1834 non-null   float64
     1    agreeableness        1834 non-null   float64
     2    emotional_stability  1834 non-null   float64
     3    conscientiousness    1834 non-null   float64
     4    extraversion         1834 non-null   float64
     5    assigned metric      1834 non-null   object 
     6    assigned condition   1834 non-null   int64  
     7    is_personalized      1834 non-null   int64  
     8    enjoy_watching       1834 non-null   int64  
    dtypes: float64(5), int64(3), object(1)
    memory usage: 129.1+ KB



```python
df.isna().sum()
```




    openness               0
    agreeableness          0
    emotional_stability    0
    conscientiousness      0
    extraversion           0
    assigned metric        0
    assigned condition     0
    is_personalized        0
    enjoy_watching         0
    dtype: int64




```python
for col in list(df.select_dtypes(object).columns):
    print({col: df[col].unique()})
```

    {'userid': array(['8e7cebf9a234c064b75016249f2ac65e',
           '77c7d756a093150d4377720abeaeef76',
           'b7e8a92987a530cc368719a0e60e26a3', ...,
           'a06386edadf3bc614dadb7044708c46c',
           'bad56d9506832cd79d874a6b66b3d813',
           '721ea658e148fc0f76ddd6e2b0e02422'], dtype=object)}
    {' assigned metric': array([' serendipity', ' all', ' popularity', ' diversity'], dtype=object)}
    {' assigned condition': array([' high', ' default', ' medium', ' low'], dtype=object)}



```python
def preprocess_inputs(df):

    df.copy()

    df = df.drop(
        [
            "userid",
            " movie_1",
            " predicted_rating_1",
            " movie_2",
            " predicted_rating_2",
            " movie_3",
            " predicted_rating_3",
            " movie_4",
            " predicted_rating_4",
            " movie_5",
            " predicted_rating_5",
            " movie_6",
            " predicted_rating_6",
            " movie_7",
            " predicted_rating_7",
            " movie_8",
            " predicted_rating_8",
            " movie_9",
            " predicted_rating_9",
            " movie_10",
            " predicted_rating_10",
            " movie_11",
            " predicted_rating_11",
            " movie_12",
            " predicted_rating_12",
        ],
        axis=1,
    )

    df[" assigned condition"] = df[" assigned condition"].map(
        {" low": 0, " medium": 1, " default": 2, " high": 3}
    )

    return df
```


```python
df = preprocess_inputs(df)
```


```python
df.columns
```




    Index([' openness', ' agreeableness', ' emotional_stability',
           ' conscientiousness', ' extraversion', ' assigned metric',
           ' assigned condition', ' is_personalized', ' enjoy_watching '],
          dtype='object')




```python

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
      <th>assigned metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>serendipity</td>
    </tr>
    <tr>
      <th>1</th>
      <td>all</td>
    </tr>
    <tr>
      <th>2</th>
      <td>serendipity</td>
    </tr>
    <tr>
      <th>3</th>
      <td>popularity</td>
    </tr>
    <tr>
      <th>4</th>
      <td>popularity</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>popularity</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>serendipity</td>
    </tr>
    <tr>
      <th>1831</th>
      <td>serendipity</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>serendipity</td>
    </tr>
    <tr>
      <th>1833</th>
      <td>popularity</td>
    </tr>
  </tbody>
</table>
<p>1834 rows × 1 columns</p>
</div>




```python
import numpy as np

numbers = [1, 2, 3]
result = np.sqrt(numbers)
print(result)
```

    [1.         1.41421356 1.73205081]



```python

```
