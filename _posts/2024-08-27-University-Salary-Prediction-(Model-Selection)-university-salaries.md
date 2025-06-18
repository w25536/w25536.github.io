---
title: "University Salary Prediction (Model Selection) university salaries"
date: 2024-08-27
last_modified_at: 2024-08-27
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "University Salary Prediction (Model Selection) university salaries 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("tysonpo/university-salaries")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/tysonpo/university-salaries/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv("/Users/jeongho/Desktop/w25536-kaggle/csv/salaries_final.csv")


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
      <th>Year</th>
      <th>Name</th>
      <th>Primary Job Title</th>
      <th>Base Pay</th>
      <th>Department</th>
      <th>College</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>Abaied, Jamie L.</td>
      <td>Assistant Professor</td>
      <td>64000.0</td>
      <td>Department of Psychological Science</td>
      <td>CAS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>Abaied, Jamie L.</td>
      <td>Assistant Professor</td>
      <td>64000.0</td>
      <td>Department of Psychological Science</td>
      <td>CAS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>Abaied, Jamie L.</td>
      <td>Assistant Professor</td>
      <td>65229.0</td>
      <td>Department of Psychological Science</td>
      <td>CAS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>Abaied, Jamie L.</td>
      <td>Assistant Professor</td>
      <td>66969.0</td>
      <td>Department of Psychological Science</td>
      <td>CAS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>Abaied, Jamie L.</td>
      <td>Assistant Professor</td>
      <td>68658.0</td>
      <td>Department of Psychological Science</td>
      <td>CAS</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14465</th>
      <td>2016</td>
      <td>van der Vliet, Albert</td>
      <td>Professor</td>
      <td>163635.0</td>
      <td>Department of Pathology&amp;Laboratory Medicine</td>
      <td>COM</td>
    </tr>
    <tr>
      <th>14466</th>
      <td>2017</td>
      <td>van der Vliet, Albert</td>
      <td>Professor</td>
      <td>175294.0</td>
      <td>Department of Pathology&amp;Laboratory Medicine</td>
      <td>COM</td>
    </tr>
    <tr>
      <th>14467</th>
      <td>2018</td>
      <td>van der Vliet, Albert</td>
      <td>Professor</td>
      <td>191000.0</td>
      <td>Department of Pathology&amp;Laboratory Medicine</td>
      <td>COM</td>
    </tr>
    <tr>
      <th>14468</th>
      <td>2019</td>
      <td>van der Vliet, Albert</td>
      <td>Professor</td>
      <td>196000.0</td>
      <td>Department of Pathology&amp;Laboratory Medicine</td>
      <td>COM</td>
    </tr>
    <tr>
      <th>14469</th>
      <td>2020</td>
      <td>van der Vliet, Albert</td>
      <td>Professor</td>
      <td>186200.0</td>
      <td>Department of Pathology&amp;Laboratory Medicine</td>
      <td>COM</td>
    </tr>
  </tbody>
</table>
<p>14470 rows × 6 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14470 entries, 0 to 14469
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Year               14470 non-null  int64  
     1   Name               14470 non-null  object 
     2   Primary Job Title  14470 non-null  object 
     3   Base Pay           14470 non-null  float64
     4   Department         14470 non-null  object 
     5   College            14470 non-null  object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 678.4+ KB



```python
df = df.drop(["Name"], axis=1)
```


```python
def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], dtype=int)
    df = pd.concat([dummies, df], axis=1)
    df = df.drop(column, axis=1)
    return df


def preprocess_input(df):
    df = df.copy()

    y = df["Base Pay"]
    X = df.drop(["Base Pay"], axis=1)

    X = onehot_encode(X, "Primary Job Title")
    X = onehot_encode(X, "Department")
    X = onehot_encode(X, "College")

    return X, y
```


```python
X, y = preprocess_input(df)
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
      <th>Business</th>
      <th>CALS</th>
      <th>CAS</th>
      <th>CEMS</th>
      <th>CESS</th>
      <th>CNHS</th>
      <th>COM</th>
      <th>Department of Ext</th>
      <th>LCOMEO</th>
      <th>Learning and Info Tech</th>
      <th>...</th>
      <th>Student/Academic Srvcs Manager</th>
      <th>Technical Support Generalist</th>
      <th>Technical Support Specialist</th>
      <th>UVM State Relations Officer</th>
      <th>VP Research</th>
      <th>Vice Pres for Enrollment Mgmnt</th>
      <th>Visiting Assistant Prof</th>
      <th>Visiting Instructor</th>
      <th>Visiting Lecturer</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>2011</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>2013</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>2014</td>
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
      <th>14465</th>
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
      <td>2016</td>
    </tr>
    <tr>
      <th>14466</th>
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
      <td>2017</td>
    </tr>
    <tr>
      <th>14467</th>
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
      <td>2018</td>
    </tr>
    <tr>
      <th>14468</th>
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
      <td>2019</td>
    </tr>
    <tr>
      <th>14469</th>
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
      <td>2020</td>
    </tr>
  </tbody>
</table>
<p>14470 rows × 269 columns</p>
</div>




```python

```


      Cell In[28], line 1
        df['year']?
                  ^
    SyntaxError: invalid syntax




```python

```
