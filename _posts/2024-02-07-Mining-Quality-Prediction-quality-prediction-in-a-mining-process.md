---
title: "Mining Quality Prediction quality prediction in a mining process"
date: 2024-02-07
last_modified_at: 2024-02-07
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Mining Quality Prediction quality prediction in a mining process 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("edumagalhaes/quality-prediction-in-a-mining-process")

print("Path to dataset files:", path)
```

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.5)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/edumagalhaes/quality-prediction-in-a-mining-process/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

df = pd.read_csv(os.path.join(path, "MiningProcess_Flotation_Plant_Database.csv"))
```


```python
for cols in df.columns:
    df[cols] = df[cols].apply(lambda x: x.replace(",", "."))
```


```python
df["date"]
```




    0         2017-03-10 01:00:00
    1         2017-03-10 01:00:00
    2         2017-03-10 01:00:00
    3         2017-03-10 01:00:00
    4         2017-03-10 01:00:00
                     ...         
    737448    2017-09-09 23:00:00
    737449    2017-09-09 23:00:00
    737450    2017-09-09 23:00:00
    737451    2017-09-09 23:00:00
    737452    2017-09-09 23:00:00
    Name: date, Length: 737453, dtype: object




```python
def preprocess_input(df):
    df.copy()
    df["date"] = pd.to_datetime(df["date"])  # adding year and date columns
    df["month"] = df["date"].dt.month

    df = df.drop(["date"], axis=1)

    df = df.astype(float)

    corr = df.corr()
    corr_cols = corr[abs(corr["% Silica Concentrate"]) > 0.1].index.tolist()

    df = df[corr_cols]

    y = df["% Silica Concentrate"]
    X = df.drop(["% Silica Concentrate"], axis=1)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y
```


```python
# Modify the preprocess_input function to include log transformation
def preprocess_input2(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    df = df.drop(["date"], axis=1)
    df = df.astype(float)

    skew_df = pd.DataFrame(df.columns, columns=["feature"])
    skew_df["skew"] = abs(scipy.stats.skew(df))
    skew_df["skew_check"] = skew_df["skew"] > 0.5

    # Get features that need transformation
    features_to_transform = skew_df[skew_df["skew_check"]]["feature"].tolist()

    # Apply log1p transformation to highly skewed features
    for feature in features_to_transform:
        df[feature] = np.log1p(df[feature])

    corr = df.corr()
    corr_cols = corr[abs(corr["% Silica Concentrate"]) > 0.1].index.tolist()

    df = df[corr_cols]

    y = df["% Silica Concentrate"]
    X = df.drop(["% Silica Concentrate"], axis=1)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y
```


```python
X, y = preprocess_input2(df)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
```


```python
from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.6809614917814494




```python

```
