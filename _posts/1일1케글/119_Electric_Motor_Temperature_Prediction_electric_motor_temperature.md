---
title: "119_Electric_Motor_Temperature_Prediction_electric_motor_temperature"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "119_Electric_Motor_Temperature_Prediction_electric_motor_temperature"
use_math: true
classes: wide
---

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("wkirgsn/electric-motor-temperature")

print("Path to dataset files:", path)
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
      warnings.warn(


    Downloading from https://www.kaggle.com/api/v1/datasets/download/wkirgsn/electric-motor-temperature?dataset_version_number=3...


    100%|██████████| 117M/117M [00:05<00:00, 24.4MB/s] 

    Extracting files...


    


    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/wkirgsn/electric-motor-temperature/versions/3



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split


df = pd.read_csv(os.path.join(path, "measures_v2.csv"))
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
      <th>u_q</th>
      <th>coolant</th>
      <th>stator_winding</th>
      <th>u_d</th>
      <th>stator_tooth</th>
      <th>motor_speed</th>
      <th>i_d</th>
      <th>i_q</th>
      <th>pm</th>
      <th>stator_yoke</th>
      <th>ambient</th>
      <th>torque</th>
      <th>profile_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.450682</td>
      <td>18.805172</td>
      <td>19.086670</td>
      <td>-0.350055</td>
      <td>18.293219</td>
      <td>0.002866</td>
      <td>0.004419</td>
      <td>0.000328</td>
      <td>24.554214</td>
      <td>18.316547</td>
      <td>19.850691</td>
      <td>1.871008e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.325737</td>
      <td>18.818571</td>
      <td>19.092390</td>
      <td>-0.305803</td>
      <td>18.294807</td>
      <td>0.000257</td>
      <td>0.000606</td>
      <td>-0.000785</td>
      <td>24.538078</td>
      <td>18.314955</td>
      <td>19.850672</td>
      <td>2.454175e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.440864</td>
      <td>18.828770</td>
      <td>19.089380</td>
      <td>-0.372503</td>
      <td>18.294094</td>
      <td>0.002355</td>
      <td>0.001290</td>
      <td>0.000386</td>
      <td>24.544693</td>
      <td>18.326307</td>
      <td>19.850657</td>
      <td>1.766153e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.327026</td>
      <td>18.835567</td>
      <td>19.083031</td>
      <td>-0.316199</td>
      <td>18.292542</td>
      <td>0.006105</td>
      <td>0.000026</td>
      <td>0.002046</td>
      <td>24.554018</td>
      <td>18.330833</td>
      <td>19.850647</td>
      <td>2.383027e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.471150</td>
      <td>18.857033</td>
      <td>19.082525</td>
      <td>-0.332272</td>
      <td>18.291428</td>
      <td>0.003133</td>
      <td>-0.064317</td>
      <td>0.037184</td>
      <td>24.565397</td>
      <td>18.326662</td>
      <td>19.850639</td>
      <td>2.081967e-01</td>
      <td>17</td>
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
    </tr>
    <tr>
      <th>1330811</th>
      <td>-2.152128</td>
      <td>30.721298</td>
      <td>45.274497</td>
      <td>0.791801</td>
      <td>41.934347</td>
      <td>0.000380</td>
      <td>-2.000169</td>
      <td>1.097528</td>
      <td>62.147780</td>
      <td>38.653720</td>
      <td>23.989078</td>
      <td>-8.116730e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330812</th>
      <td>-2.258684</td>
      <td>30.721306</td>
      <td>45.239017</td>
      <td>0.778900</td>
      <td>41.868923</td>
      <td>0.002985</td>
      <td>-2.000499</td>
      <td>1.097569</td>
      <td>62.142646</td>
      <td>38.656328</td>
      <td>23.970700</td>
      <td>-5.815891e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330813</th>
      <td>-2.130312</td>
      <td>30.721312</td>
      <td>45.211576</td>
      <td>0.804914</td>
      <td>41.804819</td>
      <td>0.002301</td>
      <td>-1.999268</td>
      <td>1.098765</td>
      <td>62.138387</td>
      <td>38.650923</td>
      <td>23.977234</td>
      <td>-4.167268e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330814</th>
      <td>-2.268498</td>
      <td>30.721316</td>
      <td>45.193508</td>
      <td>0.763091</td>
      <td>41.762220</td>
      <td>0.005662</td>
      <td>-2.000999</td>
      <td>1.095696</td>
      <td>62.133422</td>
      <td>38.655686</td>
      <td>24.001421</td>
      <td>-2.985978e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330815</th>
      <td>-2.100158</td>
      <td>30.721319</td>
      <td>45.132307</td>
      <td>0.807309</td>
      <td>41.734763</td>
      <td>0.004395</td>
      <td>-2.000792</td>
      <td>1.096487</td>
      <td>62.131429</td>
      <td>38.660370</td>
      <td>24.027522</td>
      <td>-2.139547e-60</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
<p>1330816 rows × 13 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1330816 entries, 0 to 1330815
    Data columns (total 13 columns):
     #   Column          Non-Null Count    Dtype  
    ---  ------          --------------    -----  
     0   u_q             1330816 non-null  float64
     1   coolant         1330816 non-null  float64
     2   stator_winding  1330816 non-null  float64
     3   u_d             1330816 non-null  float64
     4   stator_tooth    1330816 non-null  float64
     5   motor_speed     1330816 non-null  float64
     6   i_d             1330816 non-null  float64
     7   i_q             1330816 non-null  float64
     8   pm              1330816 non-null  float64
     9   stator_yoke     1330816 non-null  float64
     10  ambient         1330816 non-null  float64
     11  torque          1330816 non-null  float64
     12  profile_id      1330816 non-null  int64  
    dtypes: float64(12), int64(1)
    memory usage: 132.0 MB



```python
def preprocess_df(df):
    y = df["pm"].copy()
    X = df.drop(["pm"], axis=1).copy()

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, train_size=0.8, shuffle=True
    )

    return X, y
```


```python
X, y = preprocess_df(df)
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
      <th>u_q</th>
      <th>coolant</th>
      <th>stator_winding</th>
      <th>u_d</th>
      <th>stator_tooth</th>
      <th>motor_speed</th>
      <th>i_d</th>
      <th>i_q</th>
      <th>stator_yoke</th>
      <th>ambient</th>
      <th>torque</th>
      <th>profile_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.450682</td>
      <td>18.805172</td>
      <td>19.086670</td>
      <td>-0.350055</td>
      <td>18.293219</td>
      <td>0.002866</td>
      <td>0.004419</td>
      <td>0.000328</td>
      <td>18.316547</td>
      <td>19.850691</td>
      <td>1.871008e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.325737</td>
      <td>18.818571</td>
      <td>19.092390</td>
      <td>-0.305803</td>
      <td>18.294807</td>
      <td>0.000257</td>
      <td>0.000606</td>
      <td>-0.000785</td>
      <td>18.314955</td>
      <td>19.850672</td>
      <td>2.454175e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.440864</td>
      <td>18.828770</td>
      <td>19.089380</td>
      <td>-0.372503</td>
      <td>18.294094</td>
      <td>0.002355</td>
      <td>0.001290</td>
      <td>0.000386</td>
      <td>18.326307</td>
      <td>19.850657</td>
      <td>1.766153e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.327026</td>
      <td>18.835567</td>
      <td>19.083031</td>
      <td>-0.316199</td>
      <td>18.292542</td>
      <td>0.006105</td>
      <td>0.000026</td>
      <td>0.002046</td>
      <td>18.330833</td>
      <td>19.850647</td>
      <td>2.383027e-01</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.471150</td>
      <td>18.857033</td>
      <td>19.082525</td>
      <td>-0.332272</td>
      <td>18.291428</td>
      <td>0.003133</td>
      <td>-0.064317</td>
      <td>0.037184</td>
      <td>18.326662</td>
      <td>19.850639</td>
      <td>2.081967e-01</td>
      <td>17</td>
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
    </tr>
    <tr>
      <th>1330811</th>
      <td>-2.152128</td>
      <td>30.721298</td>
      <td>45.274497</td>
      <td>0.791801</td>
      <td>41.934347</td>
      <td>0.000380</td>
      <td>-2.000169</td>
      <td>1.097528</td>
      <td>38.653720</td>
      <td>23.989078</td>
      <td>-8.116730e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330812</th>
      <td>-2.258684</td>
      <td>30.721306</td>
      <td>45.239017</td>
      <td>0.778900</td>
      <td>41.868923</td>
      <td>0.002985</td>
      <td>-2.000499</td>
      <td>1.097569</td>
      <td>38.656328</td>
      <td>23.970700</td>
      <td>-5.815891e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330813</th>
      <td>-2.130312</td>
      <td>30.721312</td>
      <td>45.211576</td>
      <td>0.804914</td>
      <td>41.804819</td>
      <td>0.002301</td>
      <td>-1.999268</td>
      <td>1.098765</td>
      <td>38.650923</td>
      <td>23.977234</td>
      <td>-4.167268e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330814</th>
      <td>-2.268498</td>
      <td>30.721316</td>
      <td>45.193508</td>
      <td>0.763091</td>
      <td>41.762220</td>
      <td>0.005662</td>
      <td>-2.000999</td>
      <td>1.095696</td>
      <td>38.655686</td>
      <td>24.001421</td>
      <td>-2.985978e-60</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1330815</th>
      <td>-2.100158</td>
      <td>30.721319</td>
      <td>45.132307</td>
      <td>0.807309</td>
      <td>41.734763</td>
      <td>0.004395</td>
      <td>-2.000792</td>
      <td>1.096487</td>
      <td>38.660370</td>
      <td>24.027522</td>
      <td>-2.139547e-60</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
<p>1330816 rows × 12 columns</p>
</div>




```python
import pandas as pd

df = pd.read_csv("/Users/jeongho/Desktop/w25536-kaggle/hello.csv").reset_index(
    drop=True
)
```


```python
df = df[["index", "args"]].sort_index(ascending=False).reset_index(drop=True)
```


```python
df.drop(["index"], axis=1, inplace=True)
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
      <th>args</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>### [Predicting LoL Wins with TensorFlow - Dat...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>### [Predicting City Houses With Scikit-learn ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>### [Handwritten Digit Recognition Using Tenso...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>### [Using PCA to Understand College Admission...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>### [Analyzing Diamond Price Data - Data Every...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>249</th>
      <td>### [Product Sales Prediction (Class Imbalance...</td>
    </tr>
    <tr>
      <th>250</th>
      <td>### [Hospital Patient Type Prediction (Model S...</td>
    </tr>
    <tr>
      <th>251</th>
      <td>### [YouTube Subscriber Count Prediction - Dat...</td>
    </tr>
    <tr>
      <th>252</th>
      <td>### [Understanding Principal Component Analysi...</td>
    </tr>
    <tr>
      <th>253</th>
      <td>### [Understanding Bias-Variance Tradeoff](htt...</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 1 columns</p>
</div>




```python
df.to_csv("final.csv")
```


```python

```
