---
title: "Liver Disease Prediction indian liver patient records"
date: 2024-02-16
last_modified_at: 2024-02-16
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Liver Disease Prediction indian liver patient records 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/indian-liver-patient-records")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/uciml/indian-liver-patient-records/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv(os.path.join(path, "indian_liver_patient.csv"))
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
      <th>Age</th>
      <th>Gender</th>
      <th>Total_Bilirubin</th>
      <th>Direct_Bilirubin</th>
      <th>Alkaline_Phosphotase</th>
      <th>Alamine_Aminotransferase</th>
      <th>Aspartate_Aminotransferase</th>
      <th>Total_Protiens</th>
      <th>Albumin</th>
      <th>Albumin_and_Globulin_Ratio</th>
      <th>Dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>187</td>
      <td>16</td>
      <td>18</td>
      <td>6.8</td>
      <td>3.3</td>
      <td>0.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>1</td>
      <td>10.9</td>
      <td>5.5</td>
      <td>699</td>
      <td>64</td>
      <td>100</td>
      <td>7.5</td>
      <td>3.2</td>
      <td>0.74</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62</td>
      <td>1</td>
      <td>7.3</td>
      <td>4.1</td>
      <td>490</td>
      <td>60</td>
      <td>68</td>
      <td>7.0</td>
      <td>3.3</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.4</td>
      <td>182</td>
      <td>14</td>
      <td>20</td>
      <td>6.8</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>1</td>
      <td>3.9</td>
      <td>2.0</td>
      <td>195</td>
      <td>27</td>
      <td>59</td>
      <td>7.3</td>
      <td>2.4</td>
      <td>0.40</td>
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
    </tr>
    <tr>
      <th>578</th>
      <td>60</td>
      <td>1</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>500</td>
      <td>20</td>
      <td>34</td>
      <td>5.9</td>
      <td>1.6</td>
      <td>0.37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>579</th>
      <td>40</td>
      <td>1</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>98</td>
      <td>35</td>
      <td>31</td>
      <td>6.0</td>
      <td>3.2</td>
      <td>1.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>580</th>
      <td>52</td>
      <td>1</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>245</td>
      <td>48</td>
      <td>49</td>
      <td>6.4</td>
      <td>3.2</td>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>581</th>
      <td>31</td>
      <td>1</td>
      <td>1.3</td>
      <td>0.5</td>
      <td>184</td>
      <td>29</td>
      <td>32</td>
      <td>6.8</td>
      <td>3.4</td>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>582</th>
      <td>38</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>216</td>
      <td>21</td>
      <td>24</td>
      <td>7.3</td>
      <td>4.4</td>
      <td>1.50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>583 rows × 11 columns</p>
</div>




```python
df["Gender"] = df["Gender"].replace({"Female": 0, "Male": 1})
df["Dataset"] = df["Dataset"].replace({1: 0, 2: 1})
```

    /var/folders/v7/tlyx9w190ks2gfgzd_j0l5c80000gn/T/ipykernel_69545/765071311.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df['Gender'] = df['Gender'].replace({'Female': 0, 'Male':1})



```python
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
    df["Albumin_and_Globulin_Ratio"].mean()
)
```


```python
df.isna().sum()
```




    Age                           0
    Gender                        0
    Total_Bilirubin               0
    Direct_Bilirubin              0
    Alkaline_Phosphotase          0
    Alamine_Aminotransferase      0
    Aspartate_Aminotransferase    0
    Total_Protiens                0
    Albumin                       0
    Albumin_and_Globulin_Ratio    0
    Dataset                       0
    dtype: int64




```python
y = df["Dataset"]
X = df.drop(["Dataset"], axis=1)
```


```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
```


```python
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


LC = LGBMClassifier()
CBC = CatBoostClassifier()
RFC = RandomForestClassifier()
ETC = ExtraTreesClassifier()
GBC = GradientBoostingClassifier()

models = [LC, CBC, RFC, ETC, GBC]
result = []
d = {}
for model in models:
    model.fit(X_train, y_train)
    auc = model.score(X_test, y_test)
    print(model, ":", auc)
    result.append((model, auc))
```

    <frozen importlib._bootstrap>:228: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216 from C header, got 232 from PyObject


    [LightGBM] [Info] Number of positive: 124, number of negative: 342
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000363 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 556
    [LightGBM] [Info] Number of data points in the train set: 466, number of used features: 10
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.266094 -> initscore=-1.014529
    [LightGBM] [Info] Start training from score -1.014529
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    LGBMClassifier() : 0.7008547008547008
    Learning rate set to 0.007436
    0:	learn: 0.6900059	total: 58.4ms	remaining: 58.4s
    1:	learn: 0.6864079	total: 59.1ms	remaining: 29.5s
    2:	learn: 0.6833188	total: 59.7ms	remaining: 19.9s
    3:	learn: 0.6800684	total: 60.3ms	remaining: 15s
    4:	learn: 0.6762226	total: 60.9ms	remaining: 12.1s
    5:	learn: 0.6730621	total: 61.4ms	remaining: 10.2s
    6:	learn: 0.6694960	total: 62ms	remaining: 8.8s
    7:	learn: 0.6660943	total: 62.6ms	remaining: 7.76s
    8:	learn: 0.6633159	total: 63.1ms	remaining: 6.95s
    9:	learn: 0.6603879	total: 63.7ms	remaining: 6.31s
    10:	learn: 0.6571416	total: 64.4ms	remaining: 5.79s
    11:	learn: 0.6545837	total: 65ms	remaining: 5.35s
    12:	learn: 0.6530978	total: 65.3ms	remaining: 4.95s
    13:	learn: 0.6501036	total: 66.1ms	remaining: 4.65s
    14:	learn: 0.6469618	total: 66.7ms	remaining: 4.38s
    15:	learn: 0.6444124	total: 67.1ms	remaining: 4.13s
    16:	learn: 0.6419305	total: 67.8ms	remaining: 3.92s
    17:	learn: 0.6393913	total: 68.5ms	remaining: 3.73s
    18:	learn: 0.6368369	total: 69.1ms	remaining: 3.57s
    19:	learn: 0.6342023	total: 69.7ms	remaining: 3.41s
    20:	learn: 0.6319237	total: 70.3ms	remaining: 3.28s
    21:	learn: 0.6294605	total: 70.8ms	remaining: 3.15s
    22:	learn: 0.6272487	total: 71.6ms	remaining: 3.04s
    23:	learn: 0.6249968	total: 72.1ms	remaining: 2.93s
    24:	learn: 0.6228085	total: 72.8ms	remaining: 2.84s
    25:	learn: 0.6203367	total: 73.4ms	remaining: 2.75s
    26:	learn: 0.6179210	total: 74.1ms	remaining: 2.67s
    27:	learn: 0.6155452	total: 74.7ms	remaining: 2.59s
    28:	learn: 0.6131344	total: 75.2ms	remaining: 2.52s
    29:	learn: 0.6105930	total: 75.9ms	remaining: 2.45s
    30:	learn: 0.6079953	total: 76.4ms	remaining: 2.39s
    31:	learn: 0.6055893	total: 77ms	remaining: 2.33s
    32:	learn: 0.6029864	total: 77.6ms	remaining: 2.27s
    33:	learn: 0.6017537	total: 78ms	remaining: 2.21s
    34:	learn: 0.5994726	total: 78.5ms	remaining: 2.16s
    35:	learn: 0.5968846	total: 79ms	remaining: 2.11s
    36:	learn: 0.5947800	total: 79.5ms	remaining: 2.07s
    37:	learn: 0.5928444	total: 80ms	remaining: 2.03s
    38:	learn: 0.5907257	total: 80.5ms	remaining: 1.98s
    39:	learn: 0.5884825	total: 81.1ms	remaining: 1.95s
    40:	learn: 0.5865457	total: 81.8ms	remaining: 1.91s
    41:	learn: 0.5847299	total: 82.3ms	remaining: 1.88s
    42:	learn: 0.5829846	total: 82.9ms	remaining: 1.84s
    43:	learn: 0.5812360	total: 83.6ms	remaining: 1.82s
    44:	learn: 0.5789716	total: 84.3ms	remaining: 1.79s
    45:	learn: 0.5771148	total: 84.9ms	remaining: 1.76s
    46:	learn: 0.5758113	total: 85.4ms	remaining: 1.73s
    47:	learn: 0.5744636	total: 86ms	remaining: 1.71s
    48:	learn: 0.5726687	total: 86.5ms	remaining: 1.68s
    49:	learn: 0.5711980	total: 87.2ms	remaining: 1.66s
    50:	learn: 0.5694067	total: 87.8ms	remaining: 1.63s
    51:	learn: 0.5680455	total: 88.3ms	remaining: 1.61s
    52:	learn: 0.5663397	total: 88.9ms	remaining: 1.59s
    53:	learn: 0.5651119	total: 89.2ms	remaining: 1.56s
    54:	learn: 0.5629371	total: 89.8ms	remaining: 1.54s
    55:	learn: 0.5612038	total: 90.4ms	remaining: 1.52s
    56:	learn: 0.5596043	total: 90.9ms	remaining: 1.5s
    57:	learn: 0.5577401	total: 91.7ms	remaining: 1.49s
    58:	learn: 0.5563266	total: 92.3ms	remaining: 1.47s
    59:	learn: 0.5552347	total: 92.9ms	remaining: 1.46s
    60:	learn: 0.5534093	total: 93.6ms	remaining: 1.44s
    61:	learn: 0.5519119	total: 94.1ms	remaining: 1.42s
    62:	learn: 0.5505780	total: 94.8ms	remaining: 1.41s
    63:	learn: 0.5494235	total: 95.4ms	remaining: 1.4s
    64:	learn: 0.5476649	total: 96.4ms	remaining: 1.39s
    65:	learn: 0.5463614	total: 97.1ms	remaining: 1.37s
    66:	learn: 0.5449232	total: 97.8ms	remaining: 1.36s
    67:	learn: 0.5435599	total: 98.4ms	remaining: 1.35s
    68:	learn: 0.5424995	total: 99ms	remaining: 1.33s
    69:	learn: 0.5416870	total: 99.5ms	remaining: 1.32s
    70:	learn: 0.5404293	total: 100ms	remaining: 1.31s
    71:	learn: 0.5391267	total: 101ms	remaining: 1.3s
    72:	learn: 0.5373482	total: 101ms	remaining: 1.29s
    73:	learn: 0.5365229	total: 102ms	remaining: 1.27s
    74:	learn: 0.5352094	total: 103ms	remaining: 1.27s
    75:	learn: 0.5340867	total: 104ms	remaining: 1.26s
    76:	learn: 0.5327820	total: 104ms	remaining: 1.25s
    77:	learn: 0.5315996	total: 105ms	remaining: 1.24s
    78:	learn: 0.5304499	total: 106ms	remaining: 1.23s
    79:	learn: 0.5291836	total: 106ms	remaining: 1.22s
    80:	learn: 0.5281404	total: 107ms	remaining: 1.21s
    81:	learn: 0.5267154	total: 107ms	remaining: 1.2s
    82:	learn: 0.5254110	total: 108ms	remaining: 1.19s
    83:	learn: 0.5243634	total: 109ms	remaining: 1.18s
    84:	learn: 0.5228589	total: 109ms	remaining: 1.18s
    85:	learn: 0.5219925	total: 110ms	remaining: 1.17s
    86:	learn: 0.5212158	total: 110ms	remaining: 1.16s
    87:	learn: 0.5202507	total: 111ms	remaining: 1.15s
    88:	learn: 0.5192802	total: 112ms	remaining: 1.14s
    89:	learn: 0.5182288	total: 112ms	remaining: 1.13s
    90:	learn: 0.5166701	total: 113ms	remaining: 1.12s
    91:	learn: 0.5156167	total: 113ms	remaining: 1.12s
    92:	learn: 0.5139568	total: 114ms	remaining: 1.11s
    93:	learn: 0.5129607	total: 114ms	remaining: 1.1s
    94:	learn: 0.5120638	total: 115ms	remaining: 1.1s
    95:	learn: 0.5111236	total: 116ms	remaining: 1.09s
    96:	learn: 0.5097458	total: 116ms	remaining: 1.08s
    97:	learn: 0.5084378	total: 117ms	remaining: 1.08s
    98:	learn: 0.5076118	total: 118ms	remaining: 1.07s
    99:	learn: 0.5062589	total: 118ms	remaining: 1.06s
    100:	learn: 0.5052483	total: 119ms	remaining: 1.06s
    101:	learn: 0.5039848	total: 119ms	remaining: 1.05s
    102:	learn: 0.5030603	total: 120ms	remaining: 1.04s
    103:	learn: 0.5027424	total: 121ms	remaining: 1.04s
    104:	learn: 0.5017849	total: 121ms	remaining: 1.03s
    105:	learn: 0.5006902	total: 122ms	remaining: 1.02s
    106:	learn: 0.4998047	total: 122ms	remaining: 1.02s
    107:	learn: 0.4987271	total: 123ms	remaining: 1.01s
    108:	learn: 0.4975987	total: 124ms	remaining: 1.01s
    109:	learn: 0.4967883	total: 124ms	remaining: 1s
    110:	learn: 0.4959506	total: 125ms	remaining: 998ms
    111:	learn: 0.4950371	total: 125ms	remaining: 992ms
    112:	learn: 0.4941611	total: 126ms	remaining: 987ms
    113:	learn: 0.4927033	total: 127ms	remaining: 986ms
    114:	learn: 0.4917195	total: 127ms	remaining: 980ms
    115:	learn: 0.4910398	total: 128ms	remaining: 975ms
    116:	learn: 0.4902590	total: 129ms	remaining: 972ms
    117:	learn: 0.4896860	total: 129ms	remaining: 967ms
    118:	learn: 0.4890223	total: 130ms	remaining: 962ms
    119:	learn: 0.4882082	total: 131ms	remaining: 958ms
    120:	learn: 0.4872381	total: 131ms	remaining: 953ms
    121:	learn: 0.4863859	total: 132ms	remaining: 948ms
    122:	learn: 0.4856413	total: 132ms	remaining: 943ms
    123:	learn: 0.4848923	total: 133ms	remaining: 939ms
    124:	learn: 0.4840437	total: 133ms	remaining: 934ms
    125:	learn: 0.4832442	total: 134ms	remaining: 930ms
    126:	learn: 0.4827834	total: 135ms	remaining: 925ms
    127:	learn: 0.4822410	total: 135ms	remaining: 920ms
    128:	learn: 0.4814716	total: 136ms	remaining: 916ms
    129:	learn: 0.4807993	total: 136ms	remaining: 913ms
    130:	learn: 0.4801462	total: 137ms	remaining: 909ms
    131:	learn: 0.4794269	total: 138ms	remaining: 904ms
    132:	learn: 0.4788179	total: 138ms	remaining: 900ms
    133:	learn: 0.4781698	total: 139ms	remaining: 896ms
    134:	learn: 0.4774832	total: 139ms	remaining: 892ms
    135:	learn: 0.4768591	total: 140ms	remaining: 888ms
    136:	learn: 0.4762429	total: 140ms	remaining: 884ms
    137:	learn: 0.4757249	total: 141ms	remaining: 880ms
    138:	learn: 0.4751361	total: 142ms	remaining: 877ms
    139:	learn: 0.4745283	total: 142ms	remaining: 873ms
    140:	learn: 0.4740567	total: 143ms	remaining: 869ms
    141:	learn: 0.4733878	total: 143ms	remaining: 866ms
    142:	learn: 0.4727453	total: 144ms	remaining: 864ms
    143:	learn: 0.4720825	total: 145ms	remaining: 860ms
    144:	learn: 0.4713961	total: 145ms	remaining: 857ms
    145:	learn: 0.4701214	total: 146ms	remaining: 853ms
    146:	learn: 0.4692326	total: 146ms	remaining: 849ms
    147:	learn: 0.4686049	total: 147ms	remaining: 846ms
    148:	learn: 0.4678246	total: 147ms	remaining: 842ms
    149:	learn: 0.4671767	total: 148ms	remaining: 839ms
    150:	learn: 0.4665917	total: 149ms	remaining: 836ms
    151:	learn: 0.4658539	total: 149ms	remaining: 832ms
    152:	learn: 0.4648554	total: 150ms	remaining: 833ms
    153:	learn: 0.4641628	total: 151ms	remaining: 830ms
    154:	learn: 0.4637011	total: 152ms	remaining: 826ms
    155:	learn: 0.4627630	total: 152ms	remaining: 824ms
    156:	learn: 0.4619240	total: 153ms	remaining: 821ms
    157:	learn: 0.4608356	total: 154ms	remaining: 819ms
    158:	learn: 0.4604589	total: 154ms	remaining: 815ms
    159:	learn: 0.4598812	total: 155ms	remaining: 812ms
    160:	learn: 0.4590287	total: 155ms	remaining: 809ms
    161:	learn: 0.4585490	total: 156ms	remaining: 806ms
    162:	learn: 0.4579912	total: 157ms	remaining: 804ms
    163:	learn: 0.4576054	total: 157ms	remaining: 801ms
    164:	learn: 0.4571047	total: 158ms	remaining: 798ms
    165:	learn: 0.4565667	total: 158ms	remaining: 795ms
    166:	learn: 0.4557034	total: 159ms	remaining: 792ms
    167:	learn: 0.4553463	total: 159ms	remaining: 789ms
    168:	learn: 0.4546067	total: 160ms	remaining: 786ms
    169:	learn: 0.4539958	total: 160ms	remaining: 783ms
    170:	learn: 0.4537559	total: 161ms	remaining: 781ms
    171:	learn: 0.4529858	total: 162ms	remaining: 778ms
    172:	learn: 0.4525373	total: 162ms	remaining: 775ms
    173:	learn: 0.4520601	total: 163ms	remaining: 772ms
    174:	learn: 0.4513802	total: 163ms	remaining: 769ms
    175:	learn: 0.4507016	total: 164ms	remaining: 766ms
    176:	learn: 0.4502143	total: 164ms	remaining: 764ms
    177:	learn: 0.4499131	total: 165ms	remaining: 761ms
    178:	learn: 0.4494011	total: 165ms	remaining: 758ms
    179:	learn: 0.4485262	total: 166ms	remaining: 756ms
    180:	learn: 0.4480360	total: 166ms	remaining: 753ms
    181:	learn: 0.4474042	total: 167ms	remaining: 751ms
    182:	learn: 0.4469339	total: 168ms	remaining: 749ms
    183:	learn: 0.4464399	total: 168ms	remaining: 747ms
    184:	learn: 0.4459889	total: 169ms	remaining: 744ms
    185:	learn: 0.4453570	total: 169ms	remaining: 742ms
    186:	learn: 0.4447749	total: 170ms	remaining: 739ms
    187:	learn: 0.4440652	total: 171ms	remaining: 737ms
    188:	learn: 0.4437809	total: 171ms	remaining: 735ms
    189:	learn: 0.4431892	total: 172ms	remaining: 732ms
    190:	learn: 0.4424966	total: 172ms	remaining: 729ms
    191:	learn: 0.4419844	total: 173ms	remaining: 727ms
    192:	learn: 0.4411484	total: 173ms	remaining: 725ms
    193:	learn: 0.4405922	total: 174ms	remaining: 722ms
    194:	learn: 0.4403058	total: 174ms	remaining: 720ms
    195:	learn: 0.4397364	total: 175ms	remaining: 718ms
    196:	learn: 0.4393277	total: 175ms	remaining: 715ms
    197:	learn: 0.4388362	total: 176ms	remaining: 713ms
    198:	learn: 0.4384709	total: 177ms	remaining: 711ms
    199:	learn: 0.4378120	total: 177ms	remaining: 709ms
    200:	learn: 0.4375095	total: 178ms	remaining: 707ms
    201:	learn: 0.4371855	total: 178ms	remaining: 705ms
    202:	learn: 0.4368309	total: 179ms	remaining: 703ms
    203:	learn: 0.4366226	total: 180ms	remaining: 700ms
    204:	learn: 0.4361770	total: 180ms	remaining: 698ms
    205:	learn: 0.4358739	total: 181ms	remaining: 696ms
    206:	learn: 0.4355422	total: 182ms	remaining: 697ms
    207:	learn: 0.4352074	total: 183ms	remaining: 695ms
    208:	learn: 0.4348186	total: 183ms	remaining: 693ms
    209:	learn: 0.4344472	total: 184ms	remaining: 691ms
    210:	learn: 0.4339857	total: 184ms	remaining: 689ms
    211:	learn: 0.4335099	total: 185ms	remaining: 687ms
    212:	learn: 0.4332278	total: 185ms	remaining: 685ms
    213:	learn: 0.4326352	total: 186ms	remaining: 683ms
    214:	learn: 0.4321278	total: 186ms	remaining: 681ms
    215:	learn: 0.4317127	total: 187ms	remaining: 679ms
    216:	learn: 0.4313383	total: 188ms	remaining: 677ms
    217:	learn: 0.4305174	total: 188ms	remaining: 675ms
    218:	learn: 0.4299403	total: 189ms	remaining: 674ms
    219:	learn: 0.4296809	total: 190ms	remaining: 672ms
    220:	learn: 0.4294953	total: 190ms	remaining: 670ms
    221:	learn: 0.4290701	total: 191ms	remaining: 668ms
    222:	learn: 0.4287182	total: 191ms	remaining: 667ms
    223:	learn: 0.4282729	total: 192ms	remaining: 665ms
    224:	learn: 0.4273812	total: 193ms	remaining: 664ms
    225:	learn: 0.4270160	total: 193ms	remaining: 662ms
    226:	learn: 0.4267201	total: 194ms	remaining: 660ms
    227:	learn: 0.4259625	total: 194ms	remaining: 658ms
    228:	learn: 0.4254779	total: 195ms	remaining: 656ms
    229:	learn: 0.4247117	total: 196ms	remaining: 655ms
    230:	learn: 0.4242512	total: 196ms	remaining: 653ms
    231:	learn: 0.4235429	total: 197ms	remaining: 654ms
    232:	learn: 0.4231020	total: 198ms	remaining: 652ms
    233:	learn: 0.4228123	total: 199ms	remaining: 650ms
    234:	learn: 0.4222843	total: 199ms	remaining: 649ms
    235:	learn: 0.4220112	total: 200ms	remaining: 647ms
    236:	learn: 0.4217627	total: 200ms	remaining: 645ms
    237:	learn: 0.4213829	total: 201ms	remaining: 643ms
    238:	learn: 0.4210528	total: 202ms	remaining: 642ms
    239:	learn: 0.4204165	total: 202ms	remaining: 640ms
    240:	learn: 0.4199213	total: 203ms	remaining: 639ms
    241:	learn: 0.4195810	total: 204ms	remaining: 637ms
    242:	learn: 0.4192379	total: 204ms	remaining: 636ms
    243:	learn: 0.4189072	total: 205ms	remaining: 635ms
    244:	learn: 0.4185226	total: 205ms	remaining: 633ms
    245:	learn: 0.4181171	total: 206ms	remaining: 631ms
    246:	learn: 0.4176551	total: 206ms	remaining: 629ms
    247:	learn: 0.4173407	total: 207ms	remaining: 628ms
    248:	learn: 0.4170391	total: 208ms	remaining: 629ms
    249:	learn: 0.4165833	total: 209ms	remaining: 627ms
    250:	learn: 0.4162988	total: 210ms	remaining: 626ms
    251:	learn: 0.4160483	total: 210ms	remaining: 624ms
    252:	learn: 0.4156448	total: 211ms	remaining: 623ms
    253:	learn: 0.4153703	total: 211ms	remaining: 621ms
    254:	learn: 0.4152300	total: 212ms	remaining: 619ms
    255:	learn: 0.4148593	total: 213ms	remaining: 618ms
    256:	learn: 0.4140809	total: 213ms	remaining: 617ms
    257:	learn: 0.4133258	total: 214ms	remaining: 615ms
    258:	learn: 0.4129515	total: 215ms	remaining: 614ms
    259:	learn: 0.4125521	total: 215ms	remaining: 612ms
    260:	learn: 0.4121589	total: 216ms	remaining: 611ms
    261:	learn: 0.4118715	total: 216ms	remaining: 609ms
    262:	learn: 0.4114263	total: 217ms	remaining: 608ms
    263:	learn: 0.4109749	total: 217ms	remaining: 606ms
    264:	learn: 0.4107452	total: 218ms	remaining: 605ms
    265:	learn: 0.4104055	total: 219ms	remaining: 603ms
    266:	learn: 0.4101256	total: 219ms	remaining: 602ms
    267:	learn: 0.4093754	total: 220ms	remaining: 600ms
    268:	learn: 0.4088864	total: 220ms	remaining: 599ms
    269:	learn: 0.4085278	total: 221ms	remaining: 598ms
    270:	learn: 0.4081820	total: 222ms	remaining: 596ms
    271:	learn: 0.4077801	total: 222ms	remaining: 595ms
    272:	learn: 0.4072734	total: 223ms	remaining: 593ms
    273:	learn: 0.4067491	total: 223ms	remaining: 592ms
    274:	learn: 0.4064761	total: 224ms	remaining: 590ms
    275:	learn: 0.4061343	total: 224ms	remaining: 589ms
    276:	learn: 0.4056815	total: 225ms	remaining: 587ms
    277:	learn: 0.4053930	total: 225ms	remaining: 586ms
    278:	learn: 0.4048691	total: 226ms	remaining: 584ms
    279:	learn: 0.4046155	total: 227ms	remaining: 583ms
    280:	learn: 0.4042393	total: 227ms	remaining: 581ms
    281:	learn: 0.4038912	total: 228ms	remaining: 580ms
    282:	learn: 0.4037128	total: 228ms	remaining: 578ms
    283:	learn: 0.4033643	total: 229ms	remaining: 577ms
    284:	learn: 0.4029243	total: 229ms	remaining: 575ms
    285:	learn: 0.4026869	total: 230ms	remaining: 574ms
    286:	learn: 0.4024917	total: 231ms	remaining: 573ms
    287:	learn: 0.4021239	total: 231ms	remaining: 572ms
    288:	learn: 0.4019082	total: 232ms	remaining: 571ms
    289:	learn: 0.4016146	total: 233ms	remaining: 569ms
    290:	learn: 0.4013253	total: 233ms	remaining: 568ms
    291:	learn: 0.4008214	total: 234ms	remaining: 567ms
    292:	learn: 0.4003491	total: 234ms	remaining: 565ms
    293:	learn: 0.3999885	total: 235ms	remaining: 564ms
    294:	learn: 0.3997554	total: 236ms	remaining: 563ms
    295:	learn: 0.3993389	total: 236ms	remaining: 562ms
    296:	learn: 0.3987701	total: 237ms	remaining: 560ms
    297:	learn: 0.3985415	total: 237ms	remaining: 559ms
    298:	learn: 0.3982867	total: 238ms	remaining: 558ms
    299:	learn: 0.3981456	total: 238ms	remaining: 556ms
    300:	learn: 0.3979144	total: 239ms	remaining: 555ms
    301:	learn: 0.3974762	total: 240ms	remaining: 554ms
    302:	learn: 0.3970480	total: 240ms	remaining: 552ms
    303:	learn: 0.3965659	total: 241ms	remaining: 551ms
    304:	learn: 0.3963422	total: 241ms	remaining: 550ms
    305:	learn: 0.3960652	total: 242ms	remaining: 548ms
    306:	learn: 0.3957943	total: 242ms	remaining: 547ms
    307:	learn: 0.3955195	total: 243ms	remaining: 545ms
    308:	learn: 0.3951804	total: 243ms	remaining: 544ms
    309:	learn: 0.3949236	total: 244ms	remaining: 543ms
    310:	learn: 0.3946794	total: 245ms	remaining: 542ms
    311:	learn: 0.3943107	total: 246ms	remaining: 541ms
    312:	learn: 0.3938953	total: 246ms	remaining: 540ms
    313:	learn: 0.3936791	total: 247ms	remaining: 539ms
    314:	learn: 0.3932313	total: 247ms	remaining: 537ms
    315:	learn: 0.3928992	total: 248ms	remaining: 536ms
    316:	learn: 0.3925856	total: 248ms	remaining: 535ms
    317:	learn: 0.3923015	total: 249ms	remaining: 533ms
    318:	learn: 0.3922123	total: 249ms	remaining: 532ms
    319:	learn: 0.3918575	total: 250ms	remaining: 531ms
    320:	learn: 0.3915127	total: 250ms	remaining: 530ms
    321:	learn: 0.3912627	total: 251ms	remaining: 528ms
    322:	learn: 0.3910641	total: 252ms	remaining: 527ms
    323:	learn: 0.3907755	total: 252ms	remaining: 526ms
    324:	learn: 0.3904194	total: 253ms	remaining: 525ms
    325:	learn: 0.3897463	total: 253ms	remaining: 523ms
    326:	learn: 0.3896119	total: 254ms	remaining: 522ms
    327:	learn: 0.3893152	total: 254ms	remaining: 521ms
    328:	learn: 0.3888234	total: 255ms	remaining: 520ms
    329:	learn: 0.3885471	total: 256ms	remaining: 519ms
    330:	learn: 0.3882038	total: 256ms	remaining: 518ms
    331:	learn: 0.3880084	total: 257ms	remaining: 516ms
    332:	learn: 0.3875946	total: 257ms	remaining: 515ms
    333:	learn: 0.3871243	total: 258ms	remaining: 514ms
    334:	learn: 0.3868206	total: 258ms	remaining: 513ms
    335:	learn: 0.3866322	total: 259ms	remaining: 511ms
    336:	learn: 0.3864295	total: 259ms	remaining: 510ms
    337:	learn: 0.3861504	total: 260ms	remaining: 509ms
    338:	learn: 0.3857174	total: 261ms	remaining: 508ms
    339:	learn: 0.3854485	total: 261ms	remaining: 507ms
    340:	learn: 0.3849992	total: 262ms	remaining: 506ms
    341:	learn: 0.3847975	total: 262ms	remaining: 505ms
    342:	learn: 0.3845939	total: 263ms	remaining: 503ms
    343:	learn: 0.3842045	total: 263ms	remaining: 502ms
    344:	learn: 0.3840163	total: 264ms	remaining: 501ms
    345:	learn: 0.3835593	total: 265ms	remaining: 500ms
    346:	learn: 0.3831672	total: 265ms	remaining: 499ms
    347:	learn: 0.3823489	total: 266ms	remaining: 497ms
    348:	learn: 0.3821777	total: 266ms	remaining: 497ms
    349:	learn: 0.3819606	total: 267ms	remaining: 495ms
    350:	learn: 0.3817098	total: 267ms	remaining: 494ms
    351:	learn: 0.3814154	total: 268ms	remaining: 493ms
    352:	learn: 0.3811094	total: 268ms	remaining: 492ms
    353:	learn: 0.3809488	total: 269ms	remaining: 491ms
    354:	learn: 0.3806283	total: 269ms	remaining: 489ms
    355:	learn: 0.3801071	total: 271ms	remaining: 490ms
    356:	learn: 0.3797252	total: 271ms	remaining: 489ms
    357:	learn: 0.3794454	total: 272ms	remaining: 488ms
    358:	learn: 0.3792391	total: 273ms	remaining: 487ms
    359:	learn: 0.3789255	total: 273ms	remaining: 486ms
    360:	learn: 0.3786394	total: 274ms	remaining: 484ms
    361:	learn: 0.3784338	total: 274ms	remaining: 483ms
    362:	learn: 0.3779165	total: 275ms	remaining: 482ms
    363:	learn: 0.3775870	total: 275ms	remaining: 481ms
    364:	learn: 0.3773097	total: 276ms	remaining: 480ms
    365:	learn: 0.3770937	total: 276ms	remaining: 479ms
    366:	learn: 0.3770125	total: 277ms	remaining: 478ms
    367:	learn: 0.3767403	total: 277ms	remaining: 476ms
    368:	learn: 0.3765635	total: 278ms	remaining: 476ms
    369:	learn: 0.3763529	total: 279ms	remaining: 475ms
    370:	learn: 0.3760306	total: 279ms	remaining: 473ms
    371:	learn: 0.3756096	total: 280ms	remaining: 472ms
    372:	learn: 0.3751607	total: 280ms	remaining: 471ms
    373:	learn: 0.3748620	total: 281ms	remaining: 470ms
    374:	learn: 0.3745725	total: 281ms	remaining: 469ms
    375:	learn: 0.3744518	total: 282ms	remaining: 468ms
    376:	learn: 0.3742739	total: 283ms	remaining: 467ms
    377:	learn: 0.3736971	total: 283ms	remaining: 466ms
    378:	learn: 0.3732856	total: 284ms	remaining: 465ms
    379:	learn: 0.3729109	total: 284ms	remaining: 464ms
    380:	learn: 0.3726707	total: 285ms	remaining: 463ms
    381:	learn: 0.3722342	total: 285ms	remaining: 462ms
    382:	learn: 0.3720231	total: 287ms	remaining: 462ms
    383:	learn: 0.3716705	total: 287ms	remaining: 461ms
    384:	learn: 0.3715034	total: 288ms	remaining: 460ms
    385:	learn: 0.3712607	total: 289ms	remaining: 459ms
    386:	learn: 0.3709507	total: 289ms	remaining: 458ms
    387:	learn: 0.3706835	total: 290ms	remaining: 457ms
    388:	learn: 0.3704587	total: 290ms	remaining: 456ms
    389:	learn: 0.3700011	total: 291ms	remaining: 455ms
    390:	learn: 0.3698790	total: 292ms	remaining: 454ms
    391:	learn: 0.3695983	total: 292ms	remaining: 453ms
    392:	learn: 0.3694231	total: 293ms	remaining: 452ms
    393:	learn: 0.3691177	total: 293ms	remaining: 451ms
    394:	learn: 0.3686477	total: 294ms	remaining: 450ms
    395:	learn: 0.3683968	total: 295ms	remaining: 449ms
    396:	learn: 0.3681956	total: 295ms	remaining: 448ms
    397:	learn: 0.3679874	total: 296ms	remaining: 447ms
    398:	learn: 0.3676807	total: 296ms	remaining: 446ms
    399:	learn: 0.3673719	total: 297ms	remaining: 445ms
    400:	learn: 0.3671734	total: 297ms	remaining: 444ms
    401:	learn: 0.3669163	total: 298ms	remaining: 443ms
    402:	learn: 0.3666791	total: 299ms	remaining: 443ms
    403:	learn: 0.3664502	total: 300ms	remaining: 442ms
    404:	learn: 0.3662576	total: 301ms	remaining: 442ms
    405:	learn: 0.3659043	total: 301ms	remaining: 441ms
    406:	learn: 0.3656904	total: 302ms	remaining: 440ms
    407:	learn: 0.3652823	total: 302ms	remaining: 439ms
    408:	learn: 0.3648772	total: 303ms	remaining: 438ms
    409:	learn: 0.3646252	total: 304ms	remaining: 437ms
    410:	learn: 0.3644239	total: 304ms	remaining: 436ms
    411:	learn: 0.3642585	total: 305ms	remaining: 435ms
    412:	learn: 0.3640206	total: 305ms	remaining: 434ms
    413:	learn: 0.3637309	total: 306ms	remaining: 433ms
    414:	learn: 0.3635911	total: 307ms	remaining: 432ms
    415:	learn: 0.3634973	total: 307ms	remaining: 431ms
    416:	learn: 0.3632448	total: 308ms	remaining: 430ms
    417:	learn: 0.3630737	total: 308ms	remaining: 429ms
    418:	learn: 0.3630006	total: 309ms	remaining: 428ms
    419:	learn: 0.3628806	total: 309ms	remaining: 427ms
    420:	learn: 0.3627103	total: 310ms	remaining: 426ms
    421:	learn: 0.3625253	total: 311ms	remaining: 425ms
    422:	learn: 0.3622061	total: 311ms	remaining: 424ms
    423:	learn: 0.3620192	total: 312ms	remaining: 424ms
    424:	learn: 0.3616419	total: 312ms	remaining: 423ms
    425:	learn: 0.3613792	total: 313ms	remaining: 422ms
    426:	learn: 0.3611632	total: 313ms	remaining: 421ms
    427:	learn: 0.3609203	total: 314ms	remaining: 420ms
    428:	learn: 0.3606681	total: 315ms	remaining: 420ms
    429:	learn: 0.3605024	total: 316ms	remaining: 419ms
    430:	learn: 0.3603780	total: 316ms	remaining: 418ms
    431:	learn: 0.3602885	total: 317ms	remaining: 417ms
    432:	learn: 0.3597736	total: 318ms	remaining: 416ms
    433:	learn: 0.3595863	total: 318ms	remaining: 415ms
    434:	learn: 0.3594723	total: 319ms	remaining: 414ms
    435:	learn: 0.3591463	total: 319ms	remaining: 413ms
    436:	learn: 0.3590245	total: 320ms	remaining: 412ms
    437:	learn: 0.3587814	total: 320ms	remaining: 411ms
    438:	learn: 0.3584349	total: 321ms	remaining: 410ms
    439:	learn: 0.3581677	total: 322ms	remaining: 409ms
    440:	learn: 0.3580575	total: 322ms	remaining: 408ms
    441:	learn: 0.3577329	total: 323ms	remaining: 407ms
    442:	learn: 0.3574741	total: 323ms	remaining: 407ms
    443:	learn: 0.3572658	total: 324ms	remaining: 406ms
    444:	learn: 0.3570792	total: 325ms	remaining: 405ms
    445:	learn: 0.3568196	total: 325ms	remaining: 404ms
    446:	learn: 0.3566210	total: 326ms	remaining: 403ms
    447:	learn: 0.3561632	total: 326ms	remaining: 402ms
    448:	learn: 0.3556776	total: 327ms	remaining: 401ms
    449:	learn: 0.3555563	total: 327ms	remaining: 400ms
    450:	learn: 0.3550309	total: 328ms	remaining: 399ms
    451:	learn: 0.3547222	total: 329ms	remaining: 398ms
    452:	learn: 0.3546797	total: 329ms	remaining: 397ms
    453:	learn: 0.3542806	total: 329ms	remaining: 396ms
    454:	learn: 0.3540600	total: 330ms	remaining: 395ms
    455:	learn: 0.3538005	total: 331ms	remaining: 394ms
    456:	learn: 0.3534230	total: 331ms	remaining: 393ms
    457:	learn: 0.3533270	total: 332ms	remaining: 392ms
    458:	learn: 0.3530345	total: 333ms	remaining: 392ms
    459:	learn: 0.3526667	total: 333ms	remaining: 391ms
    460:	learn: 0.3523433	total: 334ms	remaining: 391ms
    461:	learn: 0.3521075	total: 335ms	remaining: 390ms
    462:	learn: 0.3519575	total: 335ms	remaining: 389ms
    463:	learn: 0.3516387	total: 336ms	remaining: 388ms
    464:	learn: 0.3516359	total: 336ms	remaining: 387ms
    465:	learn: 0.3513703	total: 337ms	remaining: 386ms
    466:	learn: 0.3510449	total: 337ms	remaining: 385ms
    467:	learn: 0.3509308	total: 338ms	remaining: 384ms
    468:	learn: 0.3504502	total: 338ms	remaining: 383ms
    469:	learn: 0.3498691	total: 339ms	remaining: 382ms
    470:	learn: 0.3495844	total: 339ms	remaining: 381ms
    471:	learn: 0.3494299	total: 340ms	remaining: 380ms
    472:	learn: 0.3492809	total: 340ms	remaining: 379ms
    473:	learn: 0.3491325	total: 341ms	remaining: 379ms
    474:	learn: 0.3489996	total: 342ms	remaining: 378ms
    475:	learn: 0.3487294	total: 342ms	remaining: 377ms
    476:	learn: 0.3484807	total: 343ms	remaining: 376ms
    477:	learn: 0.3482077	total: 343ms	remaining: 375ms
    478:	learn: 0.3479059	total: 344ms	remaining: 374ms
    479:	learn: 0.3475452	total: 344ms	remaining: 373ms
    480:	learn: 0.3472249	total: 345ms	remaining: 372ms
    481:	learn: 0.3470146	total: 345ms	remaining: 371ms
    482:	learn: 0.3468738	total: 346ms	remaining: 370ms
    483:	learn: 0.3466086	total: 347ms	remaining: 369ms
    484:	learn: 0.3462096	total: 347ms	remaining: 369ms
    485:	learn: 0.3459638	total: 348ms	remaining: 368ms
    486:	learn: 0.3456568	total: 348ms	remaining: 367ms
    487:	learn: 0.3451945	total: 349ms	remaining: 366ms
    488:	learn: 0.3450585	total: 349ms	remaining: 365ms
    489:	learn: 0.3448107	total: 350ms	remaining: 364ms
    490:	learn: 0.3447375	total: 351ms	remaining: 363ms
    491:	learn: 0.3445321	total: 351ms	remaining: 362ms
    492:	learn: 0.3442272	total: 352ms	remaining: 362ms
    493:	learn: 0.3439775	total: 352ms	remaining: 361ms
    494:	learn: 0.3435719	total: 353ms	remaining: 360ms
    495:	learn: 0.3433479	total: 353ms	remaining: 359ms
    496:	learn: 0.3431404	total: 354ms	remaining: 358ms
    497:	learn: 0.3427044	total: 355ms	remaining: 358ms
    498:	learn: 0.3425609	total: 355ms	remaining: 357ms
    499:	learn: 0.3424097	total: 356ms	remaining: 356ms
    500:	learn: 0.3421952	total: 357ms	remaining: 355ms
    501:	learn: 0.3416515	total: 357ms	remaining: 354ms
    502:	learn: 0.3415737	total: 358ms	remaining: 353ms
    503:	learn: 0.3413161	total: 358ms	remaining: 353ms
    504:	learn: 0.3411923	total: 359ms	remaining: 352ms
    505:	learn: 0.3410760	total: 359ms	remaining: 351ms
    506:	learn: 0.3409117	total: 360ms	remaining: 350ms
    507:	learn: 0.3406540	total: 360ms	remaining: 349ms
    508:	learn: 0.3403382	total: 361ms	remaining: 348ms
    509:	learn: 0.3399882	total: 362ms	remaining: 347ms
    510:	learn: 0.3398181	total: 362ms	remaining: 347ms
    511:	learn: 0.3392933	total: 363ms	remaining: 346ms
    512:	learn: 0.3392422	total: 363ms	remaining: 345ms
    513:	learn: 0.3387441	total: 364ms	remaining: 344ms
    514:	learn: 0.3383355	total: 364ms	remaining: 343ms
    515:	learn: 0.3378527	total: 365ms	remaining: 342ms
    516:	learn: 0.3376871	total: 365ms	remaining: 341ms
    517:	learn: 0.3374242	total: 366ms	remaining: 341ms
    518:	learn: 0.3371251	total: 366ms	remaining: 340ms
    519:	learn: 0.3369829	total: 367ms	remaining: 339ms
    520:	learn: 0.3367569	total: 368ms	remaining: 338ms
    521:	learn: 0.3362999	total: 368ms	remaining: 337ms
    522:	learn: 0.3359526	total: 369ms	remaining: 336ms
    523:	learn: 0.3358340	total: 369ms	remaining: 335ms
    524:	learn: 0.3356242	total: 370ms	remaining: 335ms
    525:	learn: 0.3355150	total: 371ms	remaining: 334ms
    526:	learn: 0.3353782	total: 372ms	remaining: 333ms
    527:	learn: 0.3350626	total: 372ms	remaining: 333ms
    528:	learn: 0.3348814	total: 373ms	remaining: 332ms
    529:	learn: 0.3346845	total: 373ms	remaining: 331ms
    530:	learn: 0.3343420	total: 374ms	remaining: 330ms
    531:	learn: 0.3339803	total: 374ms	remaining: 329ms
    532:	learn: 0.3336009	total: 375ms	remaining: 329ms
    533:	learn: 0.3332257	total: 376ms	remaining: 328ms
    534:	learn: 0.3330515	total: 376ms	remaining: 327ms
    535:	learn: 0.3327725	total: 377ms	remaining: 326ms
    536:	learn: 0.3326002	total: 377ms	remaining: 325ms
    537:	learn: 0.3323371	total: 378ms	remaining: 325ms
    538:	learn: 0.3320422	total: 379ms	remaining: 324ms
    539:	learn: 0.3318282	total: 379ms	remaining: 323ms
    540:	learn: 0.3316494	total: 380ms	remaining: 322ms
    541:	learn: 0.3314210	total: 381ms	remaining: 322ms
    542:	learn: 0.3313446	total: 382ms	remaining: 321ms
    543:	learn: 0.3312452	total: 382ms	remaining: 320ms
    544:	learn: 0.3306939	total: 383ms	remaining: 320ms
    545:	learn: 0.3305628	total: 383ms	remaining: 319ms
    546:	learn: 0.3303154	total: 384ms	remaining: 318ms
    547:	learn: 0.3299730	total: 384ms	remaining: 317ms
    548:	learn: 0.3297925	total: 385ms	remaining: 316ms
    549:	learn: 0.3296933	total: 386ms	remaining: 315ms
    550:	learn: 0.3293662	total: 386ms	remaining: 315ms
    551:	learn: 0.3291773	total: 387ms	remaining: 314ms
    552:	learn: 0.3289285	total: 387ms	remaining: 313ms
    553:	learn: 0.3288177	total: 388ms	remaining: 312ms
    554:	learn: 0.3286785	total: 388ms	remaining: 311ms
    555:	learn: 0.3282669	total: 389ms	remaining: 311ms
    556:	learn: 0.3280586	total: 390ms	remaining: 310ms
    557:	learn: 0.3279200	total: 390ms	remaining: 309ms
    558:	learn: 0.3276552	total: 391ms	remaining: 308ms
    559:	learn: 0.3273689	total: 391ms	remaining: 307ms
    560:	learn: 0.3271314	total: 392ms	remaining: 307ms
    561:	learn: 0.3270011	total: 392ms	remaining: 306ms
    562:	learn: 0.3268395	total: 393ms	remaining: 305ms
    563:	learn: 0.3266444	total: 393ms	remaining: 304ms
    564:	learn: 0.3264135	total: 394ms	remaining: 303ms
    565:	learn: 0.3262431	total: 395ms	remaining: 303ms
    566:	learn: 0.3259624	total: 395ms	remaining: 302ms
    567:	learn: 0.3258445	total: 396ms	remaining: 301ms
    568:	learn: 0.3256464	total: 396ms	remaining: 300ms
    569:	learn: 0.3254283	total: 397ms	remaining: 299ms
    570:	learn: 0.3253433	total: 397ms	remaining: 299ms
    571:	learn: 0.3252649	total: 398ms	remaining: 298ms
    572:	learn: 0.3250043	total: 398ms	remaining: 297ms
    573:	learn: 0.3246208	total: 399ms	remaining: 296ms
    574:	learn: 0.3240812	total: 400ms	remaining: 295ms
    575:	learn: 0.3240041	total: 400ms	remaining: 295ms
    576:	learn: 0.3236784	total: 401ms	remaining: 294ms
    577:	learn: 0.3234004	total: 401ms	remaining: 293ms
    578:	learn: 0.3229345	total: 402ms	remaining: 292ms
    579:	learn: 0.3226420	total: 403ms	remaining: 291ms
    580:	learn: 0.3223880	total: 403ms	remaining: 291ms
    581:	learn: 0.3221717	total: 404ms	remaining: 290ms
    582:	learn: 0.3217649	total: 404ms	remaining: 289ms
    583:	learn: 0.3215319	total: 405ms	remaining: 288ms
    584:	learn: 0.3214213	total: 405ms	remaining: 288ms
    585:	learn: 0.3212897	total: 406ms	remaining: 287ms
    586:	learn: 0.3211508	total: 407ms	remaining: 286ms
    587:	learn: 0.3209801	total: 407ms	remaining: 285ms
    588:	learn: 0.3207408	total: 408ms	remaining: 284ms
    589:	learn: 0.3204701	total: 408ms	remaining: 284ms
    590:	learn: 0.3203842	total: 409ms	remaining: 283ms
    591:	learn: 0.3201113	total: 409ms	remaining: 282ms
    592:	learn: 0.3199417	total: 410ms	remaining: 281ms
    593:	learn: 0.3196526	total: 410ms	remaining: 281ms
    594:	learn: 0.3192808	total: 411ms	remaining: 280ms
    595:	learn: 0.3189753	total: 412ms	remaining: 279ms
    596:	learn: 0.3188734	total: 412ms	remaining: 278ms
    597:	learn: 0.3186580	total: 413ms	remaining: 278ms
    598:	learn: 0.3185433	total: 413ms	remaining: 277ms
    599:	learn: 0.3184695	total: 414ms	remaining: 276ms
    600:	learn: 0.3183482	total: 414ms	remaining: 275ms
    601:	learn: 0.3181125	total: 415ms	remaining: 274ms
    602:	learn: 0.3178000	total: 416ms	remaining: 274ms
    603:	learn: 0.3173890	total: 416ms	remaining: 273ms
    604:	learn: 0.3171024	total: 417ms	remaining: 272ms
    605:	learn: 0.3169082	total: 417ms	remaining: 271ms
    606:	learn: 0.3166991	total: 418ms	remaining: 271ms
    607:	learn: 0.3164783	total: 419ms	remaining: 270ms
    608:	learn: 0.3161274	total: 419ms	remaining: 269ms
    609:	learn: 0.3159356	total: 420ms	remaining: 269ms
    610:	learn: 0.3154450	total: 421ms	remaining: 268ms
    611:	learn: 0.3152461	total: 421ms	remaining: 267ms
    612:	learn: 0.3150069	total: 422ms	remaining: 266ms
    613:	learn: 0.3147977	total: 422ms	remaining: 266ms
    614:	learn: 0.3147408	total: 423ms	remaining: 265ms
    615:	learn: 0.3145949	total: 423ms	remaining: 264ms
    616:	learn: 0.3143235	total: 424ms	remaining: 263ms
    617:	learn: 0.3139006	total: 425ms	remaining: 262ms
    618:	learn: 0.3137003	total: 425ms	remaining: 262ms
    619:	learn: 0.3134818	total: 426ms	remaining: 261ms
    620:	learn: 0.3133102	total: 426ms	remaining: 260ms
    621:	learn: 0.3130404	total: 427ms	remaining: 259ms
    622:	learn: 0.3128168	total: 428ms	remaining: 259ms
    623:	learn: 0.3126890	total: 428ms	remaining: 258ms
    624:	learn: 0.3126103	total: 429ms	remaining: 257ms
    625:	learn: 0.3123177	total: 429ms	remaining: 257ms
    626:	learn: 0.3122126	total: 430ms	remaining: 256ms
    627:	learn: 0.3120781	total: 431ms	remaining: 255ms
    628:	learn: 0.3118757	total: 431ms	remaining: 254ms
    629:	learn: 0.3116607	total: 432ms	remaining: 254ms
    630:	learn: 0.3113584	total: 432ms	remaining: 253ms
    631:	learn: 0.3110870	total: 433ms	remaining: 252ms
    632:	learn: 0.3107902	total: 433ms	remaining: 251ms
    633:	learn: 0.3105543	total: 434ms	remaining: 251ms
    634:	learn: 0.3102969	total: 435ms	remaining: 250ms
    635:	learn: 0.3102147	total: 435ms	remaining: 249ms
    636:	learn: 0.3100398	total: 436ms	remaining: 248ms
    637:	learn: 0.3100245	total: 436ms	remaining: 248ms
    638:	learn: 0.3099558	total: 437ms	remaining: 247ms
    639:	learn: 0.3097149	total: 437ms	remaining: 246ms
    640:	learn: 0.3095803	total: 438ms	remaining: 245ms
    641:	learn: 0.3093793	total: 439ms	remaining: 245ms
    642:	learn: 0.3091070	total: 439ms	remaining: 244ms
    643:	learn: 0.3087943	total: 440ms	remaining: 243ms
    644:	learn: 0.3084907	total: 440ms	remaining: 242ms
    645:	learn: 0.3082264	total: 441ms	remaining: 242ms
    646:	learn: 0.3078663	total: 442ms	remaining: 241ms
    647:	learn: 0.3077548	total: 442ms	remaining: 240ms
    648:	learn: 0.3075767	total: 443ms	remaining: 240ms
    649:	learn: 0.3074493	total: 444ms	remaining: 239ms
    650:	learn: 0.3073495	total: 444ms	remaining: 238ms
    651:	learn: 0.3072290	total: 445ms	remaining: 237ms
    652:	learn: 0.3070451	total: 445ms	remaining: 237ms
    653:	learn: 0.3068079	total: 446ms	remaining: 236ms
    654:	learn: 0.3064931	total: 446ms	remaining: 235ms
    655:	learn: 0.3061377	total: 447ms	remaining: 234ms
    656:	learn: 0.3059533	total: 447ms	remaining: 234ms
    657:	learn: 0.3058613	total: 448ms	remaining: 233ms
    658:	learn: 0.3057576	total: 449ms	remaining: 232ms
    659:	learn: 0.3055123	total: 449ms	remaining: 231ms
    660:	learn: 0.3053269	total: 450ms	remaining: 231ms
    661:	learn: 0.3051952	total: 450ms	remaining: 230ms
    662:	learn: 0.3049991	total: 451ms	remaining: 229ms
    663:	learn: 0.3047315	total: 452ms	remaining: 228ms
    664:	learn: 0.3044283	total: 452ms	remaining: 228ms
    665:	learn: 0.3041486	total: 453ms	remaining: 227ms
    666:	learn: 0.3037296	total: 453ms	remaining: 226ms
    667:	learn: 0.3035647	total: 454ms	remaining: 226ms
    668:	learn: 0.3033983	total: 454ms	remaining: 225ms
    669:	learn: 0.3030367	total: 455ms	remaining: 224ms
    670:	learn: 0.3028264	total: 456ms	remaining: 224ms
    671:	learn: 0.3027669	total: 457ms	remaining: 223ms
    672:	learn: 0.3026263	total: 457ms	remaining: 222ms
    673:	learn: 0.3025193	total: 458ms	remaining: 221ms
    674:	learn: 0.3020778	total: 458ms	remaining: 221ms
    675:	learn: 0.3019976	total: 459ms	remaining: 220ms
    676:	learn: 0.3018234	total: 460ms	remaining: 219ms
    677:	learn: 0.3016789	total: 460ms	remaining: 219ms
    678:	learn: 0.3015404	total: 461ms	remaining: 218ms
    679:	learn: 0.3011528	total: 461ms	remaining: 217ms
    680:	learn: 0.3010274	total: 462ms	remaining: 216ms
    681:	learn: 0.3007603	total: 463ms	remaining: 216ms
    682:	learn: 0.3005135	total: 463ms	remaining: 215ms
    683:	learn: 0.3001911	total: 464ms	remaining: 214ms
    684:	learn: 0.3000974	total: 465ms	remaining: 214ms
    685:	learn: 0.2999670	total: 465ms	remaining: 213ms
    686:	learn: 0.2998363	total: 466ms	remaining: 212ms
    687:	learn: 0.2994004	total: 466ms	remaining: 211ms
    688:	learn: 0.2992484	total: 467ms	remaining: 211ms
    689:	learn: 0.2991263	total: 468ms	remaining: 210ms
    690:	learn: 0.2987575	total: 468ms	remaining: 209ms
    691:	learn: 0.2986268	total: 469ms	remaining: 209ms
    692:	learn: 0.2985420	total: 469ms	remaining: 208ms
    693:	learn: 0.2984095	total: 470ms	remaining: 207ms
    694:	learn: 0.2981957	total: 471ms	remaining: 207ms
    695:	learn: 0.2978786	total: 472ms	remaining: 206ms
    696:	learn: 0.2974858	total: 472ms	remaining: 205ms
    697:	learn: 0.2972561	total: 473ms	remaining: 205ms
    698:	learn: 0.2971596	total: 473ms	remaining: 204ms
    699:	learn: 0.2968434	total: 474ms	remaining: 203ms
    700:	learn: 0.2965182	total: 475ms	remaining: 202ms
    701:	learn: 0.2963988	total: 475ms	remaining: 202ms
    702:	learn: 0.2961760	total: 477ms	remaining: 202ms
    703:	learn: 0.2958515	total: 478ms	remaining: 201ms
    704:	learn: 0.2955903	total: 479ms	remaining: 200ms
    705:	learn: 0.2955277	total: 479ms	remaining: 200ms
    706:	learn: 0.2952251	total: 480ms	remaining: 199ms
    707:	learn: 0.2950762	total: 481ms	remaining: 198ms
    708:	learn: 0.2949414	total: 481ms	remaining: 197ms
    709:	learn: 0.2946175	total: 482ms	remaining: 197ms
    710:	learn: 0.2943531	total: 482ms	remaining: 196ms
    711:	learn: 0.2943172	total: 483ms	remaining: 195ms
    712:	learn: 0.2941421	total: 483ms	remaining: 195ms
    713:	learn: 0.2937829	total: 484ms	remaining: 194ms
    714:	learn: 0.2935082	total: 484ms	remaining: 193ms
    715:	learn: 0.2933580	total: 485ms	remaining: 192ms
    716:	learn: 0.2930393	total: 486ms	remaining: 192ms
    717:	learn: 0.2929175	total: 486ms	remaining: 191ms
    718:	learn: 0.2925147	total: 487ms	remaining: 190ms
    719:	learn: 0.2922763	total: 487ms	remaining: 190ms
    720:	learn: 0.2919918	total: 488ms	remaining: 189ms
    721:	learn: 0.2919215	total: 488ms	remaining: 188ms
    722:	learn: 0.2916168	total: 489ms	remaining: 187ms
    723:	learn: 0.2914495	total: 490ms	remaining: 187ms
    724:	learn: 0.2912879	total: 490ms	remaining: 186ms
    725:	learn: 0.2908818	total: 492ms	remaining: 186ms
    726:	learn: 0.2907400	total: 492ms	remaining: 185ms
    727:	learn: 0.2906563	total: 493ms	remaining: 184ms
    728:	learn: 0.2904603	total: 494ms	remaining: 184ms
    729:	learn: 0.2902665	total: 494ms	remaining: 183ms
    730:	learn: 0.2898631	total: 495ms	remaining: 182ms
    731:	learn: 0.2894571	total: 495ms	remaining: 181ms
    732:	learn: 0.2892670	total: 496ms	remaining: 181ms
    733:	learn: 0.2890068	total: 496ms	remaining: 180ms
    734:	learn: 0.2888406	total: 497ms	remaining: 179ms
    735:	learn: 0.2887034	total: 498ms	remaining: 179ms
    736:	learn: 0.2884606	total: 498ms	remaining: 178ms
    737:	learn: 0.2880213	total: 499ms	remaining: 177ms
    738:	learn: 0.2877335	total: 499ms	remaining: 176ms
    739:	learn: 0.2875847	total: 500ms	remaining: 176ms
    740:	learn: 0.2873342	total: 501ms	remaining: 175ms
    741:	learn: 0.2870388	total: 501ms	remaining: 174ms
    742:	learn: 0.2868430	total: 502ms	remaining: 174ms
    743:	learn: 0.2865782	total: 502ms	remaining: 173ms
    744:	learn: 0.2864024	total: 503ms	remaining: 172ms
    745:	learn: 0.2862998	total: 503ms	remaining: 171ms
    746:	learn: 0.2860246	total: 504ms	remaining: 171ms
    747:	learn: 0.2858331	total: 504ms	remaining: 170ms
    748:	learn: 0.2856774	total: 505ms	remaining: 169ms
    749:	learn: 0.2854794	total: 506ms	remaining: 169ms
    750:	learn: 0.2853490	total: 506ms	remaining: 168ms
    751:	learn: 0.2852392	total: 507ms	remaining: 167ms
    752:	learn: 0.2849235	total: 507ms	remaining: 166ms
    753:	learn: 0.2846747	total: 508ms	remaining: 166ms
    754:	learn: 0.2845416	total: 508ms	remaining: 165ms
    755:	learn: 0.2844480	total: 509ms	remaining: 164ms
    756:	learn: 0.2842077	total: 510ms	remaining: 164ms
    757:	learn: 0.2839573	total: 510ms	remaining: 163ms
    758:	learn: 0.2835397	total: 511ms	remaining: 162ms
    759:	learn: 0.2834014	total: 511ms	remaining: 161ms
    760:	learn: 0.2829780	total: 512ms	remaining: 161ms
    761:	learn: 0.2828479	total: 512ms	remaining: 160ms
    762:	learn: 0.2824156	total: 513ms	remaining: 159ms
    763:	learn: 0.2820805	total: 514ms	remaining: 159ms
    764:	learn: 0.2820193	total: 514ms	remaining: 158ms
    765:	learn: 0.2818557	total: 515ms	remaining: 157ms
    766:	learn: 0.2816069	total: 515ms	remaining: 157ms
    767:	learn: 0.2813983	total: 516ms	remaining: 156ms
    768:	learn: 0.2810394	total: 516ms	remaining: 155ms
    769:	learn: 0.2806972	total: 517ms	remaining: 154ms
    770:	learn: 0.2804703	total: 517ms	remaining: 154ms
    771:	learn: 0.2802344	total: 518ms	remaining: 153ms
    772:	learn: 0.2801140	total: 519ms	remaining: 152ms
    773:	learn: 0.2799247	total: 519ms	remaining: 152ms
    774:	learn: 0.2796614	total: 520ms	remaining: 151ms
    775:	learn: 0.2791635	total: 520ms	remaining: 150ms
    776:	learn: 0.2788412	total: 521ms	remaining: 149ms
    777:	learn: 0.2786401	total: 521ms	remaining: 149ms
    778:	learn: 0.2783673	total: 522ms	remaining: 148ms
    779:	learn: 0.2779244	total: 522ms	remaining: 147ms
    780:	learn: 0.2778506	total: 523ms	remaining: 147ms
    781:	learn: 0.2777733	total: 524ms	remaining: 146ms
    782:	learn: 0.2776055	total: 524ms	remaining: 145ms
    783:	learn: 0.2775094	total: 525ms	remaining: 145ms
    784:	learn: 0.2772731	total: 525ms	remaining: 144ms
    785:	learn: 0.2771009	total: 526ms	remaining: 143ms
    786:	learn: 0.2767214	total: 527ms	remaining: 143ms
    787:	learn: 0.2766579	total: 527ms	remaining: 142ms
    788:	learn: 0.2764208	total: 528ms	remaining: 141ms
    789:	learn: 0.2762727	total: 528ms	remaining: 140ms
    790:	learn: 0.2759174	total: 529ms	remaining: 140ms
    791:	learn: 0.2757582	total: 529ms	remaining: 139ms
    792:	learn: 0.2756208	total: 530ms	remaining: 138ms
    793:	learn: 0.2752521	total: 530ms	remaining: 138ms
    794:	learn: 0.2749698	total: 531ms	remaining: 137ms
    795:	learn: 0.2746988	total: 531ms	remaining: 136ms
    796:	learn: 0.2744096	total: 532ms	remaining: 136ms
    797:	learn: 0.2741457	total: 533ms	remaining: 135ms
    798:	learn: 0.2740050	total: 533ms	remaining: 134ms
    799:	learn: 0.2737424	total: 534ms	remaining: 133ms
    800:	learn: 0.2734012	total: 535ms	remaining: 133ms
    801:	learn: 0.2732058	total: 535ms	remaining: 132ms
    802:	learn: 0.2730744	total: 536ms	remaining: 131ms
    803:	learn: 0.2727751	total: 536ms	remaining: 131ms
    804:	learn: 0.2727277	total: 537ms	remaining: 130ms
    805:	learn: 0.2726383	total: 537ms	remaining: 129ms
    806:	learn: 0.2723608	total: 538ms	remaining: 129ms
    807:	learn: 0.2719860	total: 538ms	remaining: 128ms
    808:	learn: 0.2719069	total: 539ms	remaining: 127ms
    809:	learn: 0.2717517	total: 540ms	remaining: 127ms
    810:	learn: 0.2715737	total: 540ms	remaining: 126ms
    811:	learn: 0.2715296	total: 541ms	remaining: 125ms
    812:	learn: 0.2714914	total: 541ms	remaining: 124ms
    813:	learn: 0.2713306	total: 542ms	remaining: 124ms
    814:	learn: 0.2712960	total: 542ms	remaining: 123ms
    815:	learn: 0.2709066	total: 543ms	remaining: 122ms
    816:	learn: 0.2708333	total: 544ms	remaining: 122ms
    817:	learn: 0.2703798	total: 545ms	remaining: 121ms
    818:	learn: 0.2701172	total: 545ms	remaining: 120ms
    819:	learn: 0.2696609	total: 546ms	remaining: 120ms
    820:	learn: 0.2695340	total: 546ms	remaining: 119ms
    821:	learn: 0.2692294	total: 547ms	remaining: 118ms
    822:	learn: 0.2690227	total: 548ms	remaining: 118ms
    823:	learn: 0.2689331	total: 548ms	remaining: 117ms
    824:	learn: 0.2685256	total: 549ms	remaining: 116ms
    825:	learn: 0.2684674	total: 549ms	remaining: 116ms
    826:	learn: 0.2683949	total: 550ms	remaining: 115ms
    827:	learn: 0.2681845	total: 550ms	remaining: 114ms
    828:	learn: 0.2679269	total: 551ms	remaining: 114ms
    829:	learn: 0.2676696	total: 551ms	remaining: 113ms
    830:	learn: 0.2675692	total: 552ms	remaining: 112ms
    831:	learn: 0.2673769	total: 553ms	remaining: 112ms
    832:	learn: 0.2672776	total: 553ms	remaining: 111ms
    833:	learn: 0.2670370	total: 554ms	remaining: 110ms
    834:	learn: 0.2667558	total: 554ms	remaining: 110ms
    835:	learn: 0.2664869	total: 555ms	remaining: 109ms
    836:	learn: 0.2663765	total: 555ms	remaining: 108ms
    837:	learn: 0.2659908	total: 556ms	remaining: 108ms
    838:	learn: 0.2659319	total: 557ms	remaining: 107ms
    839:	learn: 0.2657949	total: 557ms	remaining: 106ms
    840:	learn: 0.2655954	total: 558ms	remaining: 105ms
    841:	learn: 0.2654235	total: 558ms	remaining: 105ms
    842:	learn: 0.2652815	total: 559ms	remaining: 104ms
    843:	learn: 0.2650946	total: 559ms	remaining: 103ms
    844:	learn: 0.2649355	total: 560ms	remaining: 103ms
    845:	learn: 0.2646571	total: 560ms	remaining: 102ms
    846:	learn: 0.2641645	total: 561ms	remaining: 101ms
    847:	learn: 0.2640117	total: 562ms	remaining: 101ms
    848:	learn: 0.2637970	total: 562ms	remaining: 100ms
    849:	learn: 0.2633991	total: 563ms	remaining: 99.3ms
    850:	learn: 0.2630427	total: 563ms	remaining: 98.6ms
    851:	learn: 0.2628822	total: 564ms	remaining: 97.9ms
    852:	learn: 0.2627623	total: 564ms	remaining: 97.3ms
    853:	learn: 0.2624961	total: 565ms	remaining: 96.6ms
    854:	learn: 0.2622119	total: 566ms	remaining: 95.9ms
    855:	learn: 0.2621805	total: 566ms	remaining: 95.2ms
    856:	learn: 0.2619224	total: 567ms	remaining: 94.6ms
    857:	learn: 0.2617404	total: 567ms	remaining: 93.9ms
    858:	learn: 0.2613684	total: 568ms	remaining: 93.2ms
    859:	learn: 0.2612215	total: 569ms	remaining: 92.5ms
    860:	learn: 0.2610357	total: 569ms	remaining: 91.9ms
    861:	learn: 0.2608420	total: 570ms	remaining: 91.2ms
    862:	learn: 0.2605551	total: 570ms	remaining: 90.5ms
    863:	learn: 0.2602325	total: 571ms	remaining: 89.8ms
    864:	learn: 0.2600269	total: 571ms	remaining: 89.2ms
    865:	learn: 0.2595908	total: 572ms	remaining: 88.5ms
    866:	learn: 0.2592418	total: 573ms	remaining: 87.8ms
    867:	learn: 0.2592055	total: 573ms	remaining: 87.2ms
    868:	learn: 0.2588707	total: 574ms	remaining: 86.5ms
    869:	learn: 0.2586619	total: 574ms	remaining: 85.8ms
    870:	learn: 0.2584166	total: 575ms	remaining: 85.1ms
    871:	learn: 0.2581188	total: 575ms	remaining: 84.4ms
    872:	learn: 0.2579307	total: 576ms	remaining: 83.8ms
    873:	learn: 0.2576076	total: 577ms	remaining: 83.1ms
    874:	learn: 0.2574419	total: 577ms	remaining: 82.5ms
    875:	learn: 0.2573296	total: 578ms	remaining: 81.8ms
    876:	learn: 0.2571506	total: 578ms	remaining: 81.1ms
    877:	learn: 0.2569876	total: 579ms	remaining: 80.4ms
    878:	learn: 0.2568477	total: 579ms	remaining: 79.7ms
    879:	learn: 0.2565870	total: 580ms	remaining: 79.1ms
    880:	learn: 0.2562756	total: 580ms	remaining: 78.4ms
    881:	learn: 0.2560758	total: 581ms	remaining: 77.7ms
    882:	learn: 0.2557248	total: 581ms	remaining: 77ms
    883:	learn: 0.2554332	total: 582ms	remaining: 76.4ms
    884:	learn: 0.2552059	total: 583ms	remaining: 75.7ms
    885:	learn: 0.2551408	total: 583ms	remaining: 75ms
    886:	learn: 0.2550661	total: 583ms	remaining: 74.3ms
    887:	learn: 0.2548069	total: 584ms	remaining: 73.7ms
    888:	learn: 0.2545537	total: 585ms	remaining: 73ms
    889:	learn: 0.2544509	total: 585ms	remaining: 72.3ms
    890:	learn: 0.2542445	total: 586ms	remaining: 71.6ms
    891:	learn: 0.2540550	total: 586ms	remaining: 71ms
    892:	learn: 0.2539043	total: 587ms	remaining: 70.3ms
    893:	learn: 0.2536143	total: 587ms	remaining: 69.6ms
    894:	learn: 0.2534578	total: 588ms	remaining: 69ms
    895:	learn: 0.2532457	total: 589ms	remaining: 68.3ms
    896:	learn: 0.2530981	total: 589ms	remaining: 67.6ms
    897:	learn: 0.2528791	total: 590ms	remaining: 67ms
    898:	learn: 0.2525524	total: 590ms	remaining: 66.3ms
    899:	learn: 0.2524827	total: 591ms	remaining: 65.6ms
    900:	learn: 0.2521116	total: 591ms	remaining: 65ms
    901:	learn: 0.2519686	total: 592ms	remaining: 64.3ms
    902:	learn: 0.2517455	total: 592ms	remaining: 63.6ms
    903:	learn: 0.2515105	total: 593ms	remaining: 63ms
    904:	learn: 0.2512048	total: 594ms	remaining: 62.3ms
    905:	learn: 0.2510956	total: 594ms	remaining: 61.6ms
    906:	learn: 0.2509242	total: 595ms	remaining: 61ms
    907:	learn: 0.2507175	total: 595ms	remaining: 60.3ms
    908:	learn: 0.2506682	total: 596ms	remaining: 59.7ms
    909:	learn: 0.2505012	total: 597ms	remaining: 59ms
    910:	learn: 0.2501784	total: 597ms	remaining: 58.3ms
    911:	learn: 0.2498398	total: 598ms	remaining: 57.7ms
    912:	learn: 0.2497002	total: 598ms	remaining: 57ms
    913:	learn: 0.2494249	total: 599ms	remaining: 56.3ms
    914:	learn: 0.2493515	total: 599ms	remaining: 55.7ms
    915:	learn: 0.2491348	total: 600ms	remaining: 55ms
    916:	learn: 0.2488985	total: 601ms	remaining: 54.4ms
    917:	learn: 0.2487173	total: 601ms	remaining: 53.7ms
    918:	learn: 0.2485863	total: 602ms	remaining: 53ms
    919:	learn: 0.2481821	total: 602ms	remaining: 52.4ms
    920:	learn: 0.2481395	total: 603ms	remaining: 51.7ms
    921:	learn: 0.2480246	total: 603ms	remaining: 51ms
    922:	learn: 0.2478766	total: 604ms	remaining: 50.4ms
    923:	learn: 0.2476939	total: 604ms	remaining: 49.7ms
    924:	learn: 0.2475361	total: 606ms	remaining: 49.1ms
    925:	learn: 0.2470830	total: 606ms	remaining: 48.4ms
    926:	learn: 0.2468226	total: 607ms	remaining: 47.8ms
    927:	learn: 0.2465208	total: 607ms	remaining: 47.1ms
    928:	learn: 0.2462846	total: 608ms	remaining: 46.5ms
    929:	learn: 0.2461631	total: 608ms	remaining: 45.8ms
    930:	learn: 0.2458885	total: 609ms	remaining: 45.1ms
    931:	learn: 0.2456278	total: 610ms	remaining: 44.5ms
    932:	learn: 0.2455066	total: 610ms	remaining: 43.8ms
    933:	learn: 0.2453706	total: 611ms	remaining: 43.1ms
    934:	learn: 0.2451656	total: 611ms	remaining: 42.5ms
    935:	learn: 0.2450755	total: 612ms	remaining: 41.8ms
    936:	learn: 0.2449028	total: 612ms	remaining: 41.2ms
    937:	learn: 0.2445005	total: 613ms	remaining: 40.5ms
    938:	learn: 0.2444332	total: 614ms	remaining: 39.9ms
    939:	learn: 0.2441924	total: 614ms	remaining: 39.2ms
    940:	learn: 0.2438373	total: 615ms	remaining: 38.5ms
    941:	learn: 0.2435087	total: 615ms	remaining: 37.9ms
    942:	learn: 0.2433194	total: 616ms	remaining: 37.2ms
    943:	learn: 0.2432692	total: 616ms	remaining: 36.6ms
    944:	learn: 0.2430354	total: 617ms	remaining: 35.9ms
    945:	learn: 0.2429606	total: 618ms	remaining: 35.3ms
    946:	learn: 0.2427506	total: 618ms	remaining: 34.6ms
    947:	learn: 0.2426217	total: 619ms	remaining: 33.9ms
    948:	learn: 0.2422505	total: 619ms	remaining: 33.3ms
    949:	learn: 0.2419762	total: 620ms	remaining: 32.6ms
    950:	learn: 0.2419058	total: 620ms	remaining: 32ms
    951:	learn: 0.2418125	total: 621ms	remaining: 31.3ms
    952:	learn: 0.2416424	total: 622ms	remaining: 30.7ms
    953:	learn: 0.2414998	total: 622ms	remaining: 30ms
    954:	learn: 0.2411779	total: 623ms	remaining: 29.3ms
    955:	learn: 0.2410044	total: 623ms	remaining: 28.7ms
    956:	learn: 0.2406604	total: 624ms	remaining: 28ms
    957:	learn: 0.2404150	total: 624ms	remaining: 27.4ms
    958:	learn: 0.2402991	total: 625ms	remaining: 26.7ms
    959:	learn: 0.2399422	total: 625ms	remaining: 26.1ms
    960:	learn: 0.2394114	total: 626ms	remaining: 25.4ms
    961:	learn: 0.2391224	total: 627ms	remaining: 24.8ms
    962:	learn: 0.2389314	total: 627ms	remaining: 24.1ms
    963:	learn: 0.2387737	total: 628ms	remaining: 23.5ms
    964:	learn: 0.2385834	total: 629ms	remaining: 22.8ms
    965:	learn: 0.2383537	total: 629ms	remaining: 22.1ms
    966:	learn: 0.2379732	total: 630ms	remaining: 21.5ms
    967:	learn: 0.2377205	total: 630ms	remaining: 20.8ms
    968:	learn: 0.2375250	total: 631ms	remaining: 20.2ms
    969:	learn: 0.2371207	total: 632ms	remaining: 19.5ms
    970:	learn: 0.2368426	total: 632ms	remaining: 18.9ms
    971:	learn: 0.2364895	total: 633ms	remaining: 18.2ms
    972:	learn: 0.2363425	total: 634ms	remaining: 17.6ms
    973:	learn: 0.2362554	total: 634ms	remaining: 16.9ms
    974:	learn: 0.2361535	total: 635ms	remaining: 16.3ms
    975:	learn: 0.2359489	total: 635ms	remaining: 15.6ms
    976:	learn: 0.2358246	total: 636ms	remaining: 15ms
    977:	learn: 0.2356008	total: 637ms	remaining: 14.3ms
    978:	learn: 0.2354048	total: 638ms	remaining: 13.7ms
    979:	learn: 0.2353048	total: 638ms	remaining: 13ms
    980:	learn: 0.2350507	total: 639ms	remaining: 12.4ms
    981:	learn: 0.2349248	total: 639ms	remaining: 11.7ms
    982:	learn: 0.2348873	total: 640ms	remaining: 11.1ms
    983:	learn: 0.2346727	total: 641ms	remaining: 10.4ms
    984:	learn: 0.2343994	total: 641ms	remaining: 9.77ms
    985:	learn: 0.2341855	total: 642ms	remaining: 9.11ms
    986:	learn: 0.2340145	total: 642ms	remaining: 8.46ms
    987:	learn: 0.2339573	total: 643ms	remaining: 7.81ms
    988:	learn: 0.2339129	total: 643ms	remaining: 7.16ms
    989:	learn: 0.2337388	total: 644ms	remaining: 6.51ms
    990:	learn: 0.2335637	total: 645ms	remaining: 5.85ms
    991:	learn: 0.2333938	total: 645ms	remaining: 5.2ms
    992:	learn: 0.2333117	total: 646ms	remaining: 4.55ms
    993:	learn: 0.2331752	total: 646ms	remaining: 3.9ms
    994:	learn: 0.2329763	total: 647ms	remaining: 3.25ms
    995:	learn: 0.2327660	total: 647ms	remaining: 2.6ms
    996:	learn: 0.2326018	total: 648ms	remaining: 1.95ms
    997:	learn: 0.2324260	total: 649ms	remaining: 1.3ms
    998:	learn: 0.2323572	total: 649ms	remaining: 649us
    999:	learn: 0.2320288	total: 650ms	remaining: 0us
    <catboost.core.CatBoostClassifier object at 0x1734f43a0> : 0.6837606837606838
    RandomForestClassifier() : 0.6410256410256411
    ExtraTreesClassifier() : 0.7094017094017094
    GradientBoostingClassifier() : 0.6153846153846154



```python

```
