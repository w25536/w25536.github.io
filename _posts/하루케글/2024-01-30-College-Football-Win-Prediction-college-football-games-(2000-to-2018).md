---
title: "College Football Win Prediction college football games (2000 to 2018)"
date: 2024-01-30
last_modified_at: 2024-01-30
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "College Football Win Prediction college football games (2000 to 2018) 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "jeffgallini/college-football-attendance-2000-to-2018"
)

print("Path to dataset files:", path)
```

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.4)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/jeffgallini/college-football-attendance-2000-to-2018/versions/1



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
```


```python
import os

df = pd.read_csv(os.path.join(path, "CFBeattendance.csv"), encoding="latin-1")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6672 entries, 0 to 6671
    Data columns (total 25 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Date              6672 non-null   object 
     1   Team              6672 non-null   object 
     2   Time              6672 non-null   object 
     3   Opponent          6672 non-null   object 
     4   Rank              6672 non-null   object 
     5   Site              6672 non-null   object 
     6   TV                6672 non-null   object 
     7   Result            6672 non-null   object 
     8   Attendance        6672 non-null   int64  
     9   Current Wins      6672 non-null   int64  
     10  Current Losses    6672 non-null   int64  
     11  Stadium Capacity  6672 non-null   int64  
     12  Fill Rate         6672 non-null   float64
     13  New Coach         6672 non-null   bool   
     14  Tailgating        6672 non-null   bool   
     15  PRCP              6672 non-null   float64
     16  SNOW              6672 non-null   float64
     17  SNWD              6672 non-null   float64
     18  TMAX              6672 non-null   int64  
     19  TMIN              6672 non-null   int64  
     20  Opponent_Rank     6672 non-null   object 
     21  Conference        6672 non-null   object 
     22  Year              6672 non-null   int64  
     23  Month             6672 non-null   int64  
     24  Day               6672 non-null   int64  
    dtypes: bool(2), float64(4), int64(9), object(10)
    memory usage: 1.2+ MB



```python
col_to_drop = ["Date", "Site", "Team", "Opponent"]
```


```python
df = df.drop(col_to_drop, axis=1)
```


```python
df.isna().sum()  # no missing value
```




    Time                0
    Rank                0
    TV                  0
    Result              0
    Attendance          0
    Current Wins        0
    Current Losses      0
    Stadium Capacity    0
    Fill Rate           0
    New Coach           0
    Tailgating          0
    PRCP                0
    SNOW                0
    SNWD                0
    TMAX                0
    TMIN                0
    Opponent_Rank       0
    Conference          0
    Year                0
    Month               0
    Day                 0
    dtype: int64




```python
(df.dtypes)
```




    Time                 object
    Rank                 object
    TV                   object
    Result               object
    Attendance            int64
    Current Wins          int64
    Current Losses        int64
    Stadium Capacity      int64
    Fill Rate           float64
    New Coach              bool
    Tailgating             bool
    PRCP                float64
    SNOW                float64
    SNWD                float64
    TMAX                  int64
    TMIN                  int64
    Opponent_Rank        object
    Conference           object
    Year                  int64
    Month                 int64
    Day                   int64
    dtype: object




```python
categorical_features = df.dtypes[(df.dtypes == "object")]

"""
1. df.dtypes: DataFrame의 각 열(column)의 데이터 타입을 보여줍니다.
2. df.dtypes == 'object': 각 열의 데이터 타입이 'object'인지 확인하여 True/False 값을 반환합니다.
3. df.dtypes[...]: 대괄호 안의 조건이 True인 열들만 선택합니다.
"""
```




    "\n1. df.dtypes: DataFrame의 각 열(column)의 데이터 타입을 보여줍니다.\n2. df.dtypes == 'object': 각 열의 데이터 타입이 'object'인지 확인하여 True/False 값을 반환합니다.\n3. df.dtypes[...]: 대괄호 안의 조건이 True인 열들만 선택합니다.\n"




```python
categorical_features = list(categorical_features.index)

categorical_features
```




    ['Time', 'Rank', 'TV', 'Result', 'Opponent_Rank', 'Conference']




```python
categorical_features = [
    feature for feature in categorical_features if feature != "Result"
]

categorical_features
```




    ['Time', 'Rank', 'TV', 'Opponent_Rank', 'Conference']




```python
def get_uniques(df, columns):
    return {column: list(df[column].unique()) for column in columns}
```


```python
get_uniques(df, categorical_features)
```




    {'Time': ['8:00 PM',
      '6:00 PM',
      '11:30 AM',
      '2:00 PM',
      '1:30 PM',
      '6:30 PM',
      '2:30 PM',
      '1:00 PM',
      '6:45 PM',
      '7:45 PM',
      '5:00 PM',
      '7:00 PM',
      '11:00 AM',
      '11:21 AM',
      '6:15 PM',
      '3:00 PM',
      '12:00 PM',
      '9:00 PM',
      '4:30 PM',
      '7:30 PM',
      '3:05 PM',
      '4:00 PM',
      '5:30 PM',
      '6:05 PM',
      '1:05 PM',
      '8:15 PM',
      '12:30 PM',
      '8:30 PM',
      '1:45 PM',
      '3:45 PM',
      '7:15 PM',
      '10:00 AM',
      '3:30 PM',
      '5:45 PM',
      '12:20 PM',
      '11:10 AM',
      '7:20 PM',
      '1:10 PM',
      '6:10 PM',
      '11:40 AM',
      '2:35 PM',
      '2:45 PM',
      '10:30 AM',
      '3:40 PM',
      '12:10 PM',
      '12:05 PM',
      '7:05 PM',
      '4:45 PM',
      '4:05 PM',
      '8:05 PM',
      '2:05 PM',
      '6:35 PM',
      '6:50 PM',
      '3:15 PM',
      '12:15 PM',
      '1:15 PM',
      '5:10 PM',
      '4:20 PM',
      '8:04 PM',
      '11:05 AM',
      '5:05 PM',
      '4:15 PM',
      '5:15 PM',
      '7:35 PM',
      '3:36 PM',
      '7:06 PM',
      '12:35 PM',
      '12:45 PM',
      '2:15 PM'],
     'Rank': ['NR',
      '14',
      '7',
      '11',
      '17',
      '15',
      '13',
      '5',
      '20',
      '10',
      '21',
      '19',
      '12',
      '8',
      '6',
      '18',
      '24',
      '16',
      '22',
      '25',
      '23',
      '9',
      '4',
      '2',
      '3',
      '1'],
     'TV': ['Not on TV',
      'ESPN2',
      'JPS',
      'CBS',
      'ESPN',
      'LFS',
      'ESPNU',
      'Raycom',
      'PPV',
      'SECN',
      'CSS',
      'SECRN',
      'ARSN PPV',
      'SECTV',
      'FSN',
      'Versus',
      'FCS',
      'ABC/ESPN3',
      'ABC',
      'FCS Central',
      'FX',
      'FS1',
      'FOX',
      'ABC/ESPN2',
      'FS2',
      'KBCI-TV',
      'SPW',
      'ESPN+',
      'KTVB',
      'ESPNU/ESPN 3D',
      'CBSSN',
      'mtn',
      'NBCSN',
      'ESPN3',
      'KSL',
      'SWP',
      'CSTV',
      'Versus/CSTV/mtn',
      'CBS CS',
      'mtn/CBS CS',
      'BYUtv',
      'BYUtv/ESPN3',
      'Raycom/LFS',
      'ESPNGP',
      'ESPN360',
      'ABC/ESPN',
      'ACCN',
      'RSN',
      'ACCRN',
      'ACCRSN',
      'TBS',
      'BV',
      'P12N',
      'SUN',
      'SUN PPV',
      'ESPN/ESPN 3D',
      'ESPN3 (PPV)',
      'ESPNews',
      'ACCN Extra',
      'HTS',
      'ACC RSN',
      'ACCN+',
      'ESPNC',
      'Mediacom',
      'BTN',
      'MC22',
      'CYtv',
      'Cyclones.tv',
      'K-StateHD.TV',
      'WSAZ',
      'iTV',
      'WOWK',
      'MASN',
      'CSS/CSNH',
      'TW TX',
      'ASN',
      'beIN',
      'STADIUM',
      'CUSA.TV/WCHS/WVAH',
      'ACCS',
      'HDNet',
      'FSN PPV',
      'WAC.tv',
      'Cox/ESPN3',
      'WSN',
      'WSN/ALT',
      'MWN',
      'MWN/Oceanic PPV',
      'MWN/Twitter',
      'ATTSNRM',
      'CSNC',
      'CSNC/ESPN3',
      'NBC',
      'CSS PPV',
      'FSOK PPV',
      'MSG',
      'SNY',
      'Big East Network',
      'AAN',
      'CSN',
      'TWC Texas',
      'Time Warner Sports',
      'TWCSN',
      'TWCS/SNY',
      'TWCS NY/ESPN3',
      'FSSW\x96PPV',
      'FSSW PPV',
      'BCN',
      'ESPN Classic',
      'ESPN3/WTOL',
      'BCSN',
      'WTOL/ESPN3',
      'ESPN+/WTOL',
      'SportSouth',
      'SBCN',
      'SBN',
      'Troy/IMGSN',
      'FSNW2',
      'FSNPT',
      'FSNW',
      'FSPT',
      'FSW',
      'FSNNW',
      'RTNW',
      'Sun Belt Network',
      'CUSA.tv',
      'FloSports',
      'MSN',
      'RTPT',
      'AT&TSN Pitt',
      'CTSN PPV',
      'PPV/ESPN3',
      'KWBA',
      'FSNAZ',
      'KGUN',
      'FSAZ',
      'CST/CSS',
      'KATV',
      'Comcast',
      'WNDY',
      'Big East Network/ESPN3',
      'NESN',
      'Fox Sports Net',
      'ESPN app',
      'Empire',
      'YES',
      'ESPN Plus',
      'Time Warner Cable SportsNet',
      'Time Warner Cable SportsNet, Fox Sports Ohio',
      'STO',
      'TWCS',
      'TWCS/ESPN3',
      'ASN/ESPN3',
      'BCSN/ESPN3',
      'FSNBA',
      'KRON',
      'CSNCA',
      'WITN',
      'CUSA.TV',
      'KFVE',
      'Oceanic PPV, KFVE (delayed)',
      'ESPN2 Oceanic PPV, KFVE (Simulcast)',
      'KJZZ, Oceanic PPV, KFVE (delayed)',
      'Oceanic PPV',
      'ESPN3, WAC Sports Network',
      'Oceanic PPV/ESPN3',
      'WSN/ALT2',
      'Oceanic PPV/ROOT',
      'Oceanic PPV/ROOT/TWCSN',
      'Oceanic PPV/Campus Insiders',
      'SPEC PPV',
      'SPEC HI',
      'SFS',
      'Fox College Sports',
      'Fox Sports Network',
      'Local Channels',
      '6Sports',
      'Jayhawk TV',
      'Jayhawk SN',
      'JTV',
      'CSS/ESPN+',
      'Ragin Cajuns Network/ESPN3',
      'RCN/ESPN3',
      'CST',
      'KFRE-TV',
      'ESPN3, ALT, CST',
      'ESPN3, ALT2, CST, MASN',
      'ESPN3, WSN',
      'ESPN+/CST/ALT 2',
      'CI',
      'WPTY',
      'WLMT',
      'WMC',
      'ONN',
      'ONN/ESPN+',
      'FSN Ohio',
      'ONN/STO',
      'SportsTime Ohio',
      'ESPN+/ESPN3',
      'BCSN2',
      'FOX 19',
      'ESPN+/CSS',
      'SBN/CSS',
      'WUXP',
      'GBR',
      'CSS/FSMW',
      'SEC Alt.',
      'KKWB',
      'Mtn',
      'PVN',
      'Comcast/TWCSN',
      'KASY',
      'RTRM',
      'RTSW/RTRM',
      'RTRM/RTSW',
      'GTN',
      'FSN-OH',
      'FCS Pacific',
      'AZTV',
      '4SD',
      'VS.',
      'Mtn.',
      'KUSI',
      'TWCSN/KTVD',
      'RTRM/FSSD',
      'WJTC/ESPN3',
      'CSS/WJTC/ESPN3',
      'Sun Belt Network/CSS',
      'ion',
      'BHSN',
      'A10 TV',
      'CN8',
      'CSTV, CN8',
      'MetroTV New York',
      'A10TV',
      'CSN NE, CSN MA',
      'CSN NE',
      'CSNNE',
      'WSHM',
      'CSN NE/Comcast Network',
      'ELVN<U+2606>',
      'ELVN',
      'FSN West',
      'TWC',
      'TWCEP'],
     'Opponent_Rank': ['NR',
      '25',
      '24',
      '9',
      '17',
      '18',
      '10',
      '14',
      '21',
      '6',
      '13',
      '23',
      '20',
      '1',
      '15',
      '2',
      '8',
      '7',
      '12',
      '19',
      '4',
      '5',
      '16',
      '22',
      '3',
      '11'],
     'Conference': ['SEC',
      'Big-12',
      'WAC',
      'MWC',
      'Independent',
      'ACC',
      'Pac-12',
      'Big-10',
      'Mid-American',
      'CUSA',
      'AAC',
      'Big East',
      'Sun Belt',
      'FCS']}




```python
nominal_features = ["Conference"]

ordinal_features = ["Time", "Rank", "Opponent_Rank"]

binary_features = ["TV", "New Coach", "Tailgating"]
```


```python
## Binary Encode
# df['TV'] = df['TV'].apply(lambda x: x if x == 'Not on TV' else 'On TV')

df["TV"] = df["TV"].apply(lambda x: 0 if x == "Not on TV" else 1)
```


```python
df["TV"].unique()
```




    array([0, 1])




```python
df["New Coach"] = df["New Coach"].astype(int)
df["Tailgating"] = df["Tailgating"].astype(int)
```


```python
df["Rank"] = df["Rank"].apply(lambda x: 26 if x == "NR" else int(x))
df["Opponent_Rank"] = df["Opponent_Rank"].apply(lambda x: 26 if x == "NR" else int(x))

print(df["Rank"].unique())
print(df["Opponent_Rank"].unique())
```

    [26 14  7 11 17 15 13  5 20 10 21 19 12  8  6 18 24 16 22 25 23  9  4  2
      3  1]
    [26 25 24  9 17 18 10 14 21  6 13 23 20  1 15  2  8  7 12 19  4  5 16 22
      3 11]



```python
df["Time"].unique()

time_ordering = sorted(df["Time"].unique())

df["Time"] = df["Time"].apply(lambda x: time_ordering.index(x))
```


```python
df["Time"]
```




    0       63
    1       47
    2       63
    3        6
    4       47
            ..
    6667    45
    6668    45
    6669    45
    6670    16
    6671    16
    Name: Time, Length: 6672, dtype: int64




```python
# Nominal encode
df["Conference"].unique()
```




    array(['SEC', 'Big-12', 'WAC', 'MWC', 'Independent', 'ACC', 'Pac-12',
           'Big-10', 'Mid-American', 'CUSA', 'AAC', 'Big East', 'Sun Belt',
           'FCS'], dtype=object)




```python
def onehot_encode(df, column):
    dummies = pd.get_dummies(df[column], dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([column], axis=1)
    return df
```


```python
df = onehot_encode(df, "Conference")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6672 entries, 0 to 6671
    Data columns (total 34 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Time              6672 non-null   int64  
     1   Rank              6672 non-null   int64  
     2   TV                6672 non-null   int64  
     3   Result            6672 non-null   object 
     4   Attendance        6672 non-null   int64  
     5   Current Wins      6672 non-null   int64  
     6   Current Losses    6672 non-null   int64  
     7   Stadium Capacity  6672 non-null   int64  
     8   Fill Rate         6672 non-null   float64
     9   New Coach         6672 non-null   int64  
     10  Tailgating        6672 non-null   int64  
     11  PRCP              6672 non-null   float64
     12  SNOW              6672 non-null   float64
     13  SNWD              6672 non-null   float64
     14  TMAX              6672 non-null   int64  
     15  TMIN              6672 non-null   int64  
     16  Opponent_Rank     6672 non-null   int64  
     17  Year              6672 non-null   int64  
     18  Month             6672 non-null   int64  
     19  Day               6672 non-null   int64  
     20  AAC               6672 non-null   int64  
     21  ACC               6672 non-null   int64  
     22  Big East          6672 non-null   int64  
     23  Big-10            6672 non-null   int64  
     24  Big-12            6672 non-null   int64  
     25  CUSA              6672 non-null   int64  
     26  FCS               6672 non-null   int64  
     27  Independent       6672 non-null   int64  
     28  MWC               6672 non-null   int64  
     29  Mid-American      6672 non-null   int64  
     30  Pac-12            6672 non-null   int64  
     31  SEC               6672 non-null   int64  
     32  Sun Belt          6672 non-null   int64  
     33  WAC               6672 non-null   int64  
    dtypes: float64(4), int64(29), object(1)
    memory usage: 1.7+ MB



```python
df = df.drop([4355, 5442, 5449, 5456], axis=0)
```


```python
y = df["Result"]
X = df.drop(["Result"], axis=1)
```


```python
y
```




    0        W 380
    1       W 3831
    2       W 2821
    3        L 738
    4        W 526
             ...   
    6667    L 2027
    6668    L 2427
    6669     L 019
    6670    L 3248
    6671     L 739
    Name: Result, Length: 6668, dtype: object




```python
import re

y = y.apply(lambda x: re.search(r"^[^\s]*", x).group(0))


y.unique()
```




    array(['W', 'L'], dtype=object)




```python
# index_to_delete = y[(y == 'NC') | (y == 'White') | (y == 'Blue')].index
```


```python
# index_to_delete
```


```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_mappings = {index: value for index, value in enumerate(label_encoder.classes_)}
```


```python
y_mappings
```




    {0: 'L', 1: 'W'}




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
      <th>Time</th>
      <th>Rank</th>
      <th>TV</th>
      <th>Attendance</th>
      <th>Current Wins</th>
      <th>Current Losses</th>
      <th>Stadium Capacity</th>
      <th>Fill Rate</th>
      <th>New Coach</th>
      <th>Tailgating</th>
      <th>...</th>
      <th>Big-12</th>
      <th>CUSA</th>
      <th>FCS</th>
      <th>Independent</th>
      <th>MWC</th>
      <th>Mid-American</th>
      <th>Pac-12</th>
      <th>SEC</th>
      <th>Sun Belt</th>
      <th>WAC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>26</td>
      <td>0</td>
      <td>53946</td>
      <td>0</td>
      <td>0</td>
      <td>53727</td>
      <td>1.004076</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>26</td>
      <td>0</td>
      <td>54286</td>
      <td>1</td>
      <td>0</td>
      <td>53727</td>
      <td>1.010404</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>63</td>
      <td>26</td>
      <td>1</td>
      <td>51482</td>
      <td>2</td>
      <td>0</td>
      <td>50019</td>
      <td>1.029249</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>26</td>
      <td>1</td>
      <td>51162</td>
      <td>3</td>
      <td>0</td>
      <td>50019</td>
      <td>1.022851</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>26</td>
      <td>0</td>
      <td>50947</td>
      <td>3</td>
      <td>1</td>
      <td>50019</td>
      <td>1.018553</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6667</th>
      <td>45</td>
      <td>26</td>
      <td>1</td>
      <td>19412</td>
      <td>0</td>
      <td>3</td>
      <td>51500</td>
      <td>0.376932</td>
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
      <th>6668</th>
      <td>45</td>
      <td>26</td>
      <td>1</td>
      <td>12809</td>
      <td>0</td>
      <td>5</td>
      <td>51500</td>
      <td>0.248718</td>
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
      <th>6669</th>
      <td>45</td>
      <td>26</td>
      <td>1</td>
      <td>10787</td>
      <td>0</td>
      <td>7</td>
      <td>51500</td>
      <td>0.209456</td>
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
      <th>6670</th>
      <td>16</td>
      <td>26</td>
      <td>1</td>
      <td>9690</td>
      <td>1</td>
      <td>8</td>
      <td>51500</td>
      <td>0.188155</td>
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
      <th>6671</th>
      <td>16</td>
      <td>26</td>
      <td>1</td>
      <td>14962</td>
      <td>1</td>
      <td>10</td>
      <td>51500</td>
      <td>0.290524</td>
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
  </tbody>
</table>
<p>6668 rows × 33 columns</p>
</div>




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
      <th>Time</th>
      <th>Rank</th>
      <th>TV</th>
      <th>Attendance</th>
      <th>Current Wins</th>
      <th>Current Losses</th>
      <th>Stadium Capacity</th>
      <th>Fill Rate</th>
      <th>New Coach</th>
      <th>Tailgating</th>
      <th>...</th>
      <th>Big-12</th>
      <th>CUSA</th>
      <th>FCS</th>
      <th>Independent</th>
      <th>MWC</th>
      <th>Mid-American</th>
      <th>Pac-12</th>
      <th>SEC</th>
      <th>Sun Belt</th>
      <th>WAC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.926471</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.475769</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.406803</td>
      <td>0.700775</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>0.691176</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.478899</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.406803</td>
      <td>0.705510</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>0.926471</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.453085</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.365732</td>
      <td>0.719610</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>0.088235</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.450139</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.365732</td>
      <td>0.714823</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>0.691176</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.448160</td>
      <td>0.250000</td>
      <td>0.090909</td>
      <td>0.365732</td>
      <td>0.711607</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
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
      <th>6663</th>
      <td>0.661765</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.157841</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.382136</td>
      <td>0.231535</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>6664</th>
      <td>0.661765</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.097052</td>
      <td>0.000000</td>
      <td>0.454545</td>
      <td>0.382136</td>
      <td>0.135604</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>6665</th>
      <td>0.661765</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.078437</td>
      <td>0.000000</td>
      <td>0.636364</td>
      <td>0.382136</td>
      <td>0.106227</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>6666</th>
      <td>0.235294</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.068338</td>
      <td>0.083333</td>
      <td>0.727273</td>
      <td>0.382136</td>
      <td>0.090289</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>6667</th>
      <td>0.235294</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.116873</td>
      <td>0.083333</td>
      <td>0.909091</td>
      <td>0.382136</td>
      <td>0.166883</td>
      <td>0.0</td>
      <td>0.0</td>
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
  </tbody>
</table>
<p>6668 rows × 33 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
```


```python
inputs = tf.keras.Input(shape=(33,))
x = tf.keras.layers.Dense(16, activation="relu")(inputs)
x = tf.keras.layers.Dense(16, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)


# inputs = tf.keras.Input(shape=(33, ))
# x = tf.keras.layers.Dense(64, activation='relu')(inputs)
# x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout for regularization
# x = tf.keras.layers.Dense(32, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout for regularization
# outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

metrics = [
    tf.keras.metrics.BinaryAccuracy(name="acc"),
    tf.keras.metrics.AUC(name="auc"),
]

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

batch_size = 32
epochs = 7

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
)
```

    Epoch 1/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - acc: 0.5323 - auc: 0.4668 - loss: 0.6869 - val_acc: 0.6156 - val_auc: 0.5926 - val_loss: 0.6604
    Epoch 2/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6442 - auc: 0.6132 - loss: 0.6393 - val_acc: 0.6156 - val_auc: 0.6636 - val_loss: 0.6456
    Epoch 3/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6431 - auc: 0.6752 - loss: 0.6222 - val_acc: 0.6349 - val_auc: 0.6870 - val_loss: 0.6290
    Epoch 4/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6512 - auc: 0.6817 - loss: 0.6156 - val_acc: 0.6638 - val_auc: 0.7022 - val_loss: 0.6147
    Epoch 5/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6792 - auc: 0.6865 - loss: 0.6028 - val_acc: 0.6670 - val_auc: 0.7167 - val_loss: 0.6067
    Epoch 6/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6928 - auc: 0.7016 - loss: 0.5901 - val_acc: 0.6809 - val_auc: 0.7281 - val_loss: 0.5939
    Epoch 7/7
    [1m117/117[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.7034 - auc: 0.7243 - loss: 0.5748 - val_acc: 0.6906 - val_auc: 0.7354 - val_loss: 0.5852



```python
plt.figure(figsize=(14, 10))

epochs_range = range(1, epochs + 1)
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


    
![png](030_College_Football_Win_Prediction_college_football_games_%282000_to_2018%29_files/030_College_Football_Win_Prediction_college_football_games_%282000_to_2018%29_37_0.png)
    



```python
np.argmin(val_loss)
```




    6




```python
model.evaluate(X_test, y_test)
```

    [1m63/63[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - acc: 0.6685 - auc: 0.6721 - loss: 0.6038





    [0.592997133731842, 0.6836581826210022, 0.6930145025253296]






```python
y.sum() / len(y)
```




    0.6415716856628674




```python

```
