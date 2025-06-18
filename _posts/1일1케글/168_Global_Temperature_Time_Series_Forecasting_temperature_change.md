---
title: "168_Global_Temperature_Time_Series_Forecasting_temperature_change"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "168_Global_Temperature_Time_Series_Forecasting_temperature_change"
use_math: true
classes: wide
---

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sevgisarac/temperature-change")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/sevgisarac/temperature-change/versions/3



```python
import numpy as np
import pandas as pd

import plotly.express as px
```


```python
from prophet import Prophet
```


```python
import os

df = pd.read_csv(
    os.path.join(path, "Environment_Temperature_change_E_All_Data_NOFLAG.csv"),
    encoding="latin-1",
)
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
      <th>Area Code</th>
      <th>Area</th>
      <th>Months Code</th>
      <th>Months</th>
      <th>Element Code</th>
      <th>Element</th>
      <th>Unit</th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>...</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
      <th>Y2014</th>
      <th>Y2015</th>
      <th>Y2016</th>
      <th>Y2017</th>
      <th>Y2018</th>
      <th>Y2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Afghanistan</td>
      <td>7001</td>
      <td>January</td>
      <td>7271</td>
      <td>Temperature change</td>
      <td>°C</td>
      <td>0.777</td>
      <td>0.062</td>
      <td>2.744</td>
      <td>...</td>
      <td>3.601</td>
      <td>1.179</td>
      <td>-0.583</td>
      <td>1.233</td>
      <td>1.755</td>
      <td>1.943</td>
      <td>3.416</td>
      <td>1.201</td>
      <td>1.996</td>
      <td>2.951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Afghanistan</td>
      <td>7001</td>
      <td>January</td>
      <td>6078</td>
      <td>Standard Deviation</td>
      <td>°C</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>...</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
      <td>1.950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Afghanistan</td>
      <td>7002</td>
      <td>February</td>
      <td>7271</td>
      <td>Temperature change</td>
      <td>°C</td>
      <td>-1.743</td>
      <td>2.465</td>
      <td>3.919</td>
      <td>...</td>
      <td>1.212</td>
      <td>0.321</td>
      <td>-3.201</td>
      <td>1.494</td>
      <td>-3.187</td>
      <td>2.699</td>
      <td>2.251</td>
      <td>-0.323</td>
      <td>2.705</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Afghanistan</td>
      <td>7002</td>
      <td>February</td>
      <td>6078</td>
      <td>Standard Deviation</td>
      <td>°C</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>...</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
      <td>2.597</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Afghanistan</td>
      <td>7003</td>
      <td>March</td>
      <td>7271</td>
      <td>Temperature change</td>
      <td>°C</td>
      <td>0.516</td>
      <td>1.336</td>
      <td>0.403</td>
      <td>...</td>
      <td>3.390</td>
      <td>0.748</td>
      <td>-0.527</td>
      <td>2.246</td>
      <td>-0.076</td>
      <td>-0.497</td>
      <td>2.296</td>
      <td>0.834</td>
      <td>4.418</td>
      <td>0.234</td>
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
      <th>9651</th>
      <td>5873</td>
      <td>OECD</td>
      <td>7018</td>
      <td>JunJulAug</td>
      <td>6078</td>
      <td>Standard Deviation</td>
      <td>°C</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>...</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.247</td>
    </tr>
    <tr>
      <th>9652</th>
      <td>5873</td>
      <td>OECD</td>
      <td>7019</td>
      <td>SepOctNov</td>
      <td>7271</td>
      <td>Temperature change</td>
      <td>°C</td>
      <td>0.036</td>
      <td>0.461</td>
      <td>0.665</td>
      <td>...</td>
      <td>0.958</td>
      <td>1.106</td>
      <td>0.885</td>
      <td>1.041</td>
      <td>0.999</td>
      <td>1.670</td>
      <td>1.535</td>
      <td>1.194</td>
      <td>0.581</td>
      <td>1.233</td>
    </tr>
    <tr>
      <th>9653</th>
      <td>5873</td>
      <td>OECD</td>
      <td>7019</td>
      <td>SepOctNov</td>
      <td>6078</td>
      <td>Standard Deviation</td>
      <td>°C</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>...</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>9654</th>
      <td>5873</td>
      <td>OECD</td>
      <td>7020</td>
      <td>Meteorological year</td>
      <td>7271</td>
      <td>Temperature change</td>
      <td>°C</td>
      <td>0.165</td>
      <td>-0.009</td>
      <td>0.134</td>
      <td>...</td>
      <td>1.246</td>
      <td>0.805</td>
      <td>1.274</td>
      <td>0.991</td>
      <td>0.811</td>
      <td>1.282</td>
      <td>1.850</td>
      <td>1.349</td>
      <td>1.088</td>
      <td>1.297</td>
    </tr>
    <tr>
      <th>9655</th>
      <td>5873</td>
      <td>OECD</td>
      <td>7020</td>
      <td>Meteorological year</td>
      <td>6078</td>
      <td>Standard Deviation</td>
      <td>°C</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>...</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
      <td>0.260</td>
    </tr>
  </tbody>
</table>
<p>9656 rows × 66 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9656 entries, 0 to 9655
    Data columns (total 66 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Area Code     9656 non-null   int64  
     1   Area          9656 non-null   object 
     2   Months Code   9656 non-null   int64  
     3   Months        9656 non-null   object 
     4   Element Code  9656 non-null   int64  
     5   Element       9656 non-null   object 
     6   Unit          9656 non-null   object 
     7   Y1961         8287 non-null   float64
     8   Y1962         8322 non-null   float64
     9   Y1963         8294 non-null   float64
     10  Y1964         8252 non-null   float64
     11  Y1965         8281 non-null   float64
     12  Y1966         8364 non-null   float64
     13  Y1967         8347 non-null   float64
     14  Y1968         8345 non-null   float64
     15  Y1969         8326 non-null   float64
     16  Y1970         8308 non-null   float64
     17  Y1971         8303 non-null   float64
     18  Y1972         8323 non-null   float64
     19  Y1973         8394 non-null   float64
     20  Y1974         8374 non-null   float64
     21  Y1975         8280 non-null   float64
     22  Y1976         8209 non-null   float64
     23  Y1977         8257 non-null   float64
     24  Y1978         8327 non-null   float64
     25  Y1979         8290 non-null   float64
     26  Y1980         8283 non-null   float64
     27  Y1981         8276 non-null   float64
     28  Y1982         8237 non-null   float64
     29  Y1983         8205 non-null   float64
     30  Y1984         8259 non-null   float64
     31  Y1985         8216 non-null   float64
     32  Y1986         8268 non-null   float64
     33  Y1987         8284 non-null   float64
     34  Y1988         8273 non-null   float64
     35  Y1989         8257 non-null   float64
     36  Y1990         8239 non-null   float64
     37  Y1991         8158 non-null   float64
     38  Y1992         8354 non-null   float64
     39  Y1993         8315 non-null   float64
     40  Y1994         8373 non-null   float64
     41  Y1995         8409 non-null   float64
     42  Y1996         8439 non-null   float64
     43  Y1997         8309 non-null   float64
     44  Y1998         8370 non-null   float64
     45  Y1999         8324 non-null   float64
     46  Y2000         8342 non-null   float64
     47  Y2001         8241 non-null   float64
     48  Y2002         8312 non-null   float64
     49  Y2003         8390 non-null   float64
     50  Y2004         8415 non-null   float64
     51  Y2005         8424 non-null   float64
     52  Y2006         8503 non-null   float64
     53  Y2007         8534 non-null   float64
     54  Y2008         8475 non-null   float64
     55  Y2009         8419 non-null   float64
     56  Y2010         8435 non-null   float64
     57  Y2011         8437 non-null   float64
     58  Y2012         8350 non-null   float64
     59  Y2013         8427 non-null   float64
     60  Y2014         8377 non-null   float64
     61  Y2015         8361 non-null   float64
     62  Y2016         8348 non-null   float64
     63  Y2017         8366 non-null   float64
     64  Y2018         8349 non-null   float64
     65  Y2019         8365 non-null   float64
    dtypes: float64(59), int64(3), object(4)
    memory usage: 4.9+ MB



```python
df = df.query('Element == "Temperature change"')
```


```python
numeric_columns = df.select_dtypes(include=[np.number]).columns

df = df.groupby("Area")[numeric_columns].mean()
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
      <th>Area Code</th>
      <th>Months Code</th>
      <th>Element Code</th>
      <th>Y1961</th>
      <th>Y1962</th>
      <th>Y1963</th>
      <th>Y1964</th>
      <th>Y1965</th>
      <th>Y1966</th>
      <th>Y1967</th>
      <th>...</th>
      <th>Y2010</th>
      <th>Y2011</th>
      <th>Y2012</th>
      <th>Y2013</th>
      <th>Y2014</th>
      <th>Y2015</th>
      <th>Y2016</th>
      <th>Y2017</th>
      <th>Y2018</th>
      <th>Y2019</th>
    </tr>
    <tr>
      <th>Area</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>2.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.027941</td>
      <td>-0.197471</td>
      <td>0.888706</td>
      <td>-0.905647</td>
      <td>-0.051824</td>
      <td>0.222118</td>
      <td>-0.362176</td>
      <td>...</td>
      <td>1.499235</td>
      <td>1.246118</td>
      <td>0.179765</td>
      <td>1.251706</td>
      <td>0.487000</td>
      <td>1.098294</td>
      <td>1.671882</td>
      <td>1.306765</td>
      <td>1.574706</td>
      <td>0.899588</td>
    </tr>
    <tr>
      <th>Africa</th>
      <td>5100.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>-0.089000</td>
      <td>-0.006353</td>
      <td>0.077471</td>
      <td>-0.189471</td>
      <td>-0.195294</td>
      <td>0.146706</td>
      <td>-0.222588</td>
      <td>...</td>
      <td>1.518059</td>
      <td>0.892941</td>
      <td>0.814941</td>
      <td>1.002353</td>
      <td>1.049176</td>
      <td>1.207941</td>
      <td>1.483941</td>
      <td>1.188647</td>
      <td>1.236059</td>
      <td>1.428000</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>3.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.473235</td>
      <td>0.238941</td>
      <td>0.254647</td>
      <td>-0.197118</td>
      <td>-0.361588</td>
      <td>0.460294</td>
      <td>-0.078353</td>
      <td>...</td>
      <td>1.208706</td>
      <td>1.137765</td>
      <td>1.513235</td>
      <td>1.541706</td>
      <td>1.466529</td>
      <td>1.689471</td>
      <td>1.616000</td>
      <td>1.416529</td>
      <td>2.206412</td>
      <td>2.029000</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>4.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.302059</td>
      <td>0.029706</td>
      <td>0.176824</td>
      <td>0.056529</td>
      <td>-0.062353</td>
      <td>0.320412</td>
      <td>-0.072647</td>
      <td>...</td>
      <td>2.334000</td>
      <td>1.354294</td>
      <td>1.258824</td>
      <td>1.224412</td>
      <td>1.772529</td>
      <td>1.285235</td>
      <td>1.910765</td>
      <td>1.518588</td>
      <td>1.404353</td>
      <td>1.289529</td>
    </tr>
    <tr>
      <th>American Samoa</th>
      <td>5.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>-0.028941</td>
      <td>-0.106000</td>
      <td>0.096471</td>
      <td>-0.275529</td>
      <td>-0.413412</td>
      <td>0.133294</td>
      <td>-0.372353</td>
      <td>...</td>
      <td>1.089353</td>
      <td>0.854600</td>
      <td>0.938667</td>
      <td>1.166867</td>
      <td>1.084923</td>
      <td>0.841538</td>
      <td>1.564846</td>
      <td>1.271000</td>
      <td>1.119615</td>
      <td>1.465923</td>
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
      <th>World</th>
      <td>5000.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.155941</td>
      <td>0.053235</td>
      <td>0.139000</td>
      <td>-0.277353</td>
      <td>-0.170941</td>
      <td>0.171412</td>
      <td>-0.113706</td>
      <td>...</td>
      <td>1.229824</td>
      <td>0.948471</td>
      <td>1.036118</td>
      <td>1.065647</td>
      <td>1.068706</td>
      <td>1.460647</td>
      <td>1.626059</td>
      <td>1.462353</td>
      <td>1.278294</td>
      <td>1.510882</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>249.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.024647</td>
      <td>-0.005412</td>
      <td>0.168176</td>
      <td>-0.319882</td>
      <td>-0.574882</td>
      <td>0.069412</td>
      <td>-0.242000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Yugoslav SFR</th>
      <td>248.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.691059</td>
      <td>-0.244529</td>
      <td>-0.181353</td>
      <td>-0.352882</td>
      <td>-0.470353</td>
      <td>0.594765</td>
      <td>0.201353</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>251.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.149353</td>
      <td>-0.164647</td>
      <td>-0.376294</td>
      <td>-0.283118</td>
      <td>-0.364706</td>
      <td>0.292471</td>
      <td>-0.061235</td>
      <td>...</td>
      <td>1.464118</td>
      <td>1.020941</td>
      <td>1.077059</td>
      <td>1.128294</td>
      <td>1.037941</td>
      <td>1.779529</td>
      <td>1.298118</td>
      <td>0.660471</td>
      <td>1.181588</td>
      <td>1.426294</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>181.0</td>
      <td>7009.882353</td>
      <td>7271.0</td>
      <td>0.234412</td>
      <td>0.206000</td>
      <td>-0.420235</td>
      <td>-0.134941</td>
      <td>-0.305118</td>
      <td>0.182353</td>
      <td>-0.099941</td>
      <td>...</td>
      <td>1.025353</td>
      <td>0.515824</td>
      <td>0.613588</td>
      <td>0.401412</td>
      <td>0.344353</td>
      <td>1.209059</td>
      <td>1.205353</td>
      <td>0.237824</td>
      <td>0.532059</td>
      <td>1.208647</td>
    </tr>
  </tbody>
</table>
<p>284 rows × 62 columns</p>
</div>




```python
df = df.loc[:, "Y1961":]
```


```python
df = pd.DataFrame(df.mean()).reset_index(drop=False)
```


```python
df = df.rename(columns={"index": "ds", 0: "y"})
```


```python
df["ds"] = df["ds"].apply(lambda x: x[1:]).astype(int)
```


```python
time_series = df.copy()
```


```python
time_series
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1961</td>
      <td>0.143032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1962</td>
      <td>-0.028398</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1963</td>
      <td>-0.026297</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1964</td>
      <td>-0.122865</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1965</td>
      <td>-0.224154</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1966</td>
      <td>0.095070</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1967</td>
      <td>-0.131975</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1968</td>
      <td>-0.167841</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1969</td>
      <td>0.105694</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1970</td>
      <td>0.072189</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1971</td>
      <td>-0.177649</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1972</td>
      <td>-0.049936</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1973</td>
      <td>0.199149</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1974</td>
      <td>-0.128841</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1975</td>
      <td>-0.030398</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1976</td>
      <td>-0.210907</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977</td>
      <td>0.185724</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1978</td>
      <td>0.053986</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1979</td>
      <td>0.230299</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1980</td>
      <td>0.224411</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1981</td>
      <td>0.222159</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1982</td>
      <td>0.160740</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1983</td>
      <td>0.348746</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1984</td>
      <td>0.076055</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1985</td>
      <td>0.069280</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1986</td>
      <td>0.139045</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1987</td>
      <td>0.415012</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1988</td>
      <td>0.435257</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>0.283534</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1990</td>
      <td>0.579354</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1991</td>
      <td>0.335127</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1992</td>
      <td>0.254460</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1993</td>
      <td>0.243441</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1994</td>
      <td>0.559852</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1995</td>
      <td>0.603439</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1996</td>
      <td>0.317843</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1997</td>
      <td>0.578825</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1998</td>
      <td>0.951884</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1999</td>
      <td>0.732435</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2000</td>
      <td>0.689658</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2001</td>
      <td>0.806679</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2002</td>
      <td>0.917838</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2003</td>
      <td>0.862185</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2004</td>
      <td>0.787869</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2005</td>
      <td>0.886901</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2006</td>
      <td>0.910877</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2007</td>
      <td>1.004826</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2008</td>
      <td>0.813313</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2009</td>
      <td>0.943937</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2010</td>
      <td>1.080097</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2011</td>
      <td>0.863045</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2012</td>
      <td>0.901637</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2013</td>
      <td>0.977131</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2014</td>
      <td>1.131417</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2015</td>
      <td>1.326462</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2016</td>
      <td>1.440185</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2017</td>
      <td>1.299112</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2018</td>
      <td>1.310459</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2019</td>
      <td>1.464899</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.line(
    time_series,
    x="ds",
    y="y",
    labels={"ds : Year, y: Change in Temperature"},
    title="Average Global Temperature Change Over Time",
)
```


```python
fig.show()
```




```python
time_train = time_series.iloc[:44, :].copy()
time_test = time_series.iloc[44:, :].copy()
```


```python
in_model = Prophet()
in_model.fit(time_train)
```

    03:42:30 - cmdstanpy - INFO - Chain [1] start processing
    03:42:30 - cmdstanpy - INFO - Chain [1] done processing





    <prophet.forecaster.Prophet at 0x3073b1d90>




```python
forecast = in_model.predict(time_test)
forecast_result = forecast.loc[:, ["ds", "yhat"]]

forecast_result["ds"] = forecast_result["ds"].apply(lambda x: x.year)
```


```python
forecast_result
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
      <th>ds</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>0.811440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>0.809407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007</td>
      <td>0.796937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>0.774072</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>0.901347</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2010</td>
      <td>0.899313</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011</td>
      <td>0.886843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012</td>
      <td>0.863978</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>0.991253</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>0.989220</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015</td>
      <td>0.976750</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016</td>
      <td>0.953885</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017</td>
      <td>1.081160</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018</td>
      <td>1.079126</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>1.066656</td>
    </tr>
  </tbody>
</table>
</div>




```python
time_series.merge(forecast_result, on="ds")
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
      <th>ds</th>
      <th>y</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>0.886901</td>
      <td>0.811440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>0.910877</td>
      <td>0.809407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007</td>
      <td>1.004826</td>
      <td>0.796937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>0.813313</td>
      <td>0.774072</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>0.943937</td>
      <td>0.901347</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2010</td>
      <td>1.080097</td>
      <td>0.899313</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011</td>
      <td>0.863045</td>
      <td>0.886843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012</td>
      <td>0.901637</td>
      <td>0.863978</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>0.977131</td>
      <td>0.991253</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>1.131417</td>
      <td>0.989220</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015</td>
      <td>1.326462</td>
      <td>0.976750</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016</td>
      <td>1.440185</td>
      <td>0.953885</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017</td>
      <td>1.299112</td>
      <td>1.081160</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018</td>
      <td>1.310459</td>
      <td>1.079126</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>1.464899</td>
      <td>1.066656</td>
    </tr>
  </tbody>
</table>
</div>




```python
in_result = time_series.merge(forecast_result, on="ds", how="left")

in_result
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
      <th>ds</th>
      <th>y</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1961</td>
      <td>0.143032</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1962</td>
      <td>-0.028398</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1963</td>
      <td>-0.026297</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1964</td>
      <td>-0.122865</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1965</td>
      <td>-0.224154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1966</td>
      <td>0.095070</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1967</td>
      <td>-0.131975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1968</td>
      <td>-0.167841</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1969</td>
      <td>0.105694</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1970</td>
      <td>0.072189</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1971</td>
      <td>-0.177649</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1972</td>
      <td>-0.049936</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1973</td>
      <td>0.199149</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1974</td>
      <td>-0.128841</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1975</td>
      <td>-0.030398</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1976</td>
      <td>-0.210907</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1977</td>
      <td>0.185724</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1978</td>
      <td>0.053986</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1979</td>
      <td>0.230299</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1980</td>
      <td>0.224411</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1981</td>
      <td>0.222159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1982</td>
      <td>0.160740</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1983</td>
      <td>0.348746</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1984</td>
      <td>0.076055</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1985</td>
      <td>0.069280</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1986</td>
      <td>0.139045</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1987</td>
      <td>0.415012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1988</td>
      <td>0.435257</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>0.283534</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1990</td>
      <td>0.579354</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1991</td>
      <td>0.335127</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1992</td>
      <td>0.254460</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1993</td>
      <td>0.243441</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1994</td>
      <td>0.559852</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1995</td>
      <td>0.603439</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1996</td>
      <td>0.317843</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1997</td>
      <td>0.578825</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1998</td>
      <td>0.951884</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1999</td>
      <td>0.732435</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2000</td>
      <td>0.689658</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2001</td>
      <td>0.806679</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2002</td>
      <td>0.917838</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2003</td>
      <td>0.862185</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2004</td>
      <td>0.787869</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2005</td>
      <td>0.886901</td>
      <td>0.811440</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2006</td>
      <td>0.910877</td>
      <td>0.809407</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2007</td>
      <td>1.004826</td>
      <td>0.796937</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2008</td>
      <td>0.813313</td>
      <td>0.774072</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2009</td>
      <td>0.943937</td>
      <td>0.901347</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2010</td>
      <td>1.080097</td>
      <td>0.899313</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2011</td>
      <td>0.863045</td>
      <td>0.886843</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2012</td>
      <td>0.901637</td>
      <td>0.863978</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2013</td>
      <td>0.977131</td>
      <td>0.991253</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2014</td>
      <td>1.131417</td>
      <td>0.989220</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2015</td>
      <td>1.326462</td>
      <td>0.976750</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2016</td>
      <td>1.440185</td>
      <td>0.953885</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2017</td>
      <td>1.299112</td>
      <td>1.081160</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2018</td>
      <td>1.310459</td>
      <td>1.079126</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2019</td>
      <td>1.464899</td>
      <td>1.066656</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.line(
    in_result,
    x="ds",
    y=["y", "yhat"],
    color_discrete_sequence=["black", "red"],
    labels={"ds": "Year"},
    title="Average Global Temperature Change Over Time",
)
```


```python
fig.show()
```




```python
future_df = pd.DataFrame(np.arange(2020, 2046), columns=["ds"])
```


```python
future_df
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
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2026</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2027</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2028</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2029</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2030</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2031</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2032</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2033</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2034</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2035</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2036</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2037</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2038</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2039</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2040</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2041</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2042</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2043</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2044</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2045</td>
    </tr>
  </tbody>
</table>
</div>




```python
out_model = Prophet()
out_model.fit(time_series)
```

    03:50:43 - cmdstanpy - INFO - Chain [1] start processing


    03:50:43 - cmdstanpy - INFO - Chain [1] done processing





    <prophet.forecaster.Prophet at 0x307473d60>




```python
out_forcast = out_model.predict(future_df).loc[:, ["ds", "yhat"]]
out_forcast["ds"] = out_forcast["ds"].apply(lambda x: x.year)
out_forcast
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
      <th>ds</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>1.216595</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>1.328158</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>1.354660</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023</td>
      <td>1.354065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024</td>
      <td>1.326409</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2025</td>
      <td>1.437972</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2026</td>
      <td>1.464474</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2027</td>
      <td>1.463879</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2028</td>
      <td>1.436223</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2029</td>
      <td>1.547786</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2030</td>
      <td>1.574288</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2031</td>
      <td>1.573694</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2032</td>
      <td>1.546037</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2033</td>
      <td>1.657600</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2034</td>
      <td>1.684102</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2035</td>
      <td>1.683508</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2036</td>
      <td>1.655852</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2037</td>
      <td>1.767414</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2038</td>
      <td>1.793916</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2039</td>
      <td>1.793322</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2040</td>
      <td>1.765666</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2041</td>
      <td>1.877229</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2042</td>
      <td>1.903730</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2043</td>
      <td>1.903136</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2044</td>
      <td>1.875480</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2045</td>
      <td>1.987043</td>
    </tr>
  </tbody>
</table>
</div>




```python
out_result_df = pd.concat([time_series, out_forcast], axis=0)
```


```python
out_result_df
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
      <th>ds</th>
      <th>y</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1961</td>
      <td>0.143032</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1962</td>
      <td>-0.028398</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1963</td>
      <td>-0.026297</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1964</td>
      <td>-0.122865</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1965</td>
      <td>-0.224154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2041</td>
      <td>NaN</td>
      <td>1.877229</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2042</td>
      <td>NaN</td>
      <td>1.903730</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2043</td>
      <td>NaN</td>
      <td>1.903136</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2044</td>
      <td>NaN</td>
      <td>1.875480</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2045</td>
      <td>NaN</td>
      <td>1.987043</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 3 columns</p>
</div>




```python
fig = px.line(
    out_result_df,
    x="ds",
    y=["y", "yhat"],
    color_discrete_sequence=["black", "red"],
    labels={"ds": "Year"},
    title="Average Global Temperature Change Over Time",
)
```


```python
fig.show()
```




```python

```
