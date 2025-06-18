---
title: "002_Predicting_City_Houses_With_Scikit-learn_houses_to_rent"
last_modified_at: 
categories:
  - 1일1케글
tags:
  - 
excerpt: "002_Predicting_City_Houses_With_Scikit-learn_houses_to_rent"
use_math: true
classes: wide
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path

BASE_DIR = Path.cwd().parent

csv_path = os.path.join(BASE_DIR, "csv", "houses_to_rent.csv")
df = pd.read_csv(csv_path)
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
      <th>Unnamed: 0</th>
      <th>city</th>
      <th>area</th>
      <th>rooms</th>
      <th>bathroom</th>
      <th>parking spaces</th>
      <th>floor</th>
      <th>animal</th>
      <th>furniture</th>
      <th>hoa</th>
      <th>rent amount</th>
      <th>property tax</th>
      <th>fire insurance</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>240</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>-</td>
      <td>acept</td>
      <td>furnished</td>
      <td>R$0</td>
      <td>R$8,000</td>
      <td>R$1,000</td>
      <td>R$121</td>
      <td>R$9,121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>64</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>acept</td>
      <td>not furnished</td>
      <td>R$540</td>
      <td>R$820</td>
      <td>R$122</td>
      <td>R$11</td>
      <td>R$1,493</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>443</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>acept</td>
      <td>furnished</td>
      <td>R$4,172</td>
      <td>R$7,000</td>
      <td>R$1,417</td>
      <td>R$89</td>
      <td>R$12,680</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>73</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>acept</td>
      <td>not furnished</td>
      <td>R$700</td>
      <td>R$1,250</td>
      <td>R$150</td>
      <td>R$16</td>
      <td>R$2,116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-</td>
      <td>not acept</td>
      <td>not furnished</td>
      <td>R$0</td>
      <td>R$1,200</td>
      <td>R$41</td>
      <td>R$16</td>
      <td>R$1,257</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(df.columns[0], axis=1)
```


```python
df["floor"] = df["floor"].replace("-", 0)
df["animal"] = df["animal"].replace("not acept", 0)
df["animal"] = df["animal"].replace("acept", 1)
df["furniture"] = df["furniture"].replace("furnished", 1)
df["furniture"] = df["furniture"].replace("not furnished", 0)
```

    /var/folders/v7/tlyx9w190ks2gfgzd_j0l5c80000gn/T/ipykernel_21362/4035617042.py:3: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df["animal"] = df["animal"].replace("acept", 1)
    /var/folders/v7/tlyx9w190ks2gfgzd_j0l5c80000gn/T/ipykernel_21362/4035617042.py:5: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df["furniture"] = df["furniture"].replace("not furnished", 0)



```python
columns = [
    "hoa",
    "rent amount",
    "property tax",
    "fire insurance",
    "total",
]

for column in columns:
    df[column] = df[column].str.replace("R$", "", regex=False).str.strip()
    df[column] = df[column].str.replace(",", ".", regex=False).str.strip()
```


```python
df["hoa"] = df["hoa"].replace(to_replace="Sem info", value="0")
df["hoa"] = df["hoa"].replace(to_replace="Incluso", value="0")

df["property tax"] = df["property tax"].replace(to_replace="Incluso", value="0")
```


```python
df.isin(["Incluso"]).any()
```




    city              False
    area              False
    rooms             False
    bathroom          False
    parking spaces    False
    floor             False
    animal            False
    furniture         False
    hoa               False
    rent amount       False
    property tax      False
    fire insurance    False
    total             False
    dtype: bool




```python
df = df.astype(dtype=np.float64)
```


```python
y = df["city"]
X = df.drop("city", axis=1)
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
```


```python
pd.DataFrame(X)
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
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.009351</td>
      <td>0.222222</td>
      <td>0.222222</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.007014</td>
      <td>0.001002</td>
      <td>0.175074</td>
      <td>0.008136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002195</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.101010</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.541082</td>
      <td>0.820641</td>
      <td>0.122244</td>
      <td>0.011869</td>
      <td>0.000493</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.017604</td>
      <td>0.444444</td>
      <td>0.444444</td>
      <td>0.333333</td>
      <td>0.030303</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.004180</td>
      <td>0.006012</td>
      <td>0.001420</td>
      <td>0.127596</td>
      <td>0.011702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002561</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>0.083333</td>
      <td>0.121212</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.701403</td>
      <td>0.000251</td>
      <td>0.150301</td>
      <td>0.019288</td>
      <td>0.001117</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000366</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000200</td>
      <td>0.041082</td>
      <td>0.019288</td>
      <td>0.000257</td>
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
      <th>6075</th>
      <td>0.001626</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.020202</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.420842</td>
      <td>0.000150</td>
      <td>0.000000</td>
      <td>0.017804</td>
      <td>0.000585</td>
    </tr>
    <tr>
      <th>6076</th>
      <td>0.003009</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>0.083333</td>
      <td>0.161616</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.769539</td>
      <td>0.001904</td>
      <td>0.063126</td>
      <td>0.050445</td>
      <td>0.002773</td>
    </tr>
    <tr>
      <th>6077</th>
      <td>0.001545</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.131313</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.250501</td>
      <td>0.950902</td>
      <td>0.042084</td>
      <td>0.014837</td>
      <td>0.000255</td>
    </tr>
    <tr>
      <th>6078</th>
      <td>0.006099</td>
      <td>0.222222</td>
      <td>0.111111</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.002505</td>
      <td>0.250501</td>
      <td>0.074184</td>
      <td>0.002808</td>
    </tr>
    <tr>
      <th>6079</th>
      <td>0.002033</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.040404</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.489980</td>
      <td>0.000902</td>
      <td>0.000000</td>
      <td>0.032641</td>
      <td>0.001416</td>
    </tr>
  </tbody>
</table>
<p>6080 rows × 12 columns</p>
</div>




```python
print(type(X))
print(type(y))

X = pd.DataFrame(X)
```

    <class 'numpy.ndarray'>
    <class 'pandas.core.series.Series'>



```python
df = pd.concat([X, y], axis=1)
```


```python
y = df["city"]
X = df.drop("city", axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)
```


```python
from pycaret.classification import *


setup(data=X_train, target=y_train, session_id=42)
```


<style type="text/css">
#T_d6561_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_d6561">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d6561_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_d6561_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d6561_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_d6561_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_d6561_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_d6561_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_d6561_row1_col1" class="data row1 col1" >city</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_d6561_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_d6561_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d6561_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_d6561_row3_col1" class="data row3 col1" >(4864, 13)</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_d6561_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_d6561_row4_col1" class="data row4 col1" >(4864, 13)</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_d6561_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_d6561_row5_col1" class="data row5 col1" >(3404, 13)</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_d6561_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_d6561_row6_col1" class="data row6 col1" >(1460, 13)</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_d6561_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_d6561_row7_col1" class="data row7 col1" >12</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_d6561_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_d6561_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_d6561_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_d6561_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_d6561_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_d6561_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_d6561_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_d6561_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_d6561_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_d6561_row12_col1" class="data row12 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_d6561_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_d6561_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_d6561_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_d6561_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_d6561_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_d6561_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_d6561_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_d6561_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_d6561_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_d6561_row17_col1" class="data row17 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_d6561_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_d6561_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_d6561_row18_col1" class="data row18 col1" >fa1f</td>
    </tr>
  </tbody>
</table>






    <pycaret.classification.oop.ClassificationExperiment at 0x16cb69be0>




```python
compare_models()
```






<style type="text/css">
#T_78132 th {
  text-align: left;
}
#T_78132_row0_col0, #T_78132_row0_col3, #T_78132_row0_col4, #T_78132_row0_col6, #T_78132_row0_col7, #T_78132_row1_col0, #T_78132_row1_col1, #T_78132_row1_col2, #T_78132_row1_col3, #T_78132_row1_col4, #T_78132_row1_col5, #T_78132_row2_col0, #T_78132_row2_col1, #T_78132_row2_col2, #T_78132_row2_col3, #T_78132_row2_col4, #T_78132_row2_col5, #T_78132_row2_col6, #T_78132_row2_col7, #T_78132_row3_col0, #T_78132_row3_col1, #T_78132_row3_col2, #T_78132_row3_col3, #T_78132_row3_col4, #T_78132_row3_col5, #T_78132_row3_col6, #T_78132_row3_col7, #T_78132_row4_col0, #T_78132_row4_col1, #T_78132_row4_col2, #T_78132_row4_col3, #T_78132_row4_col4, #T_78132_row4_col5, #T_78132_row4_col6, #T_78132_row4_col7, #T_78132_row5_col0, #T_78132_row5_col1, #T_78132_row5_col2, #T_78132_row5_col3, #T_78132_row5_col4, #T_78132_row5_col5, #T_78132_row5_col6, #T_78132_row5_col7, #T_78132_row6_col0, #T_78132_row6_col1, #T_78132_row6_col2, #T_78132_row6_col3, #T_78132_row6_col4, #T_78132_row6_col5, #T_78132_row6_col6, #T_78132_row6_col7, #T_78132_row7_col0, #T_78132_row7_col1, #T_78132_row7_col2, #T_78132_row7_col3, #T_78132_row7_col4, #T_78132_row7_col5, #T_78132_row7_col6, #T_78132_row7_col7, #T_78132_row8_col0, #T_78132_row8_col1, #T_78132_row8_col2, #T_78132_row8_col3, #T_78132_row8_col4, #T_78132_row8_col5, #T_78132_row8_col6, #T_78132_row8_col7, #T_78132_row9_col0, #T_78132_row9_col1, #T_78132_row9_col2, #T_78132_row9_col3, #T_78132_row9_col4, #T_78132_row9_col5, #T_78132_row9_col6, #T_78132_row9_col7, #T_78132_row10_col0, #T_78132_row10_col1, #T_78132_row10_col2, #T_78132_row10_col3, #T_78132_row10_col4, #T_78132_row10_col5, #T_78132_row10_col6, #T_78132_row10_col7, #T_78132_row11_col0, #T_78132_row11_col1, #T_78132_row11_col2, #T_78132_row11_col3, #T_78132_row11_col4, #T_78132_row11_col5, #T_78132_row11_col6, #T_78132_row11_col7, #T_78132_row12_col0, #T_78132_row12_col1, #T_78132_row12_col2, #T_78132_row12_col3, #T_78132_row12_col4, #T_78132_row12_col5, #T_78132_row12_col6, #T_78132_row12_col7, #T_78132_row13_col0, #T_78132_row13_col1, #T_78132_row13_col2, #T_78132_row13_col3, #T_78132_row13_col4, #T_78132_row13_col5, #T_78132_row13_col6, #T_78132_row13_col7, #T_78132_row14_col0, #T_78132_row14_col1, #T_78132_row14_col2, #T_78132_row14_col4, #T_78132_row14_col5, #T_78132_row14_col6, #T_78132_row14_col7, #T_78132_row15_col0, #T_78132_row15_col1, #T_78132_row15_col2, #T_78132_row15_col3, #T_78132_row15_col5, #T_78132_row15_col6, #T_78132_row15_col7 {
  text-align: left;
}
#T_78132_row0_col1, #T_78132_row0_col2, #T_78132_row0_col5, #T_78132_row1_col6, #T_78132_row1_col7, #T_78132_row14_col3, #T_78132_row15_col4 {
  text-align: left;
  background-color: yellow;
}
#T_78132_row0_col8, #T_78132_row1_col8, #T_78132_row2_col8, #T_78132_row3_col8, #T_78132_row4_col8, #T_78132_row5_col8, #T_78132_row6_col8, #T_78132_row7_col8, #T_78132_row8_col8, #T_78132_row9_col8, #T_78132_row10_col8, #T_78132_row11_col8, #T_78132_row15_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_78132_row12_col8, #T_78132_row13_col8, #T_78132_row14_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_78132">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_78132_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_78132_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_78132_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_78132_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_78132_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_78132_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_78132_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_78132_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_78132_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_78132_level0_row0" class="row_heading level0 row0" >catboost</th>
      <td id="T_78132_row0_col0" class="data row0 col0" >CatBoost Classifier</td>
      <td id="T_78132_row0_col1" class="data row0 col1" >0.9098</td>
      <td id="T_78132_row0_col2" class="data row0 col2" >0.8996</td>
      <td id="T_78132_row0_col3" class="data row0 col3" >0.9833</td>
      <td id="T_78132_row0_col4" class="data row0 col4" >0.9181</td>
      <td id="T_78132_row0_col5" class="data row0 col5" >0.9496</td>
      <td id="T_78132_row0_col6" class="data row0 col6" >0.5266</td>
      <td id="T_78132_row0_col7" class="data row0 col7" >0.5582</td>
      <td id="T_78132_row0_col8" class="data row0 col8" >0.3690</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row1" class="row_heading level0 row1" >xgboost</th>
      <td id="T_78132_row1_col0" class="data row1 col0" >Extreme Gradient Boosting</td>
      <td id="T_78132_row1_col1" class="data row1 col1" >0.9095</td>
      <td id="T_78132_row1_col2" class="data row1 col2" >0.8906</td>
      <td id="T_78132_row1_col3" class="data row1 col3" >0.9724</td>
      <td id="T_78132_row1_col4" class="data row1 col4" >0.9265</td>
      <td id="T_78132_row1_col5" class="data row1 col5" >0.9489</td>
      <td id="T_78132_row1_col6" class="data row1 col6" >0.5571</td>
      <td id="T_78132_row1_col7" class="data row1 col7" >0.5712</td>
      <td id="T_78132_row1_col8" class="data row1 col8" >0.0140</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row2" class="row_heading level0 row2" >rf</th>
      <td id="T_78132_row2_col0" class="data row2 col0" >Random Forest Classifier</td>
      <td id="T_78132_row2_col1" class="data row2 col1" >0.9060</td>
      <td id="T_78132_row2_col2" class="data row2 col2" >0.8913</td>
      <td id="T_78132_row2_col3" class="data row2 col3" >0.9840</td>
      <td id="T_78132_row2_col4" class="data row2 col4" >0.9138</td>
      <td id="T_78132_row2_col5" class="data row2 col5" >0.9476</td>
      <td id="T_78132_row2_col6" class="data row2 col6" >0.4971</td>
      <td id="T_78132_row2_col7" class="data row2 col7" >0.5334</td>
      <td id="T_78132_row2_col8" class="data row2 col8" >0.0500</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row3" class="row_heading level0 row3" >lightgbm</th>
      <td id="T_78132_row3_col0" class="data row3 col0" >Light Gradient Boosting Machine</td>
      <td id="T_78132_row3_col1" class="data row3 col1" >0.9054</td>
      <td id="T_78132_row3_col2" class="data row3 col2" >0.8938</td>
      <td id="T_78132_row3_col3" class="data row3 col3" >0.9721</td>
      <td id="T_78132_row3_col4" class="data row3 col4" >0.9226</td>
      <td id="T_78132_row3_col5" class="data row3 col5" >0.9467</td>
      <td id="T_78132_row3_col6" class="data row3 col6" >0.5310</td>
      <td id="T_78132_row3_col7" class="data row3 col7" >0.5475</td>
      <td id="T_78132_row3_col8" class="data row3 col8" >0.4670</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row4" class="row_heading level0 row4" >et</th>
      <td id="T_78132_row4_col0" class="data row4 col0" >Extra Trees Classifier</td>
      <td id="T_78132_row4_col1" class="data row4 col1" >0.9051</td>
      <td id="T_78132_row4_col2" class="data row4 col2" >0.8907</td>
      <td id="T_78132_row4_col3" class="data row4 col3" >0.9830</td>
      <td id="T_78132_row4_col4" class="data row4 col4" >0.9137</td>
      <td id="T_78132_row4_col5" class="data row4 col5" >0.9471</td>
      <td id="T_78132_row4_col6" class="data row4 col6" >0.4945</td>
      <td id="T_78132_row4_col7" class="data row4 col7" >0.5283</td>
      <td id="T_78132_row4_col8" class="data row4 col8" >0.0360</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row5" class="row_heading level0 row5" >gbc</th>
      <td id="T_78132_row5_col0" class="data row5 col0" >Gradient Boosting Classifier</td>
      <td id="T_78132_row5_col1" class="data row5 col1" >0.8995</td>
      <td id="T_78132_row5_col2" class="data row5 col2" >0.8716</td>
      <td id="T_78132_row5_col3" class="data row5 col3" >0.9823</td>
      <td id="T_78132_row5_col4" class="data row5 col4" >0.9088</td>
      <td id="T_78132_row5_col5" class="data row5 col5" >0.9441</td>
      <td id="T_78132_row5_col6" class="data row5 col6" >0.4564</td>
      <td id="T_78132_row5_col7" class="data row5 col7" >0.4965</td>
      <td id="T_78132_row5_col8" class="data row5 col8" >0.0600</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row6" class="row_heading level0 row6" >ada</th>
      <td id="T_78132_row6_col0" class="data row6 col0" >Ada Boost Classifier</td>
      <td id="T_78132_row6_col1" class="data row6 col1" >0.8969</td>
      <td id="T_78132_row6_col2" class="data row6 col2" >0.8442</td>
      <td id="T_78132_row6_col3" class="data row6 col3" >0.9827</td>
      <td id="T_78132_row6_col4" class="data row6 col4" >0.9060</td>
      <td id="T_78132_row6_col5" class="data row6 col5" >0.9427</td>
      <td id="T_78132_row6_col6" class="data row6 col6" >0.4340</td>
      <td id="T_78132_row6_col7" class="data row6 col7" >0.4771</td>
      <td id="T_78132_row6_col8" class="data row6 col8" >0.0250</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row7" class="row_heading level0 row7" >svm</th>
      <td id="T_78132_row7_col0" class="data row7 col0" >SVM - Linear Kernel</td>
      <td id="T_78132_row7_col1" class="data row7 col1" >0.8890</td>
      <td id="T_78132_row7_col2" class="data row7 col2" >0.7206</td>
      <td id="T_78132_row7_col3" class="data row7 col3" >0.9949</td>
      <td id="T_78132_row7_col4" class="data row7 col4" >0.8896</td>
      <td id="T_78132_row7_col5" class="data row7 col5" >0.9393</td>
      <td id="T_78132_row7_col6" class="data row7 col6" >0.3084</td>
      <td id="T_78132_row7_col7" class="data row7 col7" >0.3988</td>
      <td id="T_78132_row7_col8" class="data row7 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row8" class="row_heading level0 row8" >ridge</th>
      <td id="T_78132_row8_col0" class="data row8 col0" >Ridge Classifier</td>
      <td id="T_78132_row8_col1" class="data row8 col1" >0.8878</td>
      <td id="T_78132_row8_col2" class="data row8 col2" >0.7899</td>
      <td id="T_78132_row8_col3" class="data row8 col3" >0.9935</td>
      <td id="T_78132_row8_col4" class="data row8 col4" >0.8895</td>
      <td id="T_78132_row8_col5" class="data row8 col5" >0.9386</td>
      <td id="T_78132_row8_col6" class="data row8 col6" >0.3050</td>
      <td id="T_78132_row8_col7" class="data row8 col7" >0.3910</td>
      <td id="T_78132_row8_col8" class="data row8 col8" >0.0050</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row9" class="row_heading level0 row9" >knn</th>
      <td id="T_78132_row9_col0" class="data row9 col0" >K Neighbors Classifier</td>
      <td id="T_78132_row9_col1" class="data row9 col1" >0.8863</td>
      <td id="T_78132_row9_col2" class="data row9 col2" >0.7918</td>
      <td id="T_78132_row9_col3" class="data row9 col3" >0.9704</td>
      <td id="T_78132_row9_col4" class="data row9 col4" >0.9049</td>
      <td id="T_78132_row9_col5" class="data row9 col5" >0.9365</td>
      <td id="T_78132_row9_col6" class="data row9 col6" >0.4002</td>
      <td id="T_78132_row9_col7" class="data row9 col7" >0.4245</td>
      <td id="T_78132_row9_col8" class="data row9 col8" >0.0100</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row10" class="row_heading level0 row10" >lr</th>
      <td id="T_78132_row10_col0" class="data row10 col0" >Logistic Regression</td>
      <td id="T_78132_row10_col1" class="data row10 col1" >0.8860</td>
      <td id="T_78132_row10_col2" class="data row10 col2" >0.7792</td>
      <td id="T_78132_row10_col3" class="data row10 col3" >0.9929</td>
      <td id="T_78132_row10_col4" class="data row10 col4" >0.8884</td>
      <td id="T_78132_row10_col5" class="data row10 col5" >0.9377</td>
      <td id="T_78132_row10_col6" class="data row10 col6" >0.2923</td>
      <td id="T_78132_row10_col7" class="data row10 col7" >0.3768</td>
      <td id="T_78132_row10_col8" class="data row10 col8" >0.3090</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row11" class="row_heading level0 row11" >nb</th>
      <td id="T_78132_row11_col0" class="data row11 col0" >Naive Bayes</td>
      <td id="T_78132_row11_col1" class="data row11 col1" >0.8837</td>
      <td id="T_78132_row11_col2" class="data row11 col2" >0.7112</td>
      <td id="T_78132_row11_col3" class="data row11 col3" >0.9813</td>
      <td id="T_78132_row11_col4" class="data row11 col4" >0.8944</td>
      <td id="T_78132_row11_col5" class="data row11 col5" >0.9358</td>
      <td id="T_78132_row11_col6" class="data row11 col6" >0.3300</td>
      <td id="T_78132_row11_col7" class="data row11 col7" >0.3760</td>
      <td id="T_78132_row11_col8" class="data row11 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row12" class="row_heading level0 row12" >qda</th>
      <td id="T_78132_row12_col0" class="data row12 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_78132_row12_col1" class="data row12 col1" >0.8834</td>
      <td id="T_78132_row12_col2" class="data row12 col2" >0.7886</td>
      <td id="T_78132_row12_col3" class="data row12 col3" >0.9806</td>
      <td id="T_78132_row12_col4" class="data row12 col4" >0.8946</td>
      <td id="T_78132_row12_col5" class="data row12 col5" >0.9356</td>
      <td id="T_78132_row12_col6" class="data row12 col6" >0.3312</td>
      <td id="T_78132_row12_col7" class="data row12 col7" >0.3762</td>
      <td id="T_78132_row12_col8" class="data row12 col8" >0.0040</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row13" class="row_heading level0 row13" >lda</th>
      <td id="T_78132_row13_col0" class="data row13 col0" >Linear Discriminant Analysis</td>
      <td id="T_78132_row13_col1" class="data row13 col1" >0.8796</td>
      <td id="T_78132_row13_col2" class="data row13 col2" >0.7941</td>
      <td id="T_78132_row13_col3" class="data row13 col3" >0.9833</td>
      <td id="T_78132_row13_col4" class="data row13 col4" >0.8890</td>
      <td id="T_78132_row13_col5" class="data row13 col5" >0.9338</td>
      <td id="T_78132_row13_col6" class="data row13 col6" >0.2844</td>
      <td id="T_78132_row13_col7" class="data row13 col7" >0.3387</td>
      <td id="T_78132_row13_col8" class="data row13 col8" >0.0040</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row14" class="row_heading level0 row14" >dummy</th>
      <td id="T_78132_row14_col0" class="data row14 col0" >Dummy Classifier</td>
      <td id="T_78132_row14_col1" class="data row14 col1" >0.8637</td>
      <td id="T_78132_row14_col2" class="data row14 col2" >0.5000</td>
      <td id="T_78132_row14_col3" class="data row14 col3" >1.0000</td>
      <td id="T_78132_row14_col4" class="data row14 col4" >0.8637</td>
      <td id="T_78132_row14_col5" class="data row14 col5" >0.9269</td>
      <td id="T_78132_row14_col6" class="data row14 col6" >0.0000</td>
      <td id="T_78132_row14_col7" class="data row14 col7" >0.0000</td>
      <td id="T_78132_row14_col8" class="data row14 col8" >0.0040</td>
    </tr>
    <tr>
      <th id="T_78132_level0_row15" class="row_heading level0 row15" >dt</th>
      <td id="T_78132_row15_col0" class="data row15 col0" >Decision Tree Classifier</td>
      <td id="T_78132_row15_col1" class="data row15 col1" >0.8599</td>
      <td id="T_78132_row15_col2" class="data row15 col2" >0.7262</td>
      <td id="T_78132_row15_col3" class="data row15 col3" >0.9099</td>
      <td id="T_78132_row15_col4" class="data row15 col4" >0.9266</td>
      <td id="T_78132_row15_col5" class="data row15 col5" >0.9181</td>
      <td id="T_78132_row15_col6" class="data row15 col6" >0.4325</td>
      <td id="T_78132_row15_col7" class="data row15 col7" >0.4339</td>
      <td id="T_78132_row15_col8" class="data row15 col8" >0.0080</td>
    </tr>
  </tbody>
</table>










    <catboost.core.CatBoostClassifier at 0x313ea3670>




```python

```
