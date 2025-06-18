---
title: "Analyzing Diamond Price Data diamonds"
date: 2024-01-05
last_modified_at: 2024-01-05
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Analyzing Diamond Price Data diamonds 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shivam2503/diamonds")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/shivam2503/diamonds/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import os

df = pd.read_csv(os.path.join(path, "diamonds.csv"))
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
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
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
      <th>53935</th>
      <td>53936</td>
      <td>0.72</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI1</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>2757</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>53937</td>
      <td>0.72</td>
      <td>Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>53938</td>
      <td>0.70</td>
      <td>Very Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>2757</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>53939</td>
      <td>0.86</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>2757</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>53940</td>
      <td>0.75</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI2</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 11 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```


```python
df = df.drop(df.columns[0], axis=1)
```


```python
X = df.drop("price", axis=1)
y = df["price"]
```


```python
col_list = [
    "cut",
    "color",
    "clarity",
]

for col in col_list:
    print(X[col].unique())
```

    ['Ideal' 'Premium' 'Good' 'Very Good' 'Fair']
    ['E' 'I' 'J' 'H' 'F' 'G' 'D']
    ['SI2' 'SI1' 'VS1' 'VS2' 'VVS2' 'VVS1' 'I1' 'IF']



```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
cut_mappings = []

for col in col_list:
    X[col] = encoder.fit_transform(X[col])
    cut_mappings.append({index: label for index, label in enumerate(encoder.classes_)})
```


```python
print(cut_mappings[0])
print(cut_mappings[1])
print(cut_mappings[2])
```

    {0: 'Fair', 1: 'Good', 2: 'Ideal', 3: 'Premium', 4: 'Very Good'}
    {0: 'D', 1: 'E', 2: 'F', 3: 'G', 4: 'H', 5: 'I', 6: 'J'}
    {0: 'I1', 1: 'IF', 2: 'SI1', 3: 'SI2', 4: 'VS1', 5: 'VS2', 6: 'VVS1', 7: 'VVS2'}



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
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
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
    </tr>
    <tr>
      <th>53935</th>
      <td>0.72</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>0.72</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>0.70</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>0.86</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>0.75</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 9 columns</p>
</div>




```python
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```


```python
X = pd.DataFrame(X)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.006237</td>
      <td>0.50</td>
      <td>0.166667</td>
      <td>0.428571</td>
      <td>0.513889</td>
      <td>0.230769</td>
      <td>0.367784</td>
      <td>0.067572</td>
      <td>0.076415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002079</td>
      <td>0.75</td>
      <td>0.166667</td>
      <td>0.285714</td>
      <td>0.466667</td>
      <td>0.346154</td>
      <td>0.362197</td>
      <td>0.065195</td>
      <td>0.072642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.006237</td>
      <td>0.25</td>
      <td>0.166667</td>
      <td>0.571429</td>
      <td>0.386111</td>
      <td>0.423077</td>
      <td>0.377095</td>
      <td>0.069100</td>
      <td>0.072642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.018711</td>
      <td>0.75</td>
      <td>0.833333</td>
      <td>0.714286</td>
      <td>0.538889</td>
      <td>0.288462</td>
      <td>0.391061</td>
      <td>0.071817</td>
      <td>0.082704</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.022869</td>
      <td>0.25</td>
      <td>1.000000</td>
      <td>0.428571</td>
      <td>0.563889</td>
      <td>0.288462</td>
      <td>0.404097</td>
      <td>0.073854</td>
      <td>0.086478</td>
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
    </tr>
    <tr>
      <th>53935</th>
      <td>0.108108</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>0.494444</td>
      <td>0.269231</td>
      <td>0.535382</td>
      <td>0.097793</td>
      <td>0.110063</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>0.108108</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>0.558333</td>
      <td>0.230769</td>
      <td>0.529795</td>
      <td>0.097623</td>
      <td>0.113522</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>0.103950</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>0.550000</td>
      <td>0.326923</td>
      <td>0.527002</td>
      <td>0.096435</td>
      <td>0.111950</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>0.137214</td>
      <td>0.75</td>
      <td>0.666667</td>
      <td>0.428571</td>
      <td>0.500000</td>
      <td>0.288462</td>
      <td>0.572626</td>
      <td>0.103905</td>
      <td>0.117610</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>0.114345</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>0.533333</td>
      <td>0.230769</td>
      <td>0.542831</td>
      <td>0.099660</td>
      <td>0.114465</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 9 columns</p>
</div>




```python
y = pd.DataFrame(y)
```


```python
df = pd.concat([X, y], axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
```


```python
from pycaret.regression import setup, compare_models
```


```python
setup(data=X_train, target=y_train)
```


<style type="text/css">
#T_ded4a_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_ded4a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ded4a_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_ded4a_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ded4a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_ded4a_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_ded4a_row0_col1" class="data row0 col1" >2666</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_ded4a_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_ded4a_row1_col1" class="data row1 col1" >8</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_ded4a_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_ded4a_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_ded4a_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_ded4a_row3_col1" class="data row3 col1" >(43152, 9)</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_ded4a_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_ded4a_row4_col1" class="data row4 col1" >(43152, 9)</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_ded4a_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_ded4a_row5_col1" class="data row5 col1" >(30206, 9)</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_ded4a_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_ded4a_row6_col1" class="data row6 col1" >(12946, 9)</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_ded4a_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_ded4a_row7_col1" class="data row7 col1" >8</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_ded4a_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_ded4a_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_ded4a_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_ded4a_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_ded4a_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_ded4a_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_ded4a_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_ded4a_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_ded4a_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_ded4a_row12_col1" class="data row12 col1" >KFold</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_ded4a_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_ded4a_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_ded4a_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_ded4a_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_ded4a_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_ded4a_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_ded4a_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_ded4a_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_ded4a_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_ded4a_row17_col1" class="data row17 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_ded4a_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_ded4a_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_ded4a_row18_col1" class="data row18 col1" >05fe</td>
    </tr>
  </tbody>
</table>






    <pycaret.regression.oop.RegressionExperiment at 0x31600ef40>




```python
compare_models()
```






<style type="text/css">
#T_0be90 th {
  text-align: left;
}
#T_0be90_row0_col0, #T_0be90_row0_col1, #T_0be90_row0_col6, #T_0be90_row1_col0, #T_0be90_row1_col1, #T_0be90_row1_col3, #T_0be90_row1_col4, #T_0be90_row1_col5, #T_0be90_row1_col6, #T_0be90_row2_col0, #T_0be90_row2_col1, #T_0be90_row2_col3, #T_0be90_row2_col4, #T_0be90_row2_col5, #T_0be90_row2_col6, #T_0be90_row3_col0, #T_0be90_row3_col1, #T_0be90_row3_col3, #T_0be90_row3_col4, #T_0be90_row3_col5, #T_0be90_row3_col6, #T_0be90_row4_col0, #T_0be90_row4_col1, #T_0be90_row4_col3, #T_0be90_row4_col4, #T_0be90_row4_col5, #T_0be90_row4_col6, #T_0be90_row5_col0, #T_0be90_row5_col1, #T_0be90_row5_col3, #T_0be90_row5_col4, #T_0be90_row5_col5, #T_0be90_row5_col6, #T_0be90_row6_col0, #T_0be90_row6_col3, #T_0be90_row6_col4, #T_0be90_row6_col5, #T_0be90_row6_col6, #T_0be90_row7_col0, #T_0be90_row7_col3, #T_0be90_row7_col4, #T_0be90_row7_col5, #T_0be90_row8_col0, #T_0be90_row8_col1, #T_0be90_row8_col3, #T_0be90_row8_col4, #T_0be90_row8_col5, #T_0be90_row8_col6, #T_0be90_row9_col0, #T_0be90_row9_col3, #T_0be90_row9_col4, #T_0be90_row9_col5, #T_0be90_row9_col6, #T_0be90_row10_col0, #T_0be90_row10_col1, #T_0be90_row10_col3, #T_0be90_row10_col4, #T_0be90_row10_col5, #T_0be90_row10_col6, #T_0be90_row11_col0, #T_0be90_row11_col1, #T_0be90_row11_col3, #T_0be90_row11_col4, #T_0be90_row11_col5, #T_0be90_row11_col6, #T_0be90_row12_col0, #T_0be90_row12_col1, #T_0be90_row12_col3, #T_0be90_row12_col4, #T_0be90_row12_col5, #T_0be90_row12_col6, #T_0be90_row13_col0, #T_0be90_row13_col1, #T_0be90_row13_col3, #T_0be90_row13_col4, #T_0be90_row13_col5, #T_0be90_row13_col6, #T_0be90_row14_col0, #T_0be90_row14_col1, #T_0be90_row14_col2, #T_0be90_row14_col3, #T_0be90_row14_col4, #T_0be90_row14_col5, #T_0be90_row14_col6, #T_0be90_row15_col0, #T_0be90_row15_col1, #T_0be90_row15_col2, #T_0be90_row15_col3, #T_0be90_row15_col4, #T_0be90_row15_col5, #T_0be90_row15_col6, #T_0be90_row16_col0, #T_0be90_row16_col1, #T_0be90_row16_col2, #T_0be90_row16_col3, #T_0be90_row16_col4, #T_0be90_row16_col5, #T_0be90_row16_col6, #T_0be90_row17_col0, #T_0be90_row17_col1, #T_0be90_row17_col2, #T_0be90_row17_col3, #T_0be90_row17_col4, #T_0be90_row17_col5, #T_0be90_row17_col6, #T_0be90_row18_col0, #T_0be90_row18_col1, #T_0be90_row18_col2, #T_0be90_row18_col3, #T_0be90_row18_col4, #T_0be90_row18_col5, #T_0be90_row18_col6, #T_0be90_row19_col0, #T_0be90_row19_col1, #T_0be90_row19_col2, #T_0be90_row19_col3, #T_0be90_row19_col4, #T_0be90_row19_col5, #T_0be90_row19_col6 {
  text-align: left;
}
#T_0be90_row0_col2, #T_0be90_row0_col3, #T_0be90_row0_col4, #T_0be90_row0_col5, #T_0be90_row1_col2, #T_0be90_row2_col2, #T_0be90_row3_col2, #T_0be90_row4_col2, #T_0be90_row5_col2, #T_0be90_row6_col1, #T_0be90_row6_col2, #T_0be90_row7_col1, #T_0be90_row7_col2, #T_0be90_row7_col6, #T_0be90_row8_col2, #T_0be90_row9_col1, #T_0be90_row9_col2, #T_0be90_row10_col2, #T_0be90_row11_col2, #T_0be90_row12_col2, #T_0be90_row13_col2 {
  text-align: left;
  background-color: yellow;
}
#T_0be90_row0_col7, #T_0be90_row1_col7, #T_0be90_row2_col7, #T_0be90_row3_col7, #T_0be90_row4_col7, #T_0be90_row5_col7, #T_0be90_row6_col7, #T_0be90_row7_col7, #T_0be90_row8_col7, #T_0be90_row9_col7, #T_0be90_row10_col7, #T_0be90_row11_col7, #T_0be90_row12_col7, #T_0be90_row13_col7, #T_0be90_row14_col7, #T_0be90_row15_col7, #T_0be90_row16_col7, #T_0be90_row17_col7, #T_0be90_row19_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_0be90_row18_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_0be90">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0be90_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_0be90_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_0be90_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_0be90_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_0be90_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_0be90_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_0be90_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_0be90_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0be90_level0_row0" class="row_heading level0 row0" >ridge</th>
      <td id="T_0be90_row0_col0" class="data row0 col0" >Ridge Regression</td>
      <td id="T_0be90_row0_col1" class="data row0 col1" >0.0006</td>
      <td id="T_0be90_row0_col2" class="data row0 col2" >0.0000</td>
      <td id="T_0be90_row0_col3" class="data row0 col3" >0.0037</td>
      <td id="T_0be90_row0_col4" class="data row0 col4" >0.9509</td>
      <td id="T_0be90_row0_col5" class="data row0 col5" >0.0030</td>
      <td id="T_0be90_row0_col6" class="data row0 col6" >0.0052</td>
      <td id="T_0be90_row0_col7" class="data row0 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row1" class="row_heading level0 row1" >gbr</th>
      <td id="T_0be90_row1_col0" class="data row1 col0" >Gradient Boosting Regressor</td>
      <td id="T_0be90_row1_col1" class="data row1 col1" >0.0006</td>
      <td id="T_0be90_row1_col2" class="data row1 col2" >0.0000</td>
      <td id="T_0be90_row1_col3" class="data row1 col3" >0.0042</td>
      <td id="T_0be90_row1_col4" class="data row1 col4" >0.9473</td>
      <td id="T_0be90_row1_col5" class="data row1 col5" >0.0034</td>
      <td id="T_0be90_row1_col6" class="data row1 col6" >0.0048</td>
      <td id="T_0be90_row1_col7" class="data row1 col7" >0.2250</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row2" class="row_heading level0 row2" >br</th>
      <td id="T_0be90_row2_col0" class="data row2 col0" >Bayesian Ridge</td>
      <td id="T_0be90_row2_col1" class="data row2 col1" >0.0005</td>
      <td id="T_0be90_row2_col2" class="data row2 col2" >0.0000</td>
      <td id="T_0be90_row2_col3" class="data row2 col3" >0.0040</td>
      <td id="T_0be90_row2_col4" class="data row2 col4" >0.9462</td>
      <td id="T_0be90_row2_col5" class="data row2 col5" >0.0032</td>
      <td id="T_0be90_row2_col6" class="data row2 col6" >0.0045</td>
      <td id="T_0be90_row2_col7" class="data row2 col7" >0.0070</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row3" class="row_heading level0 row3" >lr</th>
      <td id="T_0be90_row3_col0" class="data row3 col0" >Linear Regression</td>
      <td id="T_0be90_row3_col1" class="data row3 col1" >0.0005</td>
      <td id="T_0be90_row3_col2" class="data row3 col2" >0.0000</td>
      <td id="T_0be90_row3_col3" class="data row3 col3" >0.0040</td>
      <td id="T_0be90_row3_col4" class="data row3 col4" >0.9456</td>
      <td id="T_0be90_row3_col5" class="data row3 col5" >0.0032</td>
      <td id="T_0be90_row3_col6" class="data row3 col6" >0.0045</td>
      <td id="T_0be90_row3_col7" class="data row3 col7" >0.2310</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row4" class="row_heading level0 row4" >lar</th>
      <td id="T_0be90_row4_col0" class="data row4 col0" >Least Angle Regression</td>
      <td id="T_0be90_row4_col1" class="data row4 col1" >0.0005</td>
      <td id="T_0be90_row4_col2" class="data row4 col2" >0.0000</td>
      <td id="T_0be90_row4_col3" class="data row4 col3" >0.0040</td>
      <td id="T_0be90_row4_col4" class="data row4 col4" >0.9456</td>
      <td id="T_0be90_row4_col5" class="data row4 col5" >0.0032</td>
      <td id="T_0be90_row4_col6" class="data row4 col6" >0.0045</td>
      <td id="T_0be90_row4_col7" class="data row4 col7" >0.0070</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row5" class="row_heading level0 row5" >lightgbm</th>
      <td id="T_0be90_row5_col0" class="data row5 col0" >Light Gradient Boosting Machine</td>
      <td id="T_0be90_row5_col1" class="data row5 col1" >0.0007</td>
      <td id="T_0be90_row5_col2" class="data row5 col2" >0.0000</td>
      <td id="T_0be90_row5_col3" class="data row5 col3" >0.0043</td>
      <td id="T_0be90_row5_col4" class="data row5 col4" >0.9454</td>
      <td id="T_0be90_row5_col5" class="data row5 col5" >0.0035</td>
      <td id="T_0be90_row5_col6" class="data row5 col6" >0.0054</td>
      <td id="T_0be90_row5_col7" class="data row5 col7" >0.4080</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row6" class="row_heading level0 row6" >et</th>
      <td id="T_0be90_row6_col0" class="data row6 col0" >Extra Trees Regressor</td>
      <td id="T_0be90_row6_col1" class="data row6 col1" >0.0004</td>
      <td id="T_0be90_row6_col2" class="data row6 col2" >0.0000</td>
      <td id="T_0be90_row6_col3" class="data row6 col3" >0.0043</td>
      <td id="T_0be90_row6_col4" class="data row6 col4" >0.9446</td>
      <td id="T_0be90_row6_col5" class="data row6 col5" >0.0035</td>
      <td id="T_0be90_row6_col6" class="data row6 col6" >0.0028</td>
      <td id="T_0be90_row6_col7" class="data row6 col7" >0.4940</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row7" class="row_heading level0 row7" >rf</th>
      <td id="T_0be90_row7_col0" class="data row7 col0" >Random Forest Regressor</td>
      <td id="T_0be90_row7_col1" class="data row7 col1" >0.0004</td>
      <td id="T_0be90_row7_col2" class="data row7 col2" >0.0000</td>
      <td id="T_0be90_row7_col3" class="data row7 col3" >0.0044</td>
      <td id="T_0be90_row7_col4" class="data row7 col4" >0.9428</td>
      <td id="T_0be90_row7_col5" class="data row7 col5" >0.0036</td>
      <td id="T_0be90_row7_col6" class="data row7 col6" >0.0026</td>
      <td id="T_0be90_row7_col7" class="data row7 col7" >0.6630</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row8" class="row_heading level0 row8" >huber</th>
      <td id="T_0be90_row8_col0" class="data row8 col0" >Huber Regressor</td>
      <td id="T_0be90_row8_col1" class="data row8 col1" >0.0005</td>
      <td id="T_0be90_row8_col2" class="data row8 col2" >0.0000</td>
      <td id="T_0be90_row8_col3" class="data row8 col3" >0.0043</td>
      <td id="T_0be90_row8_col4" class="data row8 col4" >0.9378</td>
      <td id="T_0be90_row8_col5" class="data row8 col5" >0.0033</td>
      <td id="T_0be90_row8_col6" class="data row8 col6" >0.0042</td>
      <td id="T_0be90_row8_col7" class="data row8 col7" >0.0420</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row9" class="row_heading level0 row9" >catboost</th>
      <td id="T_0be90_row9_col0" class="data row9 col0" >CatBoost Regressor</td>
      <td id="T_0be90_row9_col1" class="data row9 col1" >0.0004</td>
      <td id="T_0be90_row9_col2" class="data row9 col2" >0.0000</td>
      <td id="T_0be90_row9_col3" class="data row9 col3" >0.0045</td>
      <td id="T_0be90_row9_col4" class="data row9 col4" >0.9377</td>
      <td id="T_0be90_row9_col5" class="data row9 col5" >0.0037</td>
      <td id="T_0be90_row9_col6" class="data row9 col6" >0.0033</td>
      <td id="T_0be90_row9_col7" class="data row9 col7" >0.5760</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row10" class="row_heading level0 row10" >omp</th>
      <td id="T_0be90_row10_col0" class="data row10 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_0be90_row10_col1" class="data row10 col1" >0.0020</td>
      <td id="T_0be90_row10_col2" class="data row10 col2" >0.0000</td>
      <td id="T_0be90_row10_col3" class="data row10 col3" >0.0050</td>
      <td id="T_0be90_row10_col4" class="data row10 col4" >0.9362</td>
      <td id="T_0be90_row10_col5" class="data row10 col5" >0.0041</td>
      <td id="T_0be90_row10_col6" class="data row10 col6" >0.0172</td>
      <td id="T_0be90_row10_col7" class="data row10 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row11" class="row_heading level0 row11" >xgboost</th>
      <td id="T_0be90_row11_col0" class="data row11 col0" >Extreme Gradient Boosting</td>
      <td id="T_0be90_row11_col1" class="data row11 col1" >0.0006</td>
      <td id="T_0be90_row11_col2" class="data row11 col2" >0.0000</td>
      <td id="T_0be90_row11_col3" class="data row11 col3" >0.0051</td>
      <td id="T_0be90_row11_col4" class="data row11 col4" >0.9300</td>
      <td id="T_0be90_row11_col5" class="data row11 col5" >0.0041</td>
      <td id="T_0be90_row11_col6" class="data row11 col6" >0.0045</td>
      <td id="T_0be90_row11_col7" class="data row11 col7" >0.0220</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row12" class="row_heading level0 row12" >dt</th>
      <td id="T_0be90_row12_col0" class="data row12 col0" >Decision Tree Regressor</td>
      <td id="T_0be90_row12_col1" class="data row12 col1" >0.0005</td>
      <td id="T_0be90_row12_col2" class="data row12 col2" >0.0000</td>
      <td id="T_0be90_row12_col3" class="data row12 col3" >0.0052</td>
      <td id="T_0be90_row12_col4" class="data row12 col4" >0.9286</td>
      <td id="T_0be90_row12_col5" class="data row12 col5" >0.0044</td>
      <td id="T_0be90_row12_col6" class="data row12 col6" >0.0036</td>
      <td id="T_0be90_row12_col7" class="data row12 col7" >0.0190</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row13" class="row_heading level0 row13" >knn</th>
      <td id="T_0be90_row13_col0" class="data row13 col0" >K Neighbors Regressor</td>
      <td id="T_0be90_row13_col1" class="data row13 col1" >0.0016</td>
      <td id="T_0be90_row13_col2" class="data row13 col2" >0.0000</td>
      <td id="T_0be90_row13_col3" class="data row13 col3" >0.0055</td>
      <td id="T_0be90_row13_col4" class="data row13 col4" >0.9249</td>
      <td id="T_0be90_row13_col5" class="data row13 col5" >0.0046</td>
      <td id="T_0be90_row13_col6" class="data row13 col6" >0.0136</td>
      <td id="T_0be90_row13_col7" class="data row13 col7" >0.0160</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row14" class="row_heading level0 row14" >ada</th>
      <td id="T_0be90_row14_col0" class="data row14 col0" >AdaBoost Regressor</td>
      <td id="T_0be90_row14_col1" class="data row14 col1" >0.0108</td>
      <td id="T_0be90_row14_col2" class="data row14 col2" >0.0003</td>
      <td id="T_0be90_row14_col3" class="data row14 col3" >0.0168</td>
      <td id="T_0be90_row14_col4" class="data row14 col4" >0.4260</td>
      <td id="T_0be90_row14_col5" class="data row14 col5" >0.0144</td>
      <td id="T_0be90_row14_col6" class="data row14 col6" >0.0875</td>
      <td id="T_0be90_row14_col7" class="data row14 col7" >0.3080</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row15" class="row_heading level0 row15" >lasso</th>
      <td id="T_0be90_row15_col0" class="data row15 col0" >Lasso Regression</td>
      <td id="T_0be90_row15_col1" class="data row15 col1" >0.0186</td>
      <td id="T_0be90_row15_col2" class="data row15 col2" >0.0005</td>
      <td id="T_0be90_row15_col3" class="data row15 col3" >0.0224</td>
      <td id="T_0be90_row15_col4" class="data row15 col4" >-0.0005</td>
      <td id="T_0be90_row15_col5" class="data row15 col5" >0.0199</td>
      <td id="T_0be90_row15_col6" class="data row15 col6" >0.1720</td>
      <td id="T_0be90_row15_col7" class="data row15 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row16" class="row_heading level0 row16" >en</th>
      <td id="T_0be90_row16_col0" class="data row16 col0" >Elastic Net</td>
      <td id="T_0be90_row16_col1" class="data row16 col1" >0.0186</td>
      <td id="T_0be90_row16_col2" class="data row16 col2" >0.0005</td>
      <td id="T_0be90_row16_col3" class="data row16 col3" >0.0224</td>
      <td id="T_0be90_row16_col4" class="data row16 col4" >-0.0005</td>
      <td id="T_0be90_row16_col5" class="data row16 col5" >0.0199</td>
      <td id="T_0be90_row16_col6" class="data row16 col6" >0.1720</td>
      <td id="T_0be90_row16_col7" class="data row16 col7" >0.0100</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row17" class="row_heading level0 row17" >llar</th>
      <td id="T_0be90_row17_col0" class="data row17 col0" >Lasso Least Angle Regression</td>
      <td id="T_0be90_row17_col1" class="data row17 col1" >0.0186</td>
      <td id="T_0be90_row17_col2" class="data row17 col2" >0.0005</td>
      <td id="T_0be90_row17_col3" class="data row17 col3" >0.0224</td>
      <td id="T_0be90_row17_col4" class="data row17 col4" >-0.0005</td>
      <td id="T_0be90_row17_col5" class="data row17 col5" >0.0199</td>
      <td id="T_0be90_row17_col6" class="data row17 col6" >0.1720</td>
      <td id="T_0be90_row17_col7" class="data row17 col7" >0.0070</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row18" class="row_heading level0 row18" >dummy</th>
      <td id="T_0be90_row18_col0" class="data row18 col0" >Dummy Regressor</td>
      <td id="T_0be90_row18_col1" class="data row18 col1" >0.0186</td>
      <td id="T_0be90_row18_col2" class="data row18 col2" >0.0005</td>
      <td id="T_0be90_row18_col3" class="data row18 col3" >0.0224</td>
      <td id="T_0be90_row18_col4" class="data row18 col4" >-0.0005</td>
      <td id="T_0be90_row18_col5" class="data row18 col5" >0.0199</td>
      <td id="T_0be90_row18_col6" class="data row18 col6" >0.1720</td>
      <td id="T_0be90_row18_col7" class="data row18 col7" >0.0050</td>
    </tr>
    <tr>
      <th id="T_0be90_level0_row19" class="row_heading level0 row19" >par</th>
      <td id="T_0be90_row19_col0" class="data row19 col0" >Passive Aggressive Regressor</td>
      <td id="T_0be90_row19_col1" class="data row19 col1" >0.0255</td>
      <td id="T_0be90_row19_col2" class="data row19 col2" >0.0010</td>
      <td id="T_0be90_row19_col3" class="data row19 col3" >0.0313</td>
      <td id="T_0be90_row19_col4" class="data row19 col4" >-1.0358</td>
      <td id="T_0be90_row19_col5" class="data row19 col5" >0.0282</td>
      <td id="T_0be90_row19_col6" class="data row19 col6" >0.2352</td>
      <td id="T_0be90_row19_col7" class="data row19 col7" >0.0070</td>
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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Ridge(random_state=2666)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Ridge<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html">?<span>Documentation for Ridge</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Ridge(random_state=2666)</pre></div> </div></div></div></div>




```python
from sklearn.linear_model import Ridge


model = Ridge(alpha=1.0, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```


```python
model.score(X_test, y_test)
```




    0.8824998393361988




```python

```
