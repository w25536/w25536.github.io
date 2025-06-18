---
title: "Predicting Exam Performance with Linear Regression students performance in exams"
date: 2024-01-11
last_modified_at: 2024-01-11
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Predicting Exam Performance with Linear Regression students performance in exams 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/spscientist/students-performance-in-exams/versions/1



```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ydata_profiling import ProfileReport


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


```


```python

```

    0.9495384615384616
    1.0
    1.0



```python
df = pd.read_csv(os.path.join(path, "StudentsPerformance.csv"))
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
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
    </tr>
    <tr>
      <th>995</th>
      <td>female</td>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>88</td>
      <td>99</td>
      <td>95</td>
    </tr>
    <tr>
      <th>996</th>
      <td>male</td>
      <td>group C</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>62</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>997</th>
      <td>female</td>
      <td>group C</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>59</td>
      <td>71</td>
      <td>65</td>
    </tr>
    <tr>
      <th>998</th>
      <td>female</td>
      <td>group D</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>68</td>
      <td>78</td>
      <td>77</td>
    </tr>
    <tr>
      <th>999</th>
      <td>female</td>
      <td>group D</td>
      <td>some college</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>77</td>
      <td>86</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 8 columns</p>
</div>




```python
encoder = LabelEncoder()

df["gender"] = encoder.fit_transform(df["gender"])
df["race/ethnicity"] = encoder.fit_transform(df["race/ethnicity"])
df["parental level of education"] = encoder.fit_transform(
    df["parental level of education"]
)
df["lunch"] = encoder.fit_transform(df["lunch"])
df["test preparation course"] = encoder.fit_transform(df["test preparation course"])
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
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
    </tr>
    <tr>
      <th>995</th>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>88</td>
      <td>99</td>
      <td>95</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>62</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>71</td>
      <td>65</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>68</td>
      <td>78</td>
      <td>77</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>77</td>
      <td>86</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 8 columns</p>
</div>




```python
# profile = ProfileReport(df)
# profile.to_notebook_iframe()
```


```python
corr = df.corr()
sns.heatmap(corr)
```




    <Axes: >




    
![png](011_Predicting_Exam_Performance_with_Linear_Regression_students_performance_in_exams_files/011_Predicting_Exam_Performance_with_Linear_Regression_students_performance_in_exams_8_1.png)
    



```python
df = df.drop(
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ],
    axis=1,
)
```


```python
scaler = StandardScaler()
scalded_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```


```python
X = scalded_df
```


```python
X_math = X[["reading score", "writing score"]]
X_reading = X[["math score", "writing score"]]
X_writing = X[["math score", "reading score"]]

y_math = X["math score"]
y_reading = X["reading score"]
y_writing = X["writing score"]
```


```python
X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(
    X_math, y_math, train_size=0.7, shuffle=True
)

X_reading_train, X_reading_test, y_reading_train, y_reading_test = train_test_split(
    X_reading, y_reading, train_size=0.7, shuffle=True
)

X_writing_train, X_writing_test, y_writing_train, y_writing_test = train_test_split(
    X_writing, y_writing, train_size=0.7, shuffle=True
)

math_model = LinearRegression()
math_model.fit(X_math_train, y_math_train)

reading_model = LinearRegression()
reading_model.fit(X_reading_train, y_reading_train)

writing_model = LinearRegression()
writing_model.fit(X_writing_train, y_writing_train)

# Evaluate models
math_score = math_model.score(X_math_test, y_math_test)
reading_score = reading_model.score(X_reading_test, y_reading_test)
writing_score = writing_model.score(X_writing_test, y_writing_test)

print(f"Math model R-squared: {math_score:.4f}")
print(f"Reading model R-squared: {reading_score:.4f}")
print(f"Writing model R-squared: {writing_score:.4f}")
```

    Math model R-squared: 0.6718
    Reading model R-squared: 0.9209
    Writing model R-squared: 0.9119



```python
from pycaret.regression import *


setup(X, target=X["math score"], train_size=0.7, session_id=42)
```


<style type="text/css">
#T_59578_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_59578">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_59578_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_59578_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_59578_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_59578_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_59578_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_59578_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_59578_row1_col1" class="data row1 col1" >math score_y</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_59578_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_59578_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_59578_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_59578_row3_col1" class="data row3 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_59578_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_59578_row4_col1" class="data row4 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_59578_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_59578_row5_col1" class="data row5 col1" >(700, 4)</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_59578_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_59578_row6_col1" class="data row6 col1" >(300, 4)</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_59578_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_59578_row7_col1" class="data row7 col1" >3</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_59578_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_59578_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_59578_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_59578_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_59578_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_59578_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_59578_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_59578_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_59578_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_59578_row12_col1" class="data row12 col1" >KFold</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_59578_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_59578_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_59578_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_59578_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_59578_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_59578_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_59578_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_59578_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_59578_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_59578_row17_col1" class="data row17 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_59578_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_59578_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_59578_row18_col1" class="data row18 col1" >f1d6</td>
    </tr>
  </tbody>
</table>






    <pycaret.regression.oop.RegressionExperiment at 0x32cce42b0>




```python
compare_models()
```






<style type="text/css">
#T_90b2c th {
  text-align: left;
}
#T_90b2c_row0_col0, #T_90b2c_row1_col0, #T_90b2c_row2_col0, #T_90b2c_row3_col0, #T_90b2c_row4_col0, #T_90b2c_row4_col1, #T_90b2c_row4_col3, #T_90b2c_row4_col5, #T_90b2c_row4_col6, #T_90b2c_row5_col0, #T_90b2c_row6_col0, #T_90b2c_row6_col1, #T_90b2c_row6_col2, #T_90b2c_row6_col3, #T_90b2c_row6_col4, #T_90b2c_row6_col5, #T_90b2c_row6_col6, #T_90b2c_row7_col0, #T_90b2c_row7_col1, #T_90b2c_row7_col2, #T_90b2c_row7_col3, #T_90b2c_row7_col4, #T_90b2c_row7_col5, #T_90b2c_row7_col6, #T_90b2c_row8_col0, #T_90b2c_row8_col1, #T_90b2c_row8_col2, #T_90b2c_row8_col3, #T_90b2c_row8_col4, #T_90b2c_row8_col5, #T_90b2c_row8_col6, #T_90b2c_row9_col0, #T_90b2c_row9_col1, #T_90b2c_row9_col2, #T_90b2c_row9_col3, #T_90b2c_row9_col4, #T_90b2c_row9_col5, #T_90b2c_row9_col6, #T_90b2c_row10_col0, #T_90b2c_row10_col1, #T_90b2c_row10_col2, #T_90b2c_row10_col3, #T_90b2c_row10_col4, #T_90b2c_row10_col5, #T_90b2c_row10_col6, #T_90b2c_row11_col0, #T_90b2c_row11_col1, #T_90b2c_row11_col2, #T_90b2c_row11_col3, #T_90b2c_row11_col4, #T_90b2c_row11_col5, #T_90b2c_row11_col6, #T_90b2c_row12_col0, #T_90b2c_row12_col1, #T_90b2c_row12_col2, #T_90b2c_row12_col3, #T_90b2c_row12_col4, #T_90b2c_row12_col5, #T_90b2c_row12_col6, #T_90b2c_row13_col0, #T_90b2c_row13_col1, #T_90b2c_row13_col2, #T_90b2c_row13_col3, #T_90b2c_row13_col4, #T_90b2c_row13_col5, #T_90b2c_row13_col6, #T_90b2c_row14_col0, #T_90b2c_row14_col1, #T_90b2c_row14_col2, #T_90b2c_row14_col3, #T_90b2c_row14_col4, #T_90b2c_row14_col5, #T_90b2c_row14_col6, #T_90b2c_row15_col0, #T_90b2c_row15_col1, #T_90b2c_row15_col2, #T_90b2c_row15_col3, #T_90b2c_row15_col4, #T_90b2c_row15_col5, #T_90b2c_row15_col6, #T_90b2c_row16_col0, #T_90b2c_row16_col1, #T_90b2c_row16_col2, #T_90b2c_row16_col3, #T_90b2c_row16_col4, #T_90b2c_row16_col5, #T_90b2c_row16_col6, #T_90b2c_row17_col0, #T_90b2c_row17_col1, #T_90b2c_row17_col2, #T_90b2c_row17_col3, #T_90b2c_row17_col4, #T_90b2c_row17_col5, #T_90b2c_row17_col6, #T_90b2c_row18_col0, #T_90b2c_row18_col1, #T_90b2c_row18_col2, #T_90b2c_row18_col3, #T_90b2c_row18_col4, #T_90b2c_row18_col5, #T_90b2c_row18_col6, #T_90b2c_row19_col0, #T_90b2c_row19_col1, #T_90b2c_row19_col2, #T_90b2c_row19_col3, #T_90b2c_row19_col4, #T_90b2c_row19_col5, #T_90b2c_row19_col6 {
  text-align: left;
}
#T_90b2c_row0_col1, #T_90b2c_row0_col2, #T_90b2c_row0_col3, #T_90b2c_row0_col4, #T_90b2c_row0_col5, #T_90b2c_row0_col6, #T_90b2c_row1_col1, #T_90b2c_row1_col2, #T_90b2c_row1_col3, #T_90b2c_row1_col4, #T_90b2c_row1_col5, #T_90b2c_row1_col6, #T_90b2c_row2_col1, #T_90b2c_row2_col2, #T_90b2c_row2_col3, #T_90b2c_row2_col4, #T_90b2c_row2_col5, #T_90b2c_row2_col6, #T_90b2c_row3_col1, #T_90b2c_row3_col2, #T_90b2c_row3_col3, #T_90b2c_row3_col4, #T_90b2c_row3_col5, #T_90b2c_row3_col6, #T_90b2c_row4_col2, #T_90b2c_row4_col4, #T_90b2c_row5_col1, #T_90b2c_row5_col2, #T_90b2c_row5_col3, #T_90b2c_row5_col4, #T_90b2c_row5_col5, #T_90b2c_row5_col6 {
  text-align: left;
  background-color: yellow;
}
#T_90b2c_row0_col7, #T_90b2c_row2_col7, #T_90b2c_row3_col7, #T_90b2c_row4_col7, #T_90b2c_row5_col7, #T_90b2c_row6_col7, #T_90b2c_row7_col7, #T_90b2c_row8_col7, #T_90b2c_row9_col7, #T_90b2c_row10_col7, #T_90b2c_row11_col7, #T_90b2c_row12_col7, #T_90b2c_row13_col7, #T_90b2c_row14_col7, #T_90b2c_row15_col7, #T_90b2c_row16_col7, #T_90b2c_row17_col7, #T_90b2c_row18_col7, #T_90b2c_row19_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_90b2c_row1_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_90b2c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_90b2c_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_90b2c_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_90b2c_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_90b2c_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_90b2c_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_90b2c_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_90b2c_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_90b2c_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_90b2c_level0_row0" class="row_heading level0 row0" >lr</th>
      <td id="T_90b2c_row0_col0" class="data row0 col0" >Linear Regression</td>
      <td id="T_90b2c_row0_col1" class="data row0 col1" >0.0000</td>
      <td id="T_90b2c_row0_col2" class="data row0 col2" >0.0000</td>
      <td id="T_90b2c_row0_col3" class="data row0 col3" >0.0000</td>
      <td id="T_90b2c_row0_col4" class="data row0 col4" >1.0000</td>
      <td id="T_90b2c_row0_col5" class="data row0 col5" >0.0000</td>
      <td id="T_90b2c_row0_col6" class="data row0 col6" >0.0000</td>
      <td id="T_90b2c_row0_col7" class="data row0 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row1" class="row_heading level0 row1" >omp</th>
      <td id="T_90b2c_row1_col0" class="data row1 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_90b2c_row1_col1" class="data row1 col1" >0.0000</td>
      <td id="T_90b2c_row1_col2" class="data row1 col2" >0.0000</td>
      <td id="T_90b2c_row1_col3" class="data row1 col3" >0.0000</td>
      <td id="T_90b2c_row1_col4" class="data row1 col4" >1.0000</td>
      <td id="T_90b2c_row1_col5" class="data row1 col5" >0.0000</td>
      <td id="T_90b2c_row1_col6" class="data row1 col6" >0.0000</td>
      <td id="T_90b2c_row1_col7" class="data row1 col7" >0.0030</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row2" class="row_heading level0 row2" >br</th>
      <td id="T_90b2c_row2_col0" class="data row2 col0" >Bayesian Ridge</td>
      <td id="T_90b2c_row2_col1" class="data row2 col1" >0.0000</td>
      <td id="T_90b2c_row2_col2" class="data row2 col2" >0.0000</td>
      <td id="T_90b2c_row2_col3" class="data row2 col3" >0.0000</td>
      <td id="T_90b2c_row2_col4" class="data row2 col4" >1.0000</td>
      <td id="T_90b2c_row2_col5" class="data row2 col5" >0.0000</td>
      <td id="T_90b2c_row2_col6" class="data row2 col6" >0.0000</td>
      <td id="T_90b2c_row2_col7" class="data row2 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row3" class="row_heading level0 row3" >huber</th>
      <td id="T_90b2c_row3_col0" class="data row3 col0" >Huber Regressor</td>
      <td id="T_90b2c_row3_col1" class="data row3 col1" >0.0000</td>
      <td id="T_90b2c_row3_col2" class="data row3 col2" >0.0000</td>
      <td id="T_90b2c_row3_col3" class="data row3 col3" >0.0000</td>
      <td id="T_90b2c_row3_col4" class="data row3 col4" >1.0000</td>
      <td id="T_90b2c_row3_col5" class="data row3 col5" >0.0000</td>
      <td id="T_90b2c_row3_col6" class="data row3 col6" >0.0000</td>
      <td id="T_90b2c_row3_col7" class="data row3 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row4" class="row_heading level0 row4" >ridge</th>
      <td id="T_90b2c_row4_col0" class="data row4 col0" >Ridge Regression</td>
      <td id="T_90b2c_row4_col1" class="data row4 col1" >0.0023</td>
      <td id="T_90b2c_row4_col2" class="data row4 col2" >0.0000</td>
      <td id="T_90b2c_row4_col3" class="data row4 col3" >0.0028</td>
      <td id="T_90b2c_row4_col4" class="data row4 col4" >1.0000</td>
      <td id="T_90b2c_row4_col5" class="data row4 col5" >0.0017</td>
      <td id="T_90b2c_row4_col6" class="data row4 col6" >0.0126</td>
      <td id="T_90b2c_row4_col7" class="data row4 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row5" class="row_heading level0 row5" >lar</th>
      <td id="T_90b2c_row5_col0" class="data row5 col0" >Least Angle Regression</td>
      <td id="T_90b2c_row5_col1" class="data row5 col1" >0.0000</td>
      <td id="T_90b2c_row5_col2" class="data row5 col2" >0.0000</td>
      <td id="T_90b2c_row5_col3" class="data row5 col3" >0.0000</td>
      <td id="T_90b2c_row5_col4" class="data row5 col4" >1.0000</td>
      <td id="T_90b2c_row5_col5" class="data row5 col5" >0.0000</td>
      <td id="T_90b2c_row5_col6" class="data row5 col6" >0.0000</td>
      <td id="T_90b2c_row5_col7" class="data row5 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row6" class="row_heading level0 row6" >et</th>
      <td id="T_90b2c_row6_col0" class="data row6 col0" >Extra Trees Regressor</td>
      <td id="T_90b2c_row6_col1" class="data row6 col1" >0.0043</td>
      <td id="T_90b2c_row6_col2" class="data row6 col2" >0.0008</td>
      <td id="T_90b2c_row6_col3" class="data row6 col3" >0.0185</td>
      <td id="T_90b2c_row6_col4" class="data row6 col4" >0.9992</td>
      <td id="T_90b2c_row6_col5" class="data row6 col5" >0.0057</td>
      <td id="T_90b2c_row6_col6" class="data row6 col6" >0.0076</td>
      <td id="T_90b2c_row6_col7" class="data row6 col7" >0.0140</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row7" class="row_heading level0 row7" >gbr</th>
      <td id="T_90b2c_row7_col0" class="data row7 col0" >Gradient Boosting Regressor</td>
      <td id="T_90b2c_row7_col1" class="data row7 col1" >0.0029</td>
      <td id="T_90b2c_row7_col2" class="data row7 col2" >0.0009</td>
      <td id="T_90b2c_row7_col3" class="data row7 col3" >0.0181</td>
      <td id="T_90b2c_row7_col4" class="data row7 col4" >0.9991</td>
      <td id="T_90b2c_row7_col5" class="data row7 col5" >0.0046</td>
      <td id="T_90b2c_row7_col6" class="data row7 col6" >0.0017</td>
      <td id="T_90b2c_row7_col7" class="data row7 col7" >0.0120</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row8" class="row_heading level0 row8" >catboost</th>
      <td id="T_90b2c_row8_col0" class="data row8 col0" >CatBoost Regressor</td>
      <td id="T_90b2c_row8_col1" class="data row8 col1" >0.0117</td>
      <td id="T_90b2c_row8_col2" class="data row8 col2" >0.0012</td>
      <td id="T_90b2c_row8_col3" class="data row8 col3" >0.0290</td>
      <td id="T_90b2c_row8_col4" class="data row8 col4" >0.9988</td>
      <td id="T_90b2c_row8_col5" class="data row8 col5" >0.0100</td>
      <td id="T_90b2c_row8_col6" class="data row8 col6" >0.0424</td>
      <td id="T_90b2c_row8_col7" class="data row8 col7" >0.0540</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row9" class="row_heading level0 row9" >rf</th>
      <td id="T_90b2c_row9_col0" class="data row9 col0" >Random Forest Regressor</td>
      <td id="T_90b2c_row9_col1" class="data row9 col1" >0.0045</td>
      <td id="T_90b2c_row9_col2" class="data row9 col2" >0.0012</td>
      <td id="T_90b2c_row9_col3" class="data row9 col3" >0.0242</td>
      <td id="T_90b2c_row9_col4" class="data row9 col4" >0.9988</td>
      <td id="T_90b2c_row9_col5" class="data row9 col5" >0.0063</td>
      <td id="T_90b2c_row9_col6" class="data row9 col6" >0.0019</td>
      <td id="T_90b2c_row9_col7" class="data row9 col7" >0.0200</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row10" class="row_heading level0 row10" >dt</th>
      <td id="T_90b2c_row10_col0" class="data row10 col0" >Decision Tree Regressor</td>
      <td id="T_90b2c_row10_col1" class="data row10 col1" >0.0032</td>
      <td id="T_90b2c_row10_col2" class="data row10 col2" >0.0013</td>
      <td id="T_90b2c_row10_col3" class="data row10 col3" >0.0231</td>
      <td id="T_90b2c_row10_col4" class="data row10 col4" >0.9987</td>
      <td id="T_90b2c_row10_col5" class="data row10 col5" >0.0060</td>
      <td id="T_90b2c_row10_col6" class="data row10 col6" >0.0011</td>
      <td id="T_90b2c_row10_col7" class="data row10 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row11" class="row_heading level0 row11" >par</th>
      <td id="T_90b2c_row11_col0" class="data row11 col0" >Passive Aggressive Regressor</td>
      <td id="T_90b2c_row11_col1" class="data row11 col1" >0.0284</td>
      <td id="T_90b2c_row11_col2" class="data row11 col2" >0.0013</td>
      <td id="T_90b2c_row11_col3" class="data row11 col3" >0.0350</td>
      <td id="T_90b2c_row11_col4" class="data row11 col4" >0.9987</td>
      <td id="T_90b2c_row11_col5" class="data row11 col5" >0.0220</td>
      <td id="T_90b2c_row11_col6" class="data row11 col6" >0.1992</td>
      <td id="T_90b2c_row11_col7" class="data row11 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row12" class="row_heading level0 row12" >xgboost</th>
      <td id="T_90b2c_row12_col0" class="data row12 col0" >Extreme Gradient Boosting</td>
      <td id="T_90b2c_row12_col1" class="data row12 col1" >0.0047</td>
      <td id="T_90b2c_row12_col2" class="data row12 col2" >0.0015</td>
      <td id="T_90b2c_row12_col3" class="data row12 col3" >0.0257</td>
      <td id="T_90b2c_row12_col4" class="data row12 col4" >0.9985</td>
      <td id="T_90b2c_row12_col5" class="data row12 col5" >0.0066</td>
      <td id="T_90b2c_row12_col6" class="data row12 col6" >0.0029</td>
      <td id="T_90b2c_row12_col7" class="data row12 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row13" class="row_heading level0 row13" >ada</th>
      <td id="T_90b2c_row13_col0" class="data row13 col0" >AdaBoost Regressor</td>
      <td id="T_90b2c_row13_col1" class="data row13 col1" >0.0616</td>
      <td id="T_90b2c_row13_col2" class="data row13 col2" >0.0076</td>
      <td id="T_90b2c_row13_col3" class="data row13 col3" >0.0852</td>
      <td id="T_90b2c_row13_col4" class="data row13 col4" >0.9920</td>
      <td id="T_90b2c_row13_col5" class="data row13 col5" >0.0425</td>
      <td id="T_90b2c_row13_col6" class="data row13 col6" >0.3141</td>
      <td id="T_90b2c_row13_col7" class="data row13 col7" >0.0080</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row14" class="row_heading level0 row14" >knn</th>
      <td id="T_90b2c_row14_col0" class="data row14 col0" >K Neighbors Regressor</td>
      <td id="T_90b2c_row14_col1" class="data row14 col1" >0.0576</td>
      <td id="T_90b2c_row14_col2" class="data row14 col2" >0.0079</td>
      <td id="T_90b2c_row14_col3" class="data row14 col3" >0.0863</td>
      <td id="T_90b2c_row14_col4" class="data row14 col4" >0.9918</td>
      <td id="T_90b2c_row14_col5" class="data row14 col5" >0.0457</td>
      <td id="T_90b2c_row14_col6" class="data row14 col6" >0.2638</td>
      <td id="T_90b2c_row14_col7" class="data row14 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row15" class="row_heading level0 row15" >lightgbm</th>
      <td id="T_90b2c_row15_col0" class="data row15 col0" >Light Gradient Boosting Machine</td>
      <td id="T_90b2c_row15_col1" class="data row15 col1" >0.0269</td>
      <td id="T_90b2c_row15_col2" class="data row15 col2" >0.0100</td>
      <td id="T_90b2c_row15_col3" class="data row15 col3" >0.0918</td>
      <td id="T_90b2c_row15_col4" class="data row15 col4" >0.9898</td>
      <td id="T_90b2c_row15_col5" class="data row15 col5" >0.0285</td>
      <td id="T_90b2c_row15_col6" class="data row15 col6" >0.0217</td>
      <td id="T_90b2c_row15_col7" class="data row15 col7" >0.3020</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row16" class="row_heading level0 row16" >en</th>
      <td id="T_90b2c_row16_col0" class="data row16 col0" >Elastic Net</td>
      <td id="T_90b2c_row16_col1" class="data row16 col1" >0.5261</td>
      <td id="T_90b2c_row16_col2" class="data row16 col2" >0.4398</td>
      <td id="T_90b2c_row16_col3" class="data row16 col3" >0.6630</td>
      <td id="T_90b2c_row16_col4" class="data row16 col4" >0.5362</td>
      <td id="T_90b2c_row16_col5" class="data row16 col5" >0.3545</td>
      <td id="T_90b2c_row16_col6" class="data row16 col6" >0.7814</td>
      <td id="T_90b2c_row16_col7" class="data row16 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row17" class="row_heading level0 row17" >lasso</th>
      <td id="T_90b2c_row17_col0" class="data row17 col0" >Lasso Regression</td>
      <td id="T_90b2c_row17_col1" class="data row17 col1" >0.7792</td>
      <td id="T_90b2c_row17_col2" class="data row17 col2" >0.9633</td>
      <td id="T_90b2c_row17_col3" class="data row17 col3" >0.9812</td>
      <td id="T_90b2c_row17_col4" class="data row17 col4" >-0.0162</td>
      <td id="T_90b2c_row17_col5" class="data row17 col5" >0.5833</td>
      <td id="T_90b2c_row17_col6" class="data row17 col6" >1.1510</td>
      <td id="T_90b2c_row17_col7" class="data row17 col7" >0.0050</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row18" class="row_heading level0 row18" >llar</th>
      <td id="T_90b2c_row18_col0" class="data row18 col0" >Lasso Least Angle Regression</td>
      <td id="T_90b2c_row18_col1" class="data row18 col1" >0.7792</td>
      <td id="T_90b2c_row18_col2" class="data row18 col2" >0.9633</td>
      <td id="T_90b2c_row18_col3" class="data row18 col3" >0.9812</td>
      <td id="T_90b2c_row18_col4" class="data row18 col4" >-0.0162</td>
      <td id="T_90b2c_row18_col5" class="data row18 col5" >0.5833</td>
      <td id="T_90b2c_row18_col6" class="data row18 col6" >1.1510</td>
      <td id="T_90b2c_row18_col7" class="data row18 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_90b2c_level0_row19" class="row_heading level0 row19" >dummy</th>
      <td id="T_90b2c_row19_col0" class="data row19 col0" >Dummy Regressor</td>
      <td id="T_90b2c_row19_col1" class="data row19 col1" >0.7792</td>
      <td id="T_90b2c_row19_col2" class="data row19 col2" >0.9633</td>
      <td id="T_90b2c_row19_col3" class="data row19 col3" >0.9812</td>
      <td id="T_90b2c_row19_col4" class="data row19 col4" >-0.0162</td>
      <td id="T_90b2c_row19_col5" class="data row19 col5" >0.5833</td>
      <td id="T_90b2c_row19_col6" class="data row19 col6" >1.1510</td>
      <td id="T_90b2c_row19_col7" class="data row19 col7" >0.0040</td>
    </tr>
  </tbody>
</table>










<style>#sk-container-id-4 {
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

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
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

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
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

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
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

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
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

#sk-container-id-4 a.estimator_doc_link {
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

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression(n_jobs=-1)</pre></div> </div></div></div></div>




```python
setup(X, target=X["reading score"], train_size=0.7, session_id=42)
```


<style type="text/css">
#T_1b235_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_1b235">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1b235_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_1b235_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1b235_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1b235_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_1b235_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1b235_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_1b235_row1_col1" class="data row1 col1" >reading score_y</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1b235_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_1b235_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_1b235_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_1b235_row3_col1" class="data row3 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_1b235_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_1b235_row4_col1" class="data row4 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_1b235_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_1b235_row5_col1" class="data row5 col1" >(700, 4)</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_1b235_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_1b235_row6_col1" class="data row6 col1" >(300, 4)</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_1b235_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_1b235_row7_col1" class="data row7 col1" >3</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_1b235_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_1b235_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_1b235_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_1b235_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_1b235_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_1b235_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_1b235_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_1b235_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_1b235_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_1b235_row12_col1" class="data row12 col1" >KFold</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_1b235_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_1b235_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_1b235_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_1b235_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_1b235_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_1b235_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_1b235_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_1b235_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_1b235_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_1b235_row17_col1" class="data row17 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_1b235_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_1b235_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_1b235_row18_col1" class="data row18 col1" >e262</td>
    </tr>
  </tbody>
</table>






    <pycaret.regression.oop.RegressionExperiment at 0x32e0146d0>






```python
compare_models()
```






<style type="text/css">
#T_6106c th {
  text-align: left;
}
#T_6106c_row0_col0, #T_6106c_row1_col0, #T_6106c_row1_col1, #T_6106c_row1_col3, #T_6106c_row1_col5, #T_6106c_row1_col6, #T_6106c_row2_col0, #T_6106c_row3_col0, #T_6106c_row4_col0, #T_6106c_row5_col0, #T_6106c_row6_col0, #T_6106c_row6_col1, #T_6106c_row6_col2, #T_6106c_row6_col3, #T_6106c_row6_col4, #T_6106c_row6_col5, #T_6106c_row6_col6, #T_6106c_row7_col0, #T_6106c_row7_col1, #T_6106c_row7_col2, #T_6106c_row7_col3, #T_6106c_row7_col4, #T_6106c_row7_col5, #T_6106c_row7_col6, #T_6106c_row8_col0, #T_6106c_row8_col1, #T_6106c_row8_col2, #T_6106c_row8_col3, #T_6106c_row8_col4, #T_6106c_row8_col5, #T_6106c_row8_col6, #T_6106c_row9_col0, #T_6106c_row9_col1, #T_6106c_row9_col2, #T_6106c_row9_col3, #T_6106c_row9_col4, #T_6106c_row9_col5, #T_6106c_row9_col6, #T_6106c_row10_col0, #T_6106c_row10_col1, #T_6106c_row10_col2, #T_6106c_row10_col3, #T_6106c_row10_col4, #T_6106c_row10_col5, #T_6106c_row10_col6, #T_6106c_row11_col0, #T_6106c_row11_col1, #T_6106c_row11_col2, #T_6106c_row11_col3, #T_6106c_row11_col4, #T_6106c_row11_col5, #T_6106c_row11_col6, #T_6106c_row12_col0, #T_6106c_row12_col1, #T_6106c_row12_col2, #T_6106c_row12_col3, #T_6106c_row12_col4, #T_6106c_row12_col5, #T_6106c_row12_col6, #T_6106c_row13_col0, #T_6106c_row13_col1, #T_6106c_row13_col2, #T_6106c_row13_col3, #T_6106c_row13_col4, #T_6106c_row13_col5, #T_6106c_row13_col6, #T_6106c_row14_col0, #T_6106c_row14_col1, #T_6106c_row14_col2, #T_6106c_row14_col3, #T_6106c_row14_col4, #T_6106c_row14_col5, #T_6106c_row14_col6, #T_6106c_row15_col0, #T_6106c_row15_col1, #T_6106c_row15_col2, #T_6106c_row15_col3, #T_6106c_row15_col4, #T_6106c_row15_col5, #T_6106c_row15_col6, #T_6106c_row16_col0, #T_6106c_row16_col1, #T_6106c_row16_col2, #T_6106c_row16_col3, #T_6106c_row16_col4, #T_6106c_row16_col5, #T_6106c_row16_col6, #T_6106c_row17_col0, #T_6106c_row17_col1, #T_6106c_row17_col2, #T_6106c_row17_col3, #T_6106c_row17_col4, #T_6106c_row17_col5, #T_6106c_row17_col6, #T_6106c_row18_col0, #T_6106c_row18_col1, #T_6106c_row18_col2, #T_6106c_row18_col3, #T_6106c_row18_col4, #T_6106c_row18_col5, #T_6106c_row18_col6, #T_6106c_row19_col0, #T_6106c_row19_col1, #T_6106c_row19_col2, #T_6106c_row19_col3, #T_6106c_row19_col4, #T_6106c_row19_col5, #T_6106c_row19_col6 {
  text-align: left;
}
#T_6106c_row0_col1, #T_6106c_row0_col2, #T_6106c_row0_col3, #T_6106c_row0_col4, #T_6106c_row0_col5, #T_6106c_row0_col6, #T_6106c_row1_col2, #T_6106c_row1_col4, #T_6106c_row2_col1, #T_6106c_row2_col2, #T_6106c_row2_col3, #T_6106c_row2_col4, #T_6106c_row2_col5, #T_6106c_row2_col6, #T_6106c_row3_col1, #T_6106c_row3_col2, #T_6106c_row3_col3, #T_6106c_row3_col4, #T_6106c_row3_col5, #T_6106c_row3_col6, #T_6106c_row4_col1, #T_6106c_row4_col2, #T_6106c_row4_col3, #T_6106c_row4_col4, #T_6106c_row4_col5, #T_6106c_row4_col6, #T_6106c_row5_col1, #T_6106c_row5_col2, #T_6106c_row5_col3, #T_6106c_row5_col4, #T_6106c_row5_col5, #T_6106c_row5_col6 {
  text-align: left;
  background-color: yellow;
}
#T_6106c_row0_col7, #T_6106c_row1_col7, #T_6106c_row2_col7, #T_6106c_row3_col7, #T_6106c_row4_col7, #T_6106c_row5_col7, #T_6106c_row6_col7, #T_6106c_row7_col7, #T_6106c_row8_col7, #T_6106c_row9_col7, #T_6106c_row10_col7, #T_6106c_row11_col7, #T_6106c_row12_col7, #T_6106c_row13_col7, #T_6106c_row14_col7, #T_6106c_row15_col7, #T_6106c_row16_col7, #T_6106c_row17_col7, #T_6106c_row19_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_6106c_row18_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_6106c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6106c_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_6106c_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_6106c_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_6106c_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_6106c_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_6106c_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_6106c_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_6106c_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6106c_level0_row0" class="row_heading level0 row0" >lr</th>
      <td id="T_6106c_row0_col0" class="data row0 col0" >Linear Regression</td>
      <td id="T_6106c_row0_col1" class="data row0 col1" >0.0000</td>
      <td id="T_6106c_row0_col2" class="data row0 col2" >0.0000</td>
      <td id="T_6106c_row0_col3" class="data row0 col3" >0.0000</td>
      <td id="T_6106c_row0_col4" class="data row0 col4" >1.0000</td>
      <td id="T_6106c_row0_col5" class="data row0 col5" >0.0000</td>
      <td id="T_6106c_row0_col6" class="data row0 col6" >0.0000</td>
      <td id="T_6106c_row0_col7" class="data row0 col7" >0.0090</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row1" class="row_heading level0 row1" >ridge</th>
      <td id="T_6106c_row1_col0" class="data row1 col0" >Ridge Regression</td>
      <td id="T_6106c_row1_col1" class="data row1 col1" >0.0044</td>
      <td id="T_6106c_row1_col2" class="data row1 col2" >0.0000</td>
      <td id="T_6106c_row1_col3" class="data row1 col3" >0.0055</td>
      <td id="T_6106c_row1_col4" class="data row1 col4" >1.0000</td>
      <td id="T_6106c_row1_col5" class="data row1 col5" >0.0035</td>
      <td id="T_6106c_row1_col6" class="data row1 col6" >0.0164</td>
      <td id="T_6106c_row1_col7" class="data row1 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row2" class="row_heading level0 row2" >lar</th>
      <td id="T_6106c_row2_col0" class="data row2 col0" >Least Angle Regression</td>
      <td id="T_6106c_row2_col1" class="data row2 col1" >0.0000</td>
      <td id="T_6106c_row2_col2" class="data row2 col2" >0.0000</td>
      <td id="T_6106c_row2_col3" class="data row2 col3" >0.0000</td>
      <td id="T_6106c_row2_col4" class="data row2 col4" >1.0000</td>
      <td id="T_6106c_row2_col5" class="data row2 col5" >0.0000</td>
      <td id="T_6106c_row2_col6" class="data row2 col6" >0.0000</td>
      <td id="T_6106c_row2_col7" class="data row2 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row3" class="row_heading level0 row3" >omp</th>
      <td id="T_6106c_row3_col0" class="data row3 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_6106c_row3_col1" class="data row3 col1" >0.0000</td>
      <td id="T_6106c_row3_col2" class="data row3 col2" >0.0000</td>
      <td id="T_6106c_row3_col3" class="data row3 col3" >0.0000</td>
      <td id="T_6106c_row3_col4" class="data row3 col4" >1.0000</td>
      <td id="T_6106c_row3_col5" class="data row3 col5" >0.0000</td>
      <td id="T_6106c_row3_col6" class="data row3 col6" >0.0000</td>
      <td id="T_6106c_row3_col7" class="data row3 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row4" class="row_heading level0 row4" >br</th>
      <td id="T_6106c_row4_col0" class="data row4 col0" >Bayesian Ridge</td>
      <td id="T_6106c_row4_col1" class="data row4 col1" >0.0000</td>
      <td id="T_6106c_row4_col2" class="data row4 col2" >0.0000</td>
      <td id="T_6106c_row4_col3" class="data row4 col3" >0.0000</td>
      <td id="T_6106c_row4_col4" class="data row4 col4" >1.0000</td>
      <td id="T_6106c_row4_col5" class="data row4 col5" >0.0000</td>
      <td id="T_6106c_row4_col6" class="data row4 col6" >0.0000</td>
      <td id="T_6106c_row4_col7" class="data row4 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row5" class="row_heading level0 row5" >huber</th>
      <td id="T_6106c_row5_col0" class="data row5 col0" >Huber Regressor</td>
      <td id="T_6106c_row5_col1" class="data row5 col1" >0.0000</td>
      <td id="T_6106c_row5_col2" class="data row5 col2" >0.0000</td>
      <td id="T_6106c_row5_col3" class="data row5 col3" >0.0000</td>
      <td id="T_6106c_row5_col4" class="data row5 col4" >1.0000</td>
      <td id="T_6106c_row5_col5" class="data row5 col5" >0.0000</td>
      <td id="T_6106c_row5_col6" class="data row5 col6" >0.0000</td>
      <td id="T_6106c_row5_col7" class="data row5 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row6" class="row_heading level0 row6" >et</th>
      <td id="T_6106c_row6_col0" class="data row6 col0" >Extra Trees Regressor</td>
      <td id="T_6106c_row6_col1" class="data row6 col1" >0.0039</td>
      <td id="T_6106c_row6_col2" class="data row6 col2" >0.0005</td>
      <td id="T_6106c_row6_col3" class="data row6 col3" >0.0142</td>
      <td id="T_6106c_row6_col4" class="data row6 col4" >0.9995</td>
      <td id="T_6106c_row6_col5" class="data row6 col5" >0.0053</td>
      <td id="T_6106c_row6_col6" class="data row6 col6" >0.0067</td>
      <td id="T_6106c_row6_col7" class="data row6 col7" >0.0150</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row7" class="row_heading level0 row7" >gbr</th>
      <td id="T_6106c_row7_col0" class="data row7 col0" >Gradient Boosting Regressor</td>
      <td id="T_6106c_row7_col1" class="data row7 col1" >0.0019</td>
      <td id="T_6106c_row7_col2" class="data row7 col2" >0.0005</td>
      <td id="T_6106c_row7_col3" class="data row7 col3" >0.0118</td>
      <td id="T_6106c_row7_col4" class="data row7 col4" >0.9995</td>
      <td id="T_6106c_row7_col5" class="data row7 col5" >0.0035</td>
      <td id="T_6106c_row7_col6" class="data row7 col6" >0.0037</td>
      <td id="T_6106c_row7_col7" class="data row7 col7" >0.0090</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row8" class="row_heading level0 row8" >dt</th>
      <td id="T_6106c_row8_col0" class="data row8 col0" >Decision Tree Regressor</td>
      <td id="T_6106c_row8_col1" class="data row8 col1" >0.0023</td>
      <td id="T_6106c_row8_col2" class="data row8 col2" >0.0006</td>
      <td id="T_6106c_row8_col3" class="data row8 col3" >0.0161</td>
      <td id="T_6106c_row8_col4" class="data row8 col4" >0.9994</td>
      <td id="T_6106c_row8_col5" class="data row8 col5" >0.0049</td>
      <td id="T_6106c_row8_col6" class="data row8 col6" >0.0010</td>
      <td id="T_6106c_row8_col7" class="data row8 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row9" class="row_heading level0 row9" >rf</th>
      <td id="T_6106c_row9_col0" class="data row9 col0" >Random Forest Regressor</td>
      <td id="T_6106c_row9_col1" class="data row9 col1" >0.0033</td>
      <td id="T_6106c_row9_col2" class="data row9 col2" >0.0007</td>
      <td id="T_6106c_row9_col3" class="data row9 col3" >0.0165</td>
      <td id="T_6106c_row9_col4" class="data row9 col4" >0.9993</td>
      <td id="T_6106c_row9_col5" class="data row9 col5" >0.0050</td>
      <td id="T_6106c_row9_col6" class="data row9 col6" >0.0016</td>
      <td id="T_6106c_row9_col7" class="data row9 col7" >0.0180</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row10" class="row_heading level0 row10" >xgboost</th>
      <td id="T_6106c_row10_col0" class="data row10 col0" >Extreme Gradient Boosting</td>
      <td id="T_6106c_row10_col1" class="data row10 col1" >0.0037</td>
      <td id="T_6106c_row10_col2" class="data row10 col2" >0.0010</td>
      <td id="T_6106c_row10_col3" class="data row10 col3" >0.0205</td>
      <td id="T_6106c_row10_col4" class="data row10 col4" >0.9989</td>
      <td id="T_6106c_row10_col5" class="data row10 col5" >0.0060</td>
      <td id="T_6106c_row10_col6" class="data row10 col6" >0.0022</td>
      <td id="T_6106c_row10_col7" class="data row10 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row11" class="row_heading level0 row11" >par</th>
      <td id="T_6106c_row11_col0" class="data row11 col0" >Passive Aggressive Regressor</td>
      <td id="T_6106c_row11_col1" class="data row11 col1" >0.0281</td>
      <td id="T_6106c_row11_col2" class="data row11 col2" >0.0012</td>
      <td id="T_6106c_row11_col3" class="data row11 col3" >0.0346</td>
      <td id="T_6106c_row11_col4" class="data row11 col4" >0.9987</td>
      <td id="T_6106c_row11_col5" class="data row11 col5" >0.0217</td>
      <td id="T_6106c_row11_col6" class="data row11 col6" >0.0988</td>
      <td id="T_6106c_row11_col7" class="data row11 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row12" class="row_heading level0 row12" >catboost</th>
      <td id="T_6106c_row12_col0" class="data row12 col0" >CatBoost Regressor</td>
      <td id="T_6106c_row12_col1" class="data row12 col1" >0.0155</td>
      <td id="T_6106c_row12_col2" class="data row12 col2" >0.0013</td>
      <td id="T_6106c_row12_col3" class="data row12 col3" >0.0334</td>
      <td id="T_6106c_row12_col4" class="data row12 col4" >0.9986</td>
      <td id="T_6106c_row12_col5" class="data row12 col5" >0.0133</td>
      <td id="T_6106c_row12_col6" class="data row12 col6" >0.0388</td>
      <td id="T_6106c_row12_col7" class="data row12 col7" >0.0430</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row13" class="row_heading level0 row13" >lightgbm</th>
      <td id="T_6106c_row13_col0" class="data row13 col0" >Light Gradient Boosting Machine</td>
      <td id="T_6106c_row13_col1" class="data row13 col1" >0.0176</td>
      <td id="T_6106c_row13_col2" class="data row13 col2" >0.0043</td>
      <td id="T_6106c_row13_col3" class="data row13 col3" >0.0592</td>
      <td id="T_6106c_row13_col4" class="data row13 col4" >0.9955</td>
      <td id="T_6106c_row13_col5" class="data row13 col5" >0.0194</td>
      <td id="T_6106c_row13_col6" class="data row13 col6" >0.0127</td>
      <td id="T_6106c_row13_col7" class="data row13 col7" >0.2880</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row14" class="row_heading level0 row14" >knn</th>
      <td id="T_6106c_row14_col0" class="data row14 col0" >K Neighbors Regressor</td>
      <td id="T_6106c_row14_col1" class="data row14 col1" >0.0654</td>
      <td id="T_6106c_row14_col2" class="data row14 col2" >0.0090</td>
      <td id="T_6106c_row14_col3" class="data row14 col3" >0.0924</td>
      <td id="T_6106c_row14_col4" class="data row14 col4" >0.9906</td>
      <td id="T_6106c_row14_col5" class="data row14 col5" >0.0504</td>
      <td id="T_6106c_row14_col6" class="data row14 col6" >0.2247</td>
      <td id="T_6106c_row14_col7" class="data row14 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row15" class="row_heading level0 row15" >ada</th>
      <td id="T_6106c_row15_col0" class="data row15 col0" >AdaBoost Regressor</td>
      <td id="T_6106c_row15_col1" class="data row15 col1" >0.0775</td>
      <td id="T_6106c_row15_col2" class="data row15 col2" >0.0101</td>
      <td id="T_6106c_row15_col3" class="data row15 col3" >0.0993</td>
      <td id="T_6106c_row15_col4" class="data row15 col4" >0.9890</td>
      <td id="T_6106c_row15_col5" class="data row15 col5" >0.0522</td>
      <td id="T_6106c_row15_col6" class="data row15 col6" >0.2442</td>
      <td id="T_6106c_row15_col7" class="data row15 col7" >0.0080</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row16" class="row_heading level0 row16" >en</th>
      <td id="T_6106c_row16_col0" class="data row16 col0" >Elastic Net</td>
      <td id="T_6106c_row16_col1" class="data row16 col1" >0.5105</td>
      <td id="T_6106c_row16_col2" class="data row16 col2" >0.3991</td>
      <td id="T_6106c_row16_col3" class="data row16 col3" >0.6305</td>
      <td id="T_6106c_row16_col4" class="data row16 col4" >0.5712</td>
      <td id="T_6106c_row16_col5" class="data row16 col5" >0.3401</td>
      <td id="T_6106c_row16_col6" class="data row16 col6" >0.7096</td>
      <td id="T_6106c_row16_col7" class="data row16 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row17" class="row_heading level0 row17" >lasso</th>
      <td id="T_6106c_row17_col0" class="data row17 col0" >Lasso Regression</td>
      <td id="T_6106c_row17_col1" class="data row17 col1" >0.7860</td>
      <td id="T_6106c_row17_col2" class="data row17 col2" >0.9445</td>
      <td id="T_6106c_row17_col3" class="data row17 col3" >0.9706</td>
      <td id="T_6106c_row17_col4" class="data row17 col4" >-0.0172</td>
      <td id="T_6106c_row17_col5" class="data row17 col5" >0.5781</td>
      <td id="T_6106c_row17_col6" class="data row17 col6" >1.0561</td>
      <td id="T_6106c_row17_col7" class="data row17 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row18" class="row_heading level0 row18" >llar</th>
      <td id="T_6106c_row18_col0" class="data row18 col0" >Lasso Least Angle Regression</td>
      <td id="T_6106c_row18_col1" class="data row18 col1" >0.7860</td>
      <td id="T_6106c_row18_col2" class="data row18 col2" >0.9445</td>
      <td id="T_6106c_row18_col3" class="data row18 col3" >0.9706</td>
      <td id="T_6106c_row18_col4" class="data row18 col4" >-0.0172</td>
      <td id="T_6106c_row18_col5" class="data row18 col5" >0.5781</td>
      <td id="T_6106c_row18_col6" class="data row18 col6" >1.0561</td>
      <td id="T_6106c_row18_col7" class="data row18 col7" >0.0030</td>
    </tr>
    <tr>
      <th id="T_6106c_level0_row19" class="row_heading level0 row19" >dummy</th>
      <td id="T_6106c_row19_col0" class="data row19 col0" >Dummy Regressor</td>
      <td id="T_6106c_row19_col1" class="data row19 col1" >0.7860</td>
      <td id="T_6106c_row19_col2" class="data row19 col2" >0.9445</td>
      <td id="T_6106c_row19_col3" class="data row19 col3" >0.9706</td>
      <td id="T_6106c_row19_col4" class="data row19 col4" >-0.0172</td>
      <td id="T_6106c_row19_col5" class="data row19 col5" >0.5781</td>
      <td id="T_6106c_row19_col6" class="data row19 col6" >1.0561</td>
      <td id="T_6106c_row19_col7" class="data row19 col7" >0.0050</td>
    </tr>
  </tbody>
</table>










<style>#sk-container-id-5 {
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

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
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

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
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

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
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

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
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

#sk-container-id-5 a.estimator_doc_link {
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

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression(n_jobs=-1)</pre></div> </div></div></div></div>




```python
setup(X, target=X["writing score"], train_size=0.7, session_id=42)
```


<style type="text/css">
#T_1bae0_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_1bae0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1bae0_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_1bae0_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1bae0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1bae0_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_1bae0_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1bae0_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_1bae0_row1_col1" class="data row1 col1" >writing score_y</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1bae0_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_1bae0_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_1bae0_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_1bae0_row3_col1" class="data row3 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_1bae0_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_1bae0_row4_col1" class="data row4 col1" >(1000, 4)</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_1bae0_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_1bae0_row5_col1" class="data row5 col1" >(700, 4)</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_1bae0_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_1bae0_row6_col1" class="data row6 col1" >(300, 4)</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_1bae0_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_1bae0_row7_col1" class="data row7 col1" >3</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_1bae0_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_1bae0_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_1bae0_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_1bae0_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_1bae0_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_1bae0_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_1bae0_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_1bae0_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_1bae0_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_1bae0_row12_col1" class="data row12 col1" >KFold</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_1bae0_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_1bae0_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_1bae0_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_1bae0_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_1bae0_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_1bae0_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_1bae0_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_1bae0_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_1bae0_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_1bae0_row17_col1" class="data row17 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_1bae0_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_1bae0_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_1bae0_row18_col1" class="data row18 col1" >0268</td>
    </tr>
  </tbody>
</table>






    <pycaret.regression.oop.RegressionExperiment at 0x32eeaa7c0>




```python
compare_models()
```






<style type="text/css">
#T_1e6b1 th {
  text-align: left;
}
#T_1e6b1_row0_col0, #T_1e6b1_row1_col0, #T_1e6b1_row1_col1, #T_1e6b1_row1_col3, #T_1e6b1_row1_col5, #T_1e6b1_row1_col6, #T_1e6b1_row2_col0, #T_1e6b1_row3_col0, #T_1e6b1_row4_col0, #T_1e6b1_row5_col0, #T_1e6b1_row6_col0, #T_1e6b1_row6_col1, #T_1e6b1_row6_col2, #T_1e6b1_row6_col3, #T_1e6b1_row6_col4, #T_1e6b1_row6_col5, #T_1e6b1_row6_col6, #T_1e6b1_row7_col0, #T_1e6b1_row7_col1, #T_1e6b1_row7_col2, #T_1e6b1_row7_col3, #T_1e6b1_row7_col4, #T_1e6b1_row7_col5, #T_1e6b1_row7_col6, #T_1e6b1_row8_col0, #T_1e6b1_row8_col1, #T_1e6b1_row8_col2, #T_1e6b1_row8_col3, #T_1e6b1_row8_col4, #T_1e6b1_row8_col5, #T_1e6b1_row8_col6, #T_1e6b1_row9_col0, #T_1e6b1_row9_col1, #T_1e6b1_row9_col2, #T_1e6b1_row9_col3, #T_1e6b1_row9_col4, #T_1e6b1_row9_col5, #T_1e6b1_row9_col6, #T_1e6b1_row10_col0, #T_1e6b1_row10_col1, #T_1e6b1_row10_col2, #T_1e6b1_row10_col3, #T_1e6b1_row10_col4, #T_1e6b1_row10_col5, #T_1e6b1_row10_col6, #T_1e6b1_row11_col0, #T_1e6b1_row11_col1, #T_1e6b1_row11_col2, #T_1e6b1_row11_col3, #T_1e6b1_row11_col4, #T_1e6b1_row11_col5, #T_1e6b1_row11_col6, #T_1e6b1_row12_col0, #T_1e6b1_row12_col1, #T_1e6b1_row12_col2, #T_1e6b1_row12_col3, #T_1e6b1_row12_col4, #T_1e6b1_row12_col5, #T_1e6b1_row12_col6, #T_1e6b1_row13_col0, #T_1e6b1_row13_col1, #T_1e6b1_row13_col2, #T_1e6b1_row13_col3, #T_1e6b1_row13_col4, #T_1e6b1_row13_col5, #T_1e6b1_row13_col6, #T_1e6b1_row14_col0, #T_1e6b1_row14_col1, #T_1e6b1_row14_col2, #T_1e6b1_row14_col3, #T_1e6b1_row14_col4, #T_1e6b1_row14_col5, #T_1e6b1_row14_col6, #T_1e6b1_row15_col0, #T_1e6b1_row15_col1, #T_1e6b1_row15_col2, #T_1e6b1_row15_col3, #T_1e6b1_row15_col4, #T_1e6b1_row15_col5, #T_1e6b1_row15_col6, #T_1e6b1_row16_col0, #T_1e6b1_row16_col1, #T_1e6b1_row16_col2, #T_1e6b1_row16_col3, #T_1e6b1_row16_col4, #T_1e6b1_row16_col5, #T_1e6b1_row16_col6, #T_1e6b1_row17_col0, #T_1e6b1_row17_col1, #T_1e6b1_row17_col2, #T_1e6b1_row17_col3, #T_1e6b1_row17_col4, #T_1e6b1_row17_col5, #T_1e6b1_row17_col6, #T_1e6b1_row18_col0, #T_1e6b1_row18_col1, #T_1e6b1_row18_col2, #T_1e6b1_row18_col3, #T_1e6b1_row18_col4, #T_1e6b1_row18_col5, #T_1e6b1_row18_col6, #T_1e6b1_row19_col0, #T_1e6b1_row19_col1, #T_1e6b1_row19_col2, #T_1e6b1_row19_col3, #T_1e6b1_row19_col4, #T_1e6b1_row19_col5, #T_1e6b1_row19_col6 {
  text-align: left;
}
#T_1e6b1_row0_col1, #T_1e6b1_row0_col2, #T_1e6b1_row0_col3, #T_1e6b1_row0_col4, #T_1e6b1_row0_col5, #T_1e6b1_row0_col6, #T_1e6b1_row1_col2, #T_1e6b1_row1_col4, #T_1e6b1_row2_col1, #T_1e6b1_row2_col2, #T_1e6b1_row2_col3, #T_1e6b1_row2_col4, #T_1e6b1_row2_col5, #T_1e6b1_row2_col6, #T_1e6b1_row3_col1, #T_1e6b1_row3_col2, #T_1e6b1_row3_col3, #T_1e6b1_row3_col4, #T_1e6b1_row3_col5, #T_1e6b1_row3_col6, #T_1e6b1_row4_col1, #T_1e6b1_row4_col2, #T_1e6b1_row4_col3, #T_1e6b1_row4_col4, #T_1e6b1_row4_col5, #T_1e6b1_row4_col6, #T_1e6b1_row5_col1, #T_1e6b1_row5_col2, #T_1e6b1_row5_col3, #T_1e6b1_row5_col4, #T_1e6b1_row5_col5, #T_1e6b1_row5_col6 {
  text-align: left;
  background-color: yellow;
}
#T_1e6b1_row0_col7, #T_1e6b1_row3_col7, #T_1e6b1_row4_col7, #T_1e6b1_row5_col7, #T_1e6b1_row6_col7, #T_1e6b1_row7_col7, #T_1e6b1_row8_col7, #T_1e6b1_row9_col7, #T_1e6b1_row10_col7, #T_1e6b1_row11_col7, #T_1e6b1_row12_col7, #T_1e6b1_row13_col7, #T_1e6b1_row14_col7, #T_1e6b1_row15_col7, #T_1e6b1_row16_col7, #T_1e6b1_row17_col7, #T_1e6b1_row18_col7, #T_1e6b1_row19_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_1e6b1_row1_col7, #T_1e6b1_row2_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_1e6b1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1e6b1_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_1e6b1_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_1e6b1_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_1e6b1_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_1e6b1_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_1e6b1_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_1e6b1_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_1e6b1_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1e6b1_level0_row0" class="row_heading level0 row0" >lr</th>
      <td id="T_1e6b1_row0_col0" class="data row0 col0" >Linear Regression</td>
      <td id="T_1e6b1_row0_col1" class="data row0 col1" >0.0000</td>
      <td id="T_1e6b1_row0_col2" class="data row0 col2" >0.0000</td>
      <td id="T_1e6b1_row0_col3" class="data row0 col3" >0.0000</td>
      <td id="T_1e6b1_row0_col4" class="data row0 col4" >1.0000</td>
      <td id="T_1e6b1_row0_col5" class="data row0 col5" >0.0000</td>
      <td id="T_1e6b1_row0_col6" class="data row0 col6" >0.0000</td>
      <td id="T_1e6b1_row0_col7" class="data row0 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row1" class="row_heading level0 row1" >ridge</th>
      <td id="T_1e6b1_row1_col0" class="data row1 col0" >Ridge Regression</td>
      <td id="T_1e6b1_row1_col1" class="data row1 col1" >0.0043</td>
      <td id="T_1e6b1_row1_col2" class="data row1 col2" >0.0000</td>
      <td id="T_1e6b1_row1_col3" class="data row1 col3" >0.0053</td>
      <td id="T_1e6b1_row1_col4" class="data row1 col4" >1.0000</td>
      <td id="T_1e6b1_row1_col5" class="data row1 col5" >0.0034</td>
      <td id="T_1e6b1_row1_col6" class="data row1 col6" >0.0461</td>
      <td id="T_1e6b1_row1_col7" class="data row1 col7" >0.0030</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row2" class="row_heading level0 row2" >lar</th>
      <td id="T_1e6b1_row2_col0" class="data row2 col0" >Least Angle Regression</td>
      <td id="T_1e6b1_row2_col1" class="data row2 col1" >0.0000</td>
      <td id="T_1e6b1_row2_col2" class="data row2 col2" >0.0000</td>
      <td id="T_1e6b1_row2_col3" class="data row2 col3" >0.0000</td>
      <td id="T_1e6b1_row2_col4" class="data row2 col4" >1.0000</td>
      <td id="T_1e6b1_row2_col5" class="data row2 col5" >0.0000</td>
      <td id="T_1e6b1_row2_col6" class="data row2 col6" >0.0000</td>
      <td id="T_1e6b1_row2_col7" class="data row2 col7" >0.0030</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row3" class="row_heading level0 row3" >omp</th>
      <td id="T_1e6b1_row3_col0" class="data row3 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_1e6b1_row3_col1" class="data row3 col1" >0.0000</td>
      <td id="T_1e6b1_row3_col2" class="data row3 col2" >0.0000</td>
      <td id="T_1e6b1_row3_col3" class="data row3 col3" >0.0000</td>
      <td id="T_1e6b1_row3_col4" class="data row3 col4" >1.0000</td>
      <td id="T_1e6b1_row3_col5" class="data row3 col5" >0.0000</td>
      <td id="T_1e6b1_row3_col6" class="data row3 col6" >0.0000</td>
      <td id="T_1e6b1_row3_col7" class="data row3 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row4" class="row_heading level0 row4" >br</th>
      <td id="T_1e6b1_row4_col0" class="data row4 col0" >Bayesian Ridge</td>
      <td id="T_1e6b1_row4_col1" class="data row4 col1" >0.0000</td>
      <td id="T_1e6b1_row4_col2" class="data row4 col2" >0.0000</td>
      <td id="T_1e6b1_row4_col3" class="data row4 col3" >0.0000</td>
      <td id="T_1e6b1_row4_col4" class="data row4 col4" >1.0000</td>
      <td id="T_1e6b1_row4_col5" class="data row4 col5" >0.0000</td>
      <td id="T_1e6b1_row4_col6" class="data row4 col6" >0.0000</td>
      <td id="T_1e6b1_row4_col7" class="data row4 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row5" class="row_heading level0 row5" >huber</th>
      <td id="T_1e6b1_row5_col0" class="data row5 col0" >Huber Regressor</td>
      <td id="T_1e6b1_row5_col1" class="data row5 col1" >0.0000</td>
      <td id="T_1e6b1_row5_col2" class="data row5 col2" >0.0000</td>
      <td id="T_1e6b1_row5_col3" class="data row5 col3" >0.0000</td>
      <td id="T_1e6b1_row5_col4" class="data row5 col4" >1.0000</td>
      <td id="T_1e6b1_row5_col5" class="data row5 col5" >0.0000</td>
      <td id="T_1e6b1_row5_col6" class="data row5 col6" >0.0000</td>
      <td id="T_1e6b1_row5_col7" class="data row5 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row6" class="row_heading level0 row6" >gbr</th>
      <td id="T_1e6b1_row6_col0" class="data row6 col0" >Gradient Boosting Regressor</td>
      <td id="T_1e6b1_row6_col1" class="data row6 col1" >0.0014</td>
      <td id="T_1e6b1_row6_col2" class="data row6 col2" >0.0002</td>
      <td id="T_1e6b1_row6_col3" class="data row6 col3" >0.0091</td>
      <td id="T_1e6b1_row6_col4" class="data row6 col4" >0.9998</td>
      <td id="T_1e6b1_row6_col5" class="data row6 col5" >0.0027</td>
      <td id="T_1e6b1_row6_col6" class="data row6 col6" >0.0012</td>
      <td id="T_1e6b1_row6_col7" class="data row6 col7" >0.0100</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row7" class="row_heading level0 row7" >et</th>
      <td id="T_1e6b1_row7_col0" class="data row7 col0" >Extra Trees Regressor</td>
      <td id="T_1e6b1_row7_col1" class="data row7 col1" >0.0034</td>
      <td id="T_1e6b1_row7_col2" class="data row7 col2" >0.0002</td>
      <td id="T_1e6b1_row7_col3" class="data row7 col3" >0.0119</td>
      <td id="T_1e6b1_row7_col4" class="data row7 col4" >0.9997</td>
      <td id="T_1e6b1_row7_col5" class="data row7 col5" >0.0051</td>
      <td id="T_1e6b1_row7_col6" class="data row7 col6" >0.0131</td>
      <td id="T_1e6b1_row7_col7" class="data row7 col7" >0.0140</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row8" class="row_heading level0 row8" >rf</th>
      <td id="T_1e6b1_row8_col0" class="data row8 col0" >Random Forest Regressor</td>
      <td id="T_1e6b1_row8_col1" class="data row8 col1" >0.0022</td>
      <td id="T_1e6b1_row8_col2" class="data row8 col2" >0.0003</td>
      <td id="T_1e6b1_row8_col3" class="data row8 col3" >0.0119</td>
      <td id="T_1e6b1_row8_col4" class="data row8 col4" >0.9997</td>
      <td id="T_1e6b1_row8_col5" class="data row8 col5" >0.0035</td>
      <td id="T_1e6b1_row8_col6" class="data row8 col6" >0.0013</td>
      <td id="T_1e6b1_row8_col7" class="data row8 col7" >0.0190</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row9" class="row_heading level0 row9" >dt</th>
      <td id="T_1e6b1_row9_col0" class="data row9 col0" >Decision Tree Regressor</td>
      <td id="T_1e6b1_row9_col1" class="data row9 col1" >0.0023</td>
      <td id="T_1e6b1_row9_col2" class="data row9 col2" >0.0005</td>
      <td id="T_1e6b1_row9_col3" class="data row9 col3" >0.0162</td>
      <td id="T_1e6b1_row9_col4" class="data row9 col4" >0.9994</td>
      <td id="T_1e6b1_row9_col5" class="data row9 col5" >0.0049</td>
      <td id="T_1e6b1_row9_col6" class="data row9 col6" >0.0010</td>
      <td id="T_1e6b1_row9_col7" class="data row9 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row10" class="row_heading level0 row10" >catboost</th>
      <td id="T_1e6b1_row10_col0" class="data row10 col0" >CatBoost Regressor</td>
      <td id="T_1e6b1_row10_col1" class="data row10 col1" >0.0123</td>
      <td id="T_1e6b1_row10_col2" class="data row10 col2" >0.0006</td>
      <td id="T_1e6b1_row10_col3" class="data row10 col3" >0.0239</td>
      <td id="T_1e6b1_row10_col4" class="data row10 col4" >0.9993</td>
      <td id="T_1e6b1_row10_col5" class="data row10 col5" >0.0099</td>
      <td id="T_1e6b1_row10_col6" class="data row10 col6" >0.0751</td>
      <td id="T_1e6b1_row10_col7" class="data row10 col7" >0.0500</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row11" class="row_heading level0 row11" >xgboost</th>
      <td id="T_1e6b1_row11_col0" class="data row11 col0" >Extreme Gradient Boosting</td>
      <td id="T_1e6b1_row11_col1" class="data row11 col1" >0.0039</td>
      <td id="T_1e6b1_row11_col2" class="data row11 col2" >0.0007</td>
      <td id="T_1e6b1_row11_col3" class="data row11 col3" >0.0190</td>
      <td id="T_1e6b1_row11_col4" class="data row11 col4" >0.9992</td>
      <td id="T_1e6b1_row11_col5" class="data row11 col5" >0.0057</td>
      <td id="T_1e6b1_row11_col6" class="data row11 col6" >0.0026</td>
      <td id="T_1e6b1_row11_col7" class="data row11 col7" >0.0060</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row12" class="row_heading level0 row12" >par</th>
      <td id="T_1e6b1_row12_col0" class="data row12 col0" >Passive Aggressive Regressor</td>
      <td id="T_1e6b1_row12_col1" class="data row12 col1" >0.0280</td>
      <td id="T_1e6b1_row12_col2" class="data row12 col2" >0.0012</td>
      <td id="T_1e6b1_row12_col3" class="data row12 col3" >0.0342</td>
      <td id="T_1e6b1_row12_col4" class="data row12 col4" >0.9987</td>
      <td id="T_1e6b1_row12_col5" class="data row12 col5" >0.0214</td>
      <td id="T_1e6b1_row12_col6" class="data row12 col6" >0.2958</td>
      <td id="T_1e6b1_row12_col7" class="data row12 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row13" class="row_heading level0 row13" >lightgbm</th>
      <td id="T_1e6b1_row13_col0" class="data row13 col0" >Light Gradient Boosting Machine</td>
      <td id="T_1e6b1_row13_col1" class="data row13 col1" >0.0170</td>
      <td id="T_1e6b1_row13_col2" class="data row13 col2" >0.0033</td>
      <td id="T_1e6b1_row13_col3" class="data row13 col3" >0.0549</td>
      <td id="T_1e6b1_row13_col4" class="data row13 col4" >0.9964</td>
      <td id="T_1e6b1_row13_col5" class="data row13 col5" >0.0180</td>
      <td id="T_1e6b1_row13_col6" class="data row13 col6" >0.0193</td>
      <td id="T_1e6b1_row13_col7" class="data row13 col7" >0.2760</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row14" class="row_heading level0 row14" >ada</th>
      <td id="T_1e6b1_row14_col0" class="data row14 col0" >AdaBoost Regressor</td>
      <td id="T_1e6b1_row14_col1" class="data row14 col1" >0.0571</td>
      <td id="T_1e6b1_row14_col2" class="data row14 col2" >0.0066</td>
      <td id="T_1e6b1_row14_col3" class="data row14 col3" >0.0777</td>
      <td id="T_1e6b1_row14_col4" class="data row14 col4" >0.9929</td>
      <td id="T_1e6b1_row14_col5" class="data row14 col5" >0.0387</td>
      <td id="T_1e6b1_row14_col6" class="data row14 col6" >0.4105</td>
      <td id="T_1e6b1_row14_col7" class="data row14 col7" >0.0080</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row15" class="row_heading level0 row15" >knn</th>
      <td id="T_1e6b1_row15_col0" class="data row15 col0" >K Neighbors Regressor</td>
      <td id="T_1e6b1_row15_col1" class="data row15 col1" >0.0613</td>
      <td id="T_1e6b1_row15_col2" class="data row15 col2" >0.0073</td>
      <td id="T_1e6b1_row15_col3" class="data row15 col3" >0.0849</td>
      <td id="T_1e6b1_row15_col4" class="data row15 col4" >0.9921</td>
      <td id="T_1e6b1_row15_col5" class="data row15 col5" >0.0479</td>
      <td id="T_1e6b1_row15_col6" class="data row15 col6" >0.5923</td>
      <td id="T_1e6b1_row15_col7" class="data row15 col7" >0.0050</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row16" class="row_heading level0 row16" >en</th>
      <td id="T_1e6b1_row16_col0" class="data row16 col0" >Elastic Net</td>
      <td id="T_1e6b1_row16_col1" class="data row16 col1" >0.5120</td>
      <td id="T_1e6b1_row16_col2" class="data row16 col2" >0.3988</td>
      <td id="T_1e6b1_row16_col3" class="data row16 col3" >0.6304</td>
      <td id="T_1e6b1_row16_col4" class="data row16 col4" >0.5720</td>
      <td id="T_1e6b1_row16_col5" class="data row16 col5" >0.3411</td>
      <td id="T_1e6b1_row16_col6" class="data row16 col6" >1.0179</td>
      <td id="T_1e6b1_row16_col7" class="data row16 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row17" class="row_heading level0 row17" >lasso</th>
      <td id="T_1e6b1_row17_col0" class="data row17 col0" >Lasso Regression</td>
      <td id="T_1e6b1_row17_col1" class="data row17 col1" >0.7867</td>
      <td id="T_1e6b1_row17_col2" class="data row17 col2" >0.9432</td>
      <td id="T_1e6b1_row17_col3" class="data row17 col3" >0.9701</td>
      <td id="T_1e6b1_row17_col4" class="data row17 col4" >-0.0141</td>
      <td id="T_1e6b1_row17_col5" class="data row17 col5" >0.5793</td>
      <td id="T_1e6b1_row17_col6" class="data row17 col6" >1.3910</td>
      <td id="T_1e6b1_row17_col7" class="data row17 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row18" class="row_heading level0 row18" >llar</th>
      <td id="T_1e6b1_row18_col0" class="data row18 col0" >Lasso Least Angle Regression</td>
      <td id="T_1e6b1_row18_col1" class="data row18 col1" >0.7867</td>
      <td id="T_1e6b1_row18_col2" class="data row18 col2" >0.9432</td>
      <td id="T_1e6b1_row18_col3" class="data row18 col3" >0.9701</td>
      <td id="T_1e6b1_row18_col4" class="data row18 col4" >-0.0141</td>
      <td id="T_1e6b1_row18_col5" class="data row18 col5" >0.5793</td>
      <td id="T_1e6b1_row18_col6" class="data row18 col6" >1.3910</td>
      <td id="T_1e6b1_row18_col7" class="data row18 col7" >0.0040</td>
    </tr>
    <tr>
      <th id="T_1e6b1_level0_row19" class="row_heading level0 row19" >dummy</th>
      <td id="T_1e6b1_row19_col0" class="data row19 col0" >Dummy Regressor</td>
      <td id="T_1e6b1_row19_col1" class="data row19 col1" >0.7867</td>
      <td id="T_1e6b1_row19_col2" class="data row19 col2" >0.9432</td>
      <td id="T_1e6b1_row19_col3" class="data row19 col3" >0.9701</td>
      <td id="T_1e6b1_row19_col4" class="data row19 col4" >-0.0141</td>
      <td id="T_1e6b1_row19_col5" class="data row19 col5" >0.5793</td>
      <td id="T_1e6b1_row19_col6" class="data row19 col6" >1.3910</td>
      <td id="T_1e6b1_row19_col7" class="data row19 col7" >0.0040</td>
    </tr>
  </tbody>
</table>










<style>#sk-container-id-6 {
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

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
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

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
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

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
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

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
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

#sk-container-id-6 a.estimator_doc_link {
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

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression(n_jobs=-1)</pre></div> </div></div></div></div>




```python

```
