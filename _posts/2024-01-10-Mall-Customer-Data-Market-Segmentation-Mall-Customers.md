---
title: "Mall Customer Data Market Segmentation Mall Customers"
date: 2024-01-10
last_modified_at: 2024-01-10
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Mall Customer Data Market Segmentation Mall Customers 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "vjchoudhary7/customer-segmentation-tutorial-in-python"
)

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv(os.path.join(path, "Mall_Customers.csv"))
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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>Female</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>197</td>
      <td>Female</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>198</td>
      <td>Male</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>199</td>
      <td>Male</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>200</td>
      <td>Male</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 5 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 5 columns):
     #   Column                  Non-Null Count  Dtype 
    ---  ------                  --------------  ----- 
     0   CustomerID              200 non-null    int64 
     1   Gender                  200 non-null    object
     2   Age                     200 non-null    int64 
     3   Annual Income (k$)      200 non-null    int64 
     4   Spending Score (1-100)  200 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 7.9+ KB



```python
df.shape
```




    (200, 5)




```python
df.drop(["CustomerID"], axis=1, inplace=True)
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
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>Female</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Female</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Male</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Male</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Male</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 4 columns</p>
</div>




```python
df["Gender"] = df["Gender"].replace("Male", 1)
df["Gender"] = df["Gender"].replace("Female", 0)
```

    /var/folders/v7/tlyx9w190ks2gfgzd_j0l5c80000gn/T/ipykernel_85459/1716590208.py:2: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df["Gender"] = df["Gender"].replace("Female", 0)



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
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 4 columns</p>
</div>




```python
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```


```python
scaled_df
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
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.128152</td>
      <td>-1.424569</td>
      <td>-1.738999</td>
      <td>-0.434801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.128152</td>
      <td>-1.281035</td>
      <td>-1.738999</td>
      <td>1.195704</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.886405</td>
      <td>-1.352802</td>
      <td>-1.700830</td>
      <td>-1.715913</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.886405</td>
      <td>-1.137502</td>
      <td>-1.700830</td>
      <td>1.040418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.886405</td>
      <td>-0.563369</td>
      <td>-1.662660</td>
      <td>-0.395980</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>-0.886405</td>
      <td>-0.276302</td>
      <td>2.268791</td>
      <td>1.118061</td>
    </tr>
    <tr>
      <th>196</th>
      <td>-0.886405</td>
      <td>0.441365</td>
      <td>2.497807</td>
      <td>-0.861839</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1.128152</td>
      <td>-0.491602</td>
      <td>2.497807</td>
      <td>0.923953</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1.128152</td>
      <td>-0.491602</td>
      <td>2.917671</td>
      <td>-1.250054</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1.128152</td>
      <td>-0.635135</td>
      <td>2.917671</td>
      <td>1.273347</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 4 columns</p>
</div>




```python
max_clusters = 50
```


```python
kmeans_test = [KMeans(n_clusters=i, n_init=10) for i in range(1, max_clusters)]
inertias = [kmeans_test[i].fit(scaled_df).inertia_ for i in range(len(kmeans_test))]
```


```python
plt.figure(figsize=(7, 5))
plt.plot(range(1, max_clusters), inertias)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertias")
plt.title("Choosing the Number of Clusters")
plt.show()
```


    
![png](010_Mall_Customer_Data_Market_Segmentation_Mall_Customers_files/010_Mall_Customer_Data_Market_Segmentation_Mall_Customers_14_0.png)
    



```python
kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(scaled_df)
```




<style>#sk-container-id-2 {
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

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
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

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
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

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
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

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
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

#sk-container-id-2 a.estimator_doc_link {
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

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=10, n_init=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;KMeans<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html">?<span>Documentation for KMeans</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>KMeans(n_clusters=10, n_init=10)</pre></div> </div></div></div></div>




```python
clusters = kmeans.predict(scaled_df)
```


```python
clusters
```




    array([1, 1, 2, 7, 2, 7, 2, 7, 4, 7, 4, 7, 0, 7, 4, 1, 2, 1, 4, 7, 1, 1,
           0, 1, 0, 1, 0, 1, 2, 7, 4, 7, 4, 1, 0, 7, 0, 7, 2, 7, 0, 1, 5, 2,
           0, 7, 0, 2, 2, 2, 0, 1, 2, 5, 0, 5, 0, 5, 2, 5, 5, 1, 0, 0, 5, 1,
           0, 0, 1, 2, 5, 0, 0, 0, 5, 1, 0, 1, 2, 0, 5, 1, 5, 0, 2, 5, 0, 2,
           2, 0, 0, 1, 5, 2, 2, 1, 0, 2, 5, 1, 2, 0, 5, 1, 5, 2, 0, 5, 5, 5,
           5, 2, 2, 1, 2, 2, 0, 0, 0, 0, 1, 2, 6, 3, 2, 6, 8, 3, 8, 3, 8, 3,
           2, 6, 8, 6, 9, 3, 8, 6, 9, 3, 2, 6, 8, 3, 8, 6, 9, 3, 8, 3, 9, 6,
           9, 6, 8, 6, 8, 6, 9, 6, 8, 6, 8, 6, 8, 6, 9, 3, 8, 3, 8, 3, 9, 6,
           8, 3, 8, 3, 9, 6, 8, 6, 9, 3, 9, 3, 9, 6, 9, 6, 8, 6, 9, 6, 9, 3,
           8, 3], dtype=int32)




```python
pca = PCA(n_components=2)
```


```python
reduced_data = pd.DataFrame(pca.fit_transform(scaled_df), columns=["PC1", "PC2"])
```


```python
reduced_data
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
      <th>PC1</th>
      <th>PC2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.406383</td>
      <td>-0.520714</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.427673</td>
      <td>-0.367310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.050761</td>
      <td>-1.894068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.694513</td>
      <td>-1.631908</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.313108</td>
      <td>-1.810483</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>-1.179572</td>
      <td>1.324568</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0.672751</td>
      <td>1.221061</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-0.723719</td>
      <td>2.765010</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.767096</td>
      <td>2.861930</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1.065015</td>
      <td>3.137256</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>




```python
kmeans.cluster_centers_
```




    array([[-0.88640526,  1.01325595, -0.47702244, -0.2952865 ],
           [ 1.12815215, -0.97602698, -0.73705168,  0.41603773],
           [-0.88640526, -0.74039302, -0.39925223, -0.31316059],
           [ 1.12815215, -0.39989994,  1.01344075,  1.26040667],
           [ 1.12815215,  1.19491538, -1.39547433, -1.51533492],
           [ 1.12815215,  1.42815712, -0.2675677 , -0.06405558],
           [-0.88640526, -0.45245636,  0.94327069,  1.17982252],
           [-0.88640526, -0.96084556, -1.33087991,  1.17778643],
           [ 1.12815215,  0.04664835,  0.93858626, -1.40339942],
           [-0.88640526,  0.41265847,  1.21277   , -1.11029664]])




```python
reduced_centers = pca.transform(kmeans.cluster_centers_)
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but PCA was fitted with feature names
      warnings.warn(



```python
reduced_centers
```




    array([[ 0.69507241, -1.05625787],
           [-0.68838314,  0.28733559],
           [-0.4994581 , -0.8167538 ],
           [-0.88272588,  1.65431318],
           [ 2.13571172, -0.64096681],
           [ 1.2923859 ,  0.34776935],
           [-1.33511175,  0.33485089],
           [-1.6696024 , -1.35294268],
           [ 1.25473165,  1.27579377],
           [ 0.83149037,  0.21501655]])




```python
reduced_data["cluster"] = clusters
```


```python
reduced_data
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
      <th>PC1</th>
      <th>PC2</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.406383</td>
      <td>-0.520714</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.427673</td>
      <td>-0.367310</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.050761</td>
      <td>-1.894068</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.694513</td>
      <td>-1.631908</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.313108</td>
      <td>-1.810483</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>-1.179572</td>
      <td>1.324568</td>
      <td>6</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0.672751</td>
      <td>1.221061</td>
      <td>9</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-0.723719</td>
      <td>2.765010</td>
      <td>3</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.767096</td>
      <td>2.861930</td>
      <td>8</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1.065015</td>
      <td>3.137256</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 3 columns</p>
</div>




```python
reduced_data[reduced_data["cluster"] == 7]
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
      <th>PC1</th>
      <th>PC2</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>-1.694513</td>
      <td>-1.631908</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.717446</td>
      <td>-1.599264</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.148322</td>
      <td>-1.505374</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.216295</td>
      <td>-1.616405</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-1.689470</td>
      <td>-1.545428</td>
      <td>7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.646073</td>
      <td>-1.522513</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-1.663732</td>
      <td>-1.432598</td>
      <td>7</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-1.964204</td>
      <td>-1.212120</td>
      <td>7</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-1.689831</td>
      <td>-1.224123</td>
      <td>7</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-1.903866</td>
      <td>-1.104441</td>
      <td>7</td>
    </tr>
    <tr>
      <th>37</th>
      <td>-1.246444</td>
      <td>-1.174259</td>
      <td>7</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-1.794159</td>
      <td>-1.004204</td>
      <td>7</td>
    </tr>
    <tr>
      <th>45</th>
      <td>-1.330477</td>
      <td>-1.015619</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(14, 10))
plt.scatter(
    reduced_data[reduced_data["cluster"] == 0].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 0].loc[:, "PC2"],
    c="red",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 1].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 1].loc[:, "PC2"],
    c="blue",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 2].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 2].loc[:, "PC2"],
    c="green",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 3].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 3].loc[:, "PC2"],
    c="yellow",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 4].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 4].loc[:, "PC2"],
    c="purple",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 5].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 5].loc[:, "PC2"],
    c="orange",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 6].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 6].loc[:, "PC2"],
    c="pink",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 7].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 7].loc[:, "PC2"],
    c="brown",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 8].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 8].loc[:, "PC2"],
    c="gray",
)
plt.scatter(
    reduced_data[reduced_data["cluster"] == 9].loc[:, "PC1"],
    reduced_data[reduced_data["cluster"] == 9].loc[:, "PC2"],
    c="cyan",
)

plt.scatter(
    reduced_centers[:, 0], reduced_centers[:, 1], color="black", marker="x", s=300
)


plt.xlabel("PC1")
plt.ylabel("PC2")
```




    Text(0, 0.5, 'PC2')




    
![png](010_Mall_Customer_Data_Market_Segmentation_Mall_Customers_files/010_Mall_Customer_Data_Market_Segmentation_Mall_Customers_27_1.png)
    



```python
reduced_centers[:, 0]
```




    array([[-0.88272588,  1.65431318],
           [ 0.58233488, -0.85939176],
           [ 1.47661839,  0.1540349 ],
           [-0.662429  , -0.58044771],
           [ 0.71982753, -1.68765552],
           [ 1.19961046,  1.30582744],
           [-0.73489077,  0.27816597],
           [-1.38150389,  0.3644368 ],
           [-1.6696024 , -1.35294268],
           [ 0.81659377,  0.24505923]])




```python

```
