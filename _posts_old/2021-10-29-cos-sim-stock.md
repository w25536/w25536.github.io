---
layout: page
title:  "코사인 유사도(cosine similarity)로 과거 주가의 유사 패턴을 찾아 미래 예측하기"
description: "코사인 유사도(cosine similarity)로 과거 주가의 유사 패턴을 찾아 미래 예측하는 방법을 알아 보도록 하겠습니다."
headline: "코사인 유사도(cosine similarity)로 과거 주가의 유사 패턴을 찾아 미래 예측하는 방법을 알아 보도록 하겠습니다."
categories: pandas
tags: [python, 인공지능 책, 테디노트 책, 테디노트, 파이썬, 딥러닝 책 추천, 파이썬 책 추천, 머신러닝 책 추천, 파이썬 딥러닝 텐서플로, 텐서플로우 책 추천, 텐서플로 책, 인공지능 서적, data science, 데이터 분석, 딥러닝]
comments: true
published: true
---

주가의 과거 패턴을 찾아 미래를 예측하는 것이 가능할까요?

이를 직접 눈으로 확인해 보기 위하여 코드로 직접 구현해 봤습니다.

데모 페이지는 [주식 패턴 검색기](http://teddynote.herokuapp.com/stock)에서 확인해 보실 수 있습니다.

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 패키지 설치


본 실습에 대한 진행을 위해서는 `numpy`, `pandas`, `matplotlib`외에도



**주가 데이터 로드** 및 **주식 캔들차트 시각화**를 위하여



`finance-datareader` 와 `mpl_finance` 라이브러리 설치가 필요합니다.


**패키지 정보**



- [FinanceDataReder](https://github.com/FinanceData/FinanceDataReader): 주가 정보 로드

- [mpl-finance](https://github.com/matplotlib/mpl-finance): candle 차트 시각화



```python
# finance-datareader 설치
!pip install finance-datareader

# mpl-finance
!pip install mpl-finance
```

## 모듈 import 



```python
import numpy as np
import pandas as pd

import FinanceDataReader as fdr
from mpl_finance import candlestick_ohlc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 차트의 기본 캔버스 크기 설정
plt.rcParams["figure.figsize"] = (10, 8)
# 차트의 기본 라인 굵기 설정
plt.rcParams['lines.linewidth'] = 1.5
# 차트의 기본 라인 컬러 설정
plt.rcParams['lines.color'] = 'tomato'
```

`finance-datareader`로부터 **코스피(Kospi)** 정보 불러오기



```python
data = fdr.DataReader('KS11')
data
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
      <th>Close</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>1981-05-01</th>
      <td>123.60</td>
      <td>123.60</td>
      <td>123.60</td>
      <td>123.60</td>
      <td>3330000.0</td>
      <td>0.0098</td>
    </tr>
    <tr>
      <th>1981-05-02</th>
      <td>123.50</td>
      <td>123.50</td>
      <td>123.50</td>
      <td>123.50</td>
      <td>2040000.0</td>
      <td>-0.0008</td>
    </tr>
    <tr>
      <th>1981-05-04</th>
      <td>120.60</td>
      <td>120.60</td>
      <td>120.60</td>
      <td>120.60</td>
      <td>1930000.0</td>
      <td>-0.0235</td>
    </tr>
    <tr>
      <th>1981-05-06</th>
      <td>120.70</td>
      <td>120.70</td>
      <td>120.70</td>
      <td>120.70</td>
      <td>1690000.0</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>1981-05-07</th>
      <td>119.30</td>
      <td>119.30</td>
      <td>119.30</td>
      <td>119.30</td>
      <td>1480000.0</td>
      <td>-0.0116</td>
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
      <th>2021-10-25</th>
      <td>3020.54</td>
      <td>3001.10</td>
      <td>3025.27</td>
      <td>2983.29</td>
      <td>791800000.0</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>2021-10-26</th>
      <td>3049.08</td>
      <td>3039.82</td>
      <td>3051.65</td>
      <td>3030.53</td>
      <td>564560000.0</td>
      <td>0.0094</td>
    </tr>
    <tr>
      <th>2021-10-27</th>
      <td>3025.49</td>
      <td>3045.83</td>
      <td>3049.02</td>
      <td>3019.00</td>
      <td>607880000.0</td>
      <td>-0.0077</td>
    </tr>
    <tr>
      <th>2021-10-28</th>
      <td>3009.55</td>
      <td>3023.17</td>
      <td>3034.42</td>
      <td>3009.55</td>
      <td>617260000.0</td>
      <td>-0.0053</td>
    </tr>
    <tr>
      <th>2021-10-29</th>
      <td>2970.68</td>
      <td>3025.67</td>
      <td>3030.17</td>
      <td>2965.40</td>
      <td>535480.0</td>
      <td>-0.0129</td>
    </tr>
  </tbody>
</table>
<p>10816 rows × 6 columns</p>
</div>


`finance-datareader`로부터 **주가 종목 코드로** 주가정보 불러오기



```python
# 카카오(035720) 종목코드 입력
data = fdr.DataReader('035720')
data
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>1999-11-11</th>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1999-11-12</th>
      <td>1115</td>
      <td>1115</td>
      <td>1115</td>
      <td>1115</td>
      <td>140</td>
      <td>0.116116</td>
    </tr>
    <tr>
      <th>1999-11-15</th>
      <td>1249</td>
      <td>1249</td>
      <td>1249</td>
      <td>1249</td>
      <td>405</td>
      <td>0.120179</td>
    </tr>
    <tr>
      <th>1999-11-16</th>
      <td>1396</td>
      <td>1396</td>
      <td>1396</td>
      <td>1396</td>
      <td>214</td>
      <td>0.117694</td>
    </tr>
    <tr>
      <th>1999-11-17</th>
      <td>1561</td>
      <td>1561</td>
      <td>1561</td>
      <td>1561</td>
      <td>191</td>
      <td>0.118195</td>
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
      <th>2021-10-25</th>
      <td>126500</td>
      <td>127500</td>
      <td>123500</td>
      <td>126000</td>
      <td>2309608</td>
      <td>-0.011765</td>
    </tr>
    <tr>
      <th>2021-10-26</th>
      <td>126000</td>
      <td>128000</td>
      <td>126000</td>
      <td>127500</td>
      <td>1265299</td>
      <td>0.011905</td>
    </tr>
    <tr>
      <th>2021-10-27</th>
      <td>128000</td>
      <td>129000</td>
      <td>126500</td>
      <td>128500</td>
      <td>1449071</td>
      <td>0.007843</td>
    </tr>
    <tr>
      <th>2021-10-28</th>
      <td>129000</td>
      <td>130000</td>
      <td>125000</td>
      <td>125500</td>
      <td>1783396</td>
      <td>-0.023346</td>
    </tr>
    <tr>
      <th>2021-10-29</th>
      <td>125500</td>
      <td>126500</td>
      <td>123000</td>
      <td>125500</td>
      <td>1872380</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5422 rows × 6 columns</p>
</div>


## 주가 차트와 거래량 정보 시각화


시작일(startdate), 종료일(enddate)를 정의합니다.



```python
startdate = '2021-09-01'
enddate = '2021-09-20'
```

`startdate`와 `enddate`사이에 있는 데이터만 추출합니다.



```python
# 주가 정보의 시작: 종료 추출
data_ = data.loc[startdate:enddate]
data_
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2021-09-01</th>
      <td>155000</td>
      <td>156000</td>
      <td>154000</td>
      <td>154000</td>
      <td>2011434</td>
      <td>-0.006452</td>
    </tr>
    <tr>
      <th>2021-09-02</th>
      <td>154500</td>
      <td>156000</td>
      <td>153500</td>
      <td>155000</td>
      <td>1649156</td>
      <td>0.006494</td>
    </tr>
    <tr>
      <th>2021-09-03</th>
      <td>155500</td>
      <td>157500</td>
      <td>154500</td>
      <td>156500</td>
      <td>1934669</td>
      <td>0.009677</td>
    </tr>
    <tr>
      <th>2021-09-06</th>
      <td>156000</td>
      <td>156500</td>
      <td>152500</td>
      <td>155500</td>
      <td>1883428</td>
      <td>-0.006390</td>
    </tr>
    <tr>
      <th>2021-09-07</th>
      <td>155500</td>
      <td>156000</td>
      <td>153500</td>
      <td>154000</td>
      <td>1072249</td>
      <td>-0.009646</td>
    </tr>
    <tr>
      <th>2021-09-08</th>
      <td>151500</td>
      <td>151500</td>
      <td>136500</td>
      <td>138500</td>
      <td>16920382</td>
      <td>-0.100649</td>
    </tr>
    <tr>
      <th>2021-09-09</th>
      <td>134000</td>
      <td>134500</td>
      <td>128000</td>
      <td>128500</td>
      <td>14534253</td>
      <td>-0.072202</td>
    </tr>
    <tr>
      <th>2021-09-10</th>
      <td>127000</td>
      <td>133500</td>
      <td>126000</td>
      <td>130000</td>
      <td>9918050</td>
      <td>0.011673</td>
    </tr>
    <tr>
      <th>2021-09-13</th>
      <td>126500</td>
      <td>130000</td>
      <td>122500</td>
      <td>124500</td>
      <td>8675498</td>
      <td>-0.042308</td>
    </tr>
    <tr>
      <th>2021-09-14</th>
      <td>122500</td>
      <td>126000</td>
      <td>118000</td>
      <td>124000</td>
      <td>18895148</td>
      <td>-0.004016</td>
    </tr>
    <tr>
      <th>2021-09-15</th>
      <td>123500</td>
      <td>127500</td>
      <td>122000</td>
      <td>122500</td>
      <td>9078817</td>
      <td>-0.012097</td>
    </tr>
    <tr>
      <th>2021-09-16</th>
      <td>123000</td>
      <td>125000</td>
      <td>121000</td>
      <td>121500</td>
      <td>4770936</td>
      <td>-0.008163</td>
    </tr>
    <tr>
      <th>2021-09-17</th>
      <td>121500</td>
      <td>121500</td>
      <td>118000</td>
      <td>119500</td>
      <td>4807631</td>
      <td>-0.016461</td>
    </tr>
  </tbody>
</table>
</div>


**주식 차트 생성**



```python
fig = plt.figure()
fig.set_facecolor('w')
# 2개의 캔버스 생성 후 1번째는 차트를 2번째는 거래량
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

axes = []
axes.append(plt.subplot(gs[0]))
axes.append(plt.subplot(gs[1], sharex=axes[0]))
axes[0].get_xaxis().set_visible(False)

x = np.arange(len(data_.index))
ohlc = data_[['Open', 'High', 'Low', 'Close']].values
dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))

# 봉차트
candlestick_ohlc(axes[0], dohlc, width=0.5, colorup='r', colordown='b')

# 거래량 차트
axes[1].bar(x, data_['Volume'], color='grey', width=0.6, align='center')
axes[1].set_xticks(range(len(x)))
axes[1].set_xticklabels(list(data_.index.strftime('%Y-%m-%d')), rotation=90)
axes[1].get_yaxis().set_visible(False)

plt.tight_layout()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsgAAAI4CAYAAAB3OR9vAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAA7BElEQVR4nO3dfXRV1YH+8edCkKnDApLwkkwSiXADKCEEvECsiiKNFOoKdsnwUkYSQ80qpaXFVmFWq5B2lKCtdaZIMTaVMFNJLbTEdiBisZnaFswKmFpJqwETTEJIk5DA1FqQsH9/MN7fjkDeINn3hO9nrbtWss+55z4nXOLjYZ99fcYYIwAAAACSpH6uAwAAAAChhIIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAIAlzHWAK23YsGGKj493HQMAAAAhrqqqSo2NjReM97mCHB8fr9LSUtcxAAAAEOICgcBFx5liAQAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMi4Om3Z4joBAAAIURRkXJ2qqlwnAAAAIYqCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCjL4hKkry+Tr/yM7u2v5RUa7PMIglnAEA6FkUZPQN9fXePn4XsIQzAAA9i4IMAAAAWCjIAAAAgIWC7Jin55N6Onxo6coUag9Pn5bE2wYAEPooyI55ej6pp8OHlp6c4hxC06cl8bYBAIQ+CjIAAABgoSADAAAAFgoy+oaRIz19/J48fE//aHpyCepQmz8NALg6dFiQMzMzNWLECCUmJgbH1q1bp5iYGCUnJys5OVm7du2SJFVVVekTn/hEcPwLX/hC8DkHDhzQxIkT5ff7tXLlShljJEknTpxQamqqEhISlJqaqubmZkmSMUYrV66U3+9XUlKSDh48eEVPHH3M8eOSMZ1/rF3btf2PHw+Z+CEW/aqaPw0AuDp0WJAzMjJUVFR0wfiqVatUVlamsrIyzZ07Nzg+ZsyY4PjmzZuD48uXL9dzzz2niooKVVRUBI+Zk5OjWbNmqaKiQrNmzVJOTo4kaffu3cF9c3NztXz58ss+WQAAAKAjHRbkGTNmKCIi4rJepK6uTqdOnVJKSop8Pp+WLl2qnTt3SpIKCwuVnp4uSUpPT28zvnTpUvl8PqWkpKilpUV1dXWXlQMAAADoSLfnIG/cuFFJSUnKzMwMTouQpMrKSk2ePFm33367XnvtNUlSbW2tYmNjg/vExsaqtrZWklRfX6/o6GhJUlRUlOr/799Ua2trFRcXd9HnfFxubq4CgYACgYAaGhq6e0pXRE/Ox+zxOZmeDg8AAHBldKsgL1++XEeOHFFZWZmio6P1ta99TZIUHR2t9957T2+88Yaeeuopfe5zn9OpU6c6fVyfzyefz9flPFlZWSotLVVpaamGDx/e5edfST09Z7JHj+/p8AAAAFdGtwryyJEj1b9/f/Xr108PPPCASkpKJEkDBw5UZGSkJOmmm27SmDFj9M477ygmJkY1NTXB59fU1CgmJiZ4rI+mTtTV1WnEiBGSpJiYGFVXV1/0OQAAAEBP6VZBtucC//znPw+ucNHQ0KDW1lZJ0rvvvquKigqNHj1a0dHRGjx4sPbv3y9jjLZu3ap58+ZJktLS0pSfny9Jys/PbzO+detWGWO0f/9+DRkyJDgVAwAAAOgpYR3tsHjxYhUXF6uxsVGxsbHKzs5WcXGxysrK5PP5FB8fr2effVaS9Jvf/EaPPvqoBgwYoH79+mnz5s3BG/w2bdqkjIwMffDBB5ozZ47mzJkjSVqzZo0WLFigvLw8jRo1Si+++KIkae7cudq1a5f8fr+uvfZaPf/88z31M8BHRo7s2WkQPb0gb1fEx7tOAAAAQpTPfLQgcR8RCARUWlrq7PW7MYW6y0LmT2zduvMP9KpQ+7H39Hs+ZN7vAIA+51K9kU/SAwAAACwUZAAAAMBCQb7CenqabShN4wWknn1P8n4HALhAQb7Cjh8/P2eypx7Hj7s+Q6Ctrr7n167l/Q4ACG0UZAAAAMBCQQYAAAAsFGR0H2sJoxt42wAAQh0FGd2XkeE6ATyItw0AINRRkAEAAAALBRkAAACwUJABAAAACwUZAAAAsFCQAQAAAAsFGQAAALBQkAEAAAALBRnwGD5oAwCAnkVBBjyGD9oAAKBnUZABAAAACwUZAAAAsFCQAQAAAAsFGQAAALBQkAEAAAALBRkAAACwUJABXLWioiSfr2ceUVGuzw4A0F0UZABXrfp6bx4bANCzKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACApcOCnJmZqREjRigxMTE4tm7dOsXExCg5OVnJycnatWtXcNv69evl9/s1btw4vfzyy8HxoqIijRs3Tn6/Xzk5OcHxyspKTZ8+XX6/XwsXLtSZM2ckSadPn9bChQvl9/s1ffp0VVVVXYnzBQAAANrVYUHOyMhQUVHRBeOrVq1SWVmZysrKNHfuXElSeXm5CgoKdOjQIRUVFemLX/yiWltb1draqhUrVmj37t0qLy/Xtm3bVF5eLklavXq1Vq1apcOHDys8PFx5eXmSpLy8PIWHh+vw4cNatWqVVq9efSXPGwAAALioDgvyjBkzFBER0amDFRYWatGiRRo4cKCuv/56+f1+lZSUqKSkRH6/X6NHj9Y111yjRYsWqbCwUMYYvfrqq5o/f74kKT09XTt37gweKz09XZI0f/587d27V8aYbp4mAAAA0DndnoO8ceNGJSUlKTMzU83NzZKk2tpaxcXFBfeJjY1VbW3tJcebmpo0dOhQhYWFtRn/+LHCwsI0ZMgQNTU1XTRLbm6uAoGAAoGAGhoauntKAAAAQPcK8vLly3XkyBGVlZUpOjpaX/va1650ri7JyspSaWmpSktLNXz4cKdZAAAA4G3dKsgjR45U//791a9fPz3wwAMqKSmRJMXExKi6ujq4X01NjWJiYi45HhkZqZaWFp09e7bN+MePdfbsWZ08eVKRkZHdO0sAAACgk7pVkOvq6oJf//znPw+ucJGWlqaCggKdPn1alZWVqqio0LRp0zR16lRVVFSosrJSZ86cUUFBgdLS0uTz+TRz5kxt375dkpSfn6958+YFj5Wfny9J2r59u+688075fL7LOlkAAACgI2Ed7bB48WIVFxersbFRsbGxys7OVnFxscrKyuTz+RQfH69nn31WkjRhwgQtWLBAN954o8LCwvTMM8+of//+ks7PWZ49e7ZaW1uVmZmpCRMmSJI2bNigRYsW6Zvf/KYmT56sZcuWSZKWLVum++67T36/XxERESooKOipnwEAAAAQ5DN9bGmIQCCg0tJS1zEAeEBP/6NU3/rtCgB9z6V6I5+kBwAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAICFggwAAABYKMgAAACAhYIMAAAAWCjIAAAAgIWCDAAAAFgoyAAAAIClw4KcmZmpESNGKDEx8YJt3/3ud+Xz+dTY2ChJKi4u1pAhQ5ScnKzk5GR961vfCu5bVFSkcePGye/3KycnJzheWVmp6dOny+/3a+HChTpz5owk6fTp01q4cKH8fr+mT5+uqqqqyz1XAAAAoEMdFuSMjAwVFRVdMF5dXa09e/bouuuuazN+2223qaysTGVlZXr00UclSa2trVqxYoV2796t8vJybdu2TeXl5ZKk1atXa9WqVTp8+LDCw8OVl5cnScrLy1N4eLgOHz6sVatWafXq1Zd9sgAAAEBHOizIM2bMUERExAXjq1at0hNPPCGfz9fhi5SUlMjv92v06NG65pprtGjRIhUWFsoYo1dffVXz58+XJKWnp2vnzp2SpMLCQqWnp0uS5s+fr71798oY05VzAwAAALqsW3OQCwsLFRMTo0mTJl2wbd++fZo0aZLmzJmjQ4cOSZJqa2sVFxcX3Cc2Nla1tbVqamrS0KFDFRYW1mb8488JCwvTkCFD1NTUdNE8ubm5CgQCCgQCamho6M4pAQAAAJKksK4+4W9/+5sef/xx7dmz54JtU6ZM0dGjRzVo0CDt2rVL99xzjyoqKq5I0PZkZWUpKytLkhQIBHr89QAAANB3dfkK8pEjR1RZWalJkyYpPj5eNTU1mjJlio4fP67Bgwdr0KBBkqS5c+fqww8/VGNjo2JiYlRdXR08Rk1NjWJiYhQZGamWlhadPXu2zbikNs85e/asTp48qcjIyMs+YQAAAKA9XS7IEydO1F/+8hdVVVWpqqpKsbGxOnjwoKKionT8+PHgPOGSkhKdO3dOkZGRmjp1qioqKlRZWakzZ86ooKBAaWlp8vl8mjlzprZv3y5Jys/P17x58yRJaWlpys/PlyRt375dd955Z6fmOwMAAACXo8OCvHjxYt188816++23FRsbG1xl4mK2b9+uxMRETZo0SStXrlRBQYF8Pp/CwsK0ceNGzZ49WzfccIMWLFigCRMmSJI2bNigp556Sn6/X01NTVq2bJkkadmyZWpqapLf79dTTz3VZmk4AAAAoKf4TB9bGiIQCKi0tNR1DAAe0NP/KNW3frsCQN9zqd7IJ+kBAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYOlUQc7MzNSIESOUmJh4wbbvfve78vl8amxslCQZY7Ry5Ur5/X4lJSXp4MGDwX3z8/OVkJCghIQE5efnB8cPHDigiRMnyu/3a+XKlTLGSJJOnDih1NRUJSQkKDU1Vc3NzZd1sgAAAEBHOlWQMzIyVFRUdMF4dXW19uzZo+uuuy44tnv3blVUVKiiokK5ublavny5pPNlNzs7W6+//rpKSkqUnZ0dLLzLly/Xc889F3zeR6+Vk5OjWbNmqaKiQrNmzVJOTs5lnzAAAADQnk4V5BkzZigiIuKC8VWrVumJJ56Qz+cLjhUWFmrp0qXy+XxKSUlRS0uL6urq9PLLLys1NVUREREKDw9XamqqioqKVFdXp1OnTiklJUU+n09Lly7Vzp07g8dKT0+XJKWnpwfHAQAAgJ4S1t0nFhYWKiYmRpMmTWozXltbq7i4uOD3sbGxqq2tbXc8Njb2gnFJqq+vV3R0tCQpKipK9fX1F82Sm5ur3NxcSVJDQ0N3TwkAAADoXkH+29/+pscff1x79uy50nkuyefztblSbcvKylJWVpYkKRAI9FomAAAA9D3dWsXiyJEjqqys1KRJkxQfH6+amhpNmTJFx48fV0xMjKqrq4P71tTUKCYmpt3xmpqaC8YlaeTIkaqrq5Mk1dXVacSIEd06SQAAAKCzulWQJ06cqL/85S+qqqpSVVWVYmNjdfDgQUVFRSktLU1bt26VMUb79+/XkCFDFB0drdmzZ2vPnj1qbm5Wc3Oz9uzZo9mzZys6OlqDBw/W/v37ZYzR1q1bNW/ePElSWlpacLWL/Pz84DgAAADQUzpVkBcvXqybb75Zb7/9tmJjY5WXl3fJfefOnavRo0fL7/frgQce0KZNmyRJEREReuSRRzR16lRNnTpVjz76aPDGv02bNunzn/+8/H6/xowZozlz5kiS1qxZo1deeUUJCQn61a9+pTVr1lzu+QIAAADt8pmPFh3uIwKBgEpLS13HAOABl7it4YrpW79dAaDvuVRv5JP0AAAAAAsFGQAAALBQkAEAAAALBRkAAACwUJABAAAACwUZAAAAsFCQAVy1Ro705rEBAD2LggzgqnX8+Pm1ijv7WLu28/seP+767AAA3UVBBgAAACwUZAAAAMBCQQYAAAAsFGQAAADAQkEGAAAALBRkAAAAwEJBBgAAACwUZAAAAMBCQQYAAAAsFGQAAADAQkEGAAAALBRkAAAAwEJBBgAAACwUZAAAAMBCQQYAAAAsFGQA6KT4eNcJAAC9gYIMAJ2UkeE6AQCgN1CQAQAAAAsFGQAAALBQkAEAAAALBRkAAACwUJABAAAACwUZAAAAsFCQAQAAAEuHBTkzM1MjRoxQYmJicOyRRx5RUlKSkpOTddddd+nYsWOSpOLiYg0ZMkTJyclKTk7Wt771reBzioqKNG7cOPn9fuXk5ATHKysrNX36dPn9fi1cuFBnzpyRJJ0+fVoLFy6U3+/X9OnTVVVVdaXOGQAAALikDgtyRkaGioqK2ow99NBDevPNN1VWVqa77767TRG+7bbbVFZWprKyMj366KOSpNbWVq1YsUK7d+9WeXm5tm3bpvLycknS6tWrtWrVKh0+fFjh4eHKy8uTJOXl5Sk8PFyHDx/WqlWrtHr16it20gAAAMCldFiQZ8yYoYiIiDZjgwcPDn79/vvvy+fztXuMkpIS+f1+jR49Wtdcc40WLVqkwsJCGWP06quvav78+ZKk9PR07dy5U5JUWFio9PR0SdL8+fO1d+9eGWO6dHIAAABAV3V7DvI3vvENxcXF6cc//nGbK8j79u3TpEmTNGfOHB06dEiSVFtbq7i4uOA+sbGxqq2tVVNTk4YOHaqwsLA24x9/TlhYmIYMGaKmpqaLZsnNzVUgEFAgEFBDQ0N3TwkAAADofkF+7LHHVF1drSVLlmjjxo2SpClTpujo0aP6wx/+oC9/+cu65557rlTOdmVlZam0tFSlpaUaPnx4r7wmAAAA+qbLXsViyZIl2rFjh6TzUy8GDRokSZo7d64+/PBDNTY2KiYmRtXV1cHn1NTUKCYmRpGRkWppadHZs2fbjEtq85yzZ8/q5MmTioyMvNy4AAAAQLu6VZArKiqCXxcWFmr8+PGSpOPHjwfnCZeUlOjcuXOKjIzU1KlTVVFRocrKSp05c0YFBQVKS0uTz+fTzJkztX37dklSfn6+5s2bJ0lKS0tTfn6+JGn79u268847O5zrDAAAAFyusI52WLx4sYqLi9XY2KjY2FhlZ2dr165devvtt9WvXz+NGjVKmzdvlnS+yP7gBz9QWFiYPvGJT6igoEA+n09hYWHauHGjZs+erdbWVmVmZmrChAmSpA0bNmjRokX65je/qcmTJ2vZsmWSpGXLlum+++6T3+9XRESECgoKevDHAAB925YtUkaG6xQA4A0+08eWhggEAiotLXUdAwBCyrp15x8AgP/vUr2RT9IDAAAALBRkAAAAwEJBBgAPioqSfL7OP7Kzu7Z/VJTrMwQAdyjIAOBB9fXePj4AhDIKMgAAAGChIAMAAAAWCjIAILRt2eI6AYCrDAUZABDaqqpcJwBwlaEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAMAAAAWCjIAAABgoSADAAAAFgoyAAAAYOlUQc7MzNSIESOUmJgYHHvkkUeUlJSk5ORk3XXXXTp27JgkyRijlStXyu/3KykpSQcPHgw+Jz8/XwkJCUpISFB+fn5w/MCBA5o4caL8fr9WrlwpY4wk6cSJE0pNTVVCQoJSU1PV3Nx8RU4aAAAAuJROFeSMjAwVFRW1GXvooYf05ptvqqysTHfffbe+9a1vSZJ2796tiooKVVRUKDc3V8uXL5d0vuxmZ2fr9ddfV0lJibKzs4OFd/ny5XruueeCz/votXJycjRr1ixVVFRo1qxZysnJuWInDgAAAFxMpwryjBkzFBER0WZs8ODBwa/ff/99+Xw+SVJhYaGWLl0qn8+nlJQUtbS0qK6uTi+//LJSU1MVERGh8PBwpaamqqioSHV1dTp16pRSUlLk8/m0dOlS7dy5M3is9PR0SVJ6enpwHAAAAOgplzUH+Rvf+Ibi4uL04x//OHgFuba2VnFxccF9YmNjVVtb2+54bGzsBeOSVF9fr+joaElSVFSU6uvrL5ojNzdXgUBAgUBADQ0Nl3NKAICeFhUl+Xydf2Rnd23/qCjXZxi0ZYvrBAC647IK8mOPPabq6motWbJEGzduvFKZLsrn8wWvUn9cVlaWSktLVVpaquHDh/doDgDAZbrExQ7PHL8LqqpcJwDQHVdkFYslS5Zox44dkqSYmBhVV1cHt9XU1CgmJqbd8ZqamgvGJWnkyJGqq6uTJNXV1WnEiBFXIi4AAABwSd0uyBUVFcGvCwsLNX78eElSWlqatm7dKmOM9u/fryFDhig6OlqzZ8/Wnj171NzcrObmZu3Zs0ezZ89WdHS0Bg8erP3798sYo61bt2revHnBY3202kV+fn5wHAAAAOgpYZ3ZafHixSouLlZjY6NiY2OVnZ2tXbt26e2331a/fv00atQobd68WZI0d+5c7dq1S36/X9dee62ef/55SVJERIQeeeQRTZ06VZL06KOPBm/827RpkzIyMvTBBx9ozpw5mjNnjiRpzZo1WrBggfLy8jRq1Ci9+OKLV/wHAAAAANh85qNFh/uIQCCg0tJS1zEAoEdd4paMK6rH/uvg6fBds27d+QeA0HSp3sgn6QEAAAAWCjIAAABgoSADAAAAFgoyAAAAYKEgAwAAABYKMgAAAGChIAOAB40c6e3jA0AooyADgAcdP35+qd/OPtau7dr+x4+7PkMAcIeCDAAAAFgoyAAAAICFggwAAABYKMgAcBWIj3edAAC8g4IMAFeBjAzXCQDAOyjIAAAAgIWCDAAAAFgoyAAAoE/ZssV1AngdBRkAAPQpVVWuE8DrKMgAAACAhYIMAAAAWCjIAIDeNXKkt48PoM+jIAMAetfx45IxnX+sXdu1/Y8fd32GADyOggwAAABYKMgAAACAhYIMAEAnRUVJPl/nH9nZXds/Ksr1GQKQKMgAAHRafb23jw+gcyjIAAAAgIWCDAAAAFgoyACA0BYf7zrB1WnLFtcJAGcoyACA0JaR4TrB1amqynUCwBkKMgAAAGChIAMAAAAWCjIAAABg6bAgZ2ZmasSIEUpMTAyOPfTQQxo/frySkpL02c9+Vi0tLZKkqqoqfeITn1BycrKSk5P1hS98IficAwcOaOLEifL7/Vq5cqWMMZKkEydOKDU1VQkJCUpNTVVzc7MkyRijlStXyu/3KykpSQcPHryS5w0AAABcVIcFOSMjQ0VFRW3GUlNT9dZbb+nNN9/U2LFjtX79+uC2MWPGqKysTGVlZdq8eXNwfPny5XruuedUUVGhioqK4DFzcnI0a9YsVVRUaNasWcrJyZEk7d69O7hvbm6uli9ffkVOGAAAAGhPhwV5xowZioiIaDN21113KSwsTJKUkpKimpqado9RV1enU6dOKSUlRT6fT0uXLtXOnTslSYWFhUpPT5ckpaentxlfunSpfD6fUlJS1NLSorq6uq6eHwAAANAllz0H+Uc/+pHmzJkT/L6yslKTJ0/W7bffrtdee02SVFtbq9jY2OA+sbGxqq2tlSTV19crOjpakhQVFaX6//uczdraWsXFxV30OR+Xm5urQCCgQCCghoaGyz0lAAAAZ1iC2r2wy3nyY489prCwMC1ZskSSFB0drffee0+RkZE6cOCA7rnnHh06dKjTx/P5fPL5fF3OkZWVpaysLElSIBDo8vMBAABCBUtQu9ftgrxlyxb98pe/1N69e4OlduDAgRo4cKAk6aabbtKYMWP0zjvvKCYmps00jJqaGsXExEiSRo4cqbq6OkVHR6uurk4jRoyQJMXExKi6uvqizwEAAAB6SremWBQVFemJJ57QSy+9pGuvvTY43tDQoNbWVknSu+++q4qKCo0ePVrR0dEaPHiw9u/fL2OMtm7dqnnz5kmS0tLSlJ+fL0nKz89vM75161YZY7R//34NGTIkOBUDAAAA6CkdXkFevHixiouL1djYqNjYWGVnZ2v9+vU6ffq0UlNTJZ2/UW/z5s36zW9+o0cffVQDBgxQv379tHnz5uANfps2bVJGRoY++OADzZkzJzhvec2aNVqwYIHy8vI0atQovfjii5KkuXPnateuXfL7/br22mv1/PPP99TPAACAThk5Uvq/W2V67PgA3OuwIG/btu2CsWXLll1033vvvVf33nvvRbcFAgG99dZbF4xHRkZq7969F4z7fD4988wzHcUDAKDXHD/etf3XrTv/AOAtfJIeAAAAYKEgAwAAABYKMgAAV4uoKMnn69wjO7vz+/p8548N9BEUZAAArhY9eYdhTx4b6GUUZAAAAMBCQQYAAAAsFGQAABDymD6N3kRBBgAAIY/p0+hNFGQAAADAQkEGAAAALBRkAAAAwEJBBgAA6EFducGwqzcZcoNhz6AgAwAA9CBuMPQeCjIAAABgoSADAAAAFgoyAAA9JD7e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P/ywHn74Yd1+++1KSUnRwYMHXcdr1+9+9zvdcMMNmjBhgl5//XWlpqZq6tSpiouL0759+1zHa9fPfvazNo8dO3YoKysr+L3XePV/TA4fPqwdO3aovLzcdZQOtbS0uI7Qswx6zY4dOy762L59uxk2bJjreO361Kc+ZX7wgx+YN954w3zpS18yN998s2lsbDTGGJOcnOw4Xftuu+0289JLL5kXXnjBXHfddWbbtm3m3Llz5qWXXjJ33nmn63jtSkxMDH598803m8rKSmOMMQ0NDSYpKclRqs4ZO3as+f73v28++clPmn/6p38yK1euNPv27XMdq1MmTZpkysvLze9//3sTERERzF1eXm4mT57sOF3HZs6caX79618bY87/3vnqV79q/vrXv5pvfOMb5oEHHnAbrgOTJk0y+/fvv2B83759If+enzp1qnnzzTfN73//exMZGWlee+01Y4wxBw4cMJ/85Ccdp2tfWFiY+cxnPmPuv/9+k5GRYTIyMsygQYNMRkaGuf/++13Ha9e3v/3t4NeHDh0yCQkJJj4+3owaNeqi76VQcscdd5iGhgZjjDFbt241CQkJZtmyZSYxMdH8x3/8h+N07evfv7+ZNWuW+eEPf2iam5tdx7niKMi9KCwszKSnpwd/+diPQYMGuY7XrkmTJrX5/j//8z/NjTfeaA4fPhzyhcEu8GPGjGmzLdSz33jjjebkyZPGGGNuueUW09ra2mZbKLN/tkePHjUbNmwwkydPNtdff73513/9V4fJOmZnj42NbbPt438XQtHHi+SUKVOCX48bN66343SJ3++/5LaP//0NNfbvmvHjx7fZFuq/a0pKSsydd95pNm3aFByLj493mKjz7J/t3Llzza5du4wxxrz++uvm5ptvdhWrUyZMmBD8OhAIBC88vf/++2bixImuYnVKYmKi+cUvfmE+97nPmYiICJOWlma2bdtm/va3v7mOdkWEub6CfTVJSkrS17/+dSUmJl6w7Ve/+pWDRJ334Ycf6u9//7v+4R/+QdL5f/qPiorS7Nmz9f777ztO177W1tbg1w8++GCbbWfOnOntOF2ydu1azZw5UytWrNAtt9yif/7nf1ZaWpp+/etf69Of/rTreO0y1rSK6667LvjP5X/+85/1k5/8xGGyjg0dOlTPPvusTp06pfDwcH3ve9/TggUL9Ktf/UqDBg1yHa9D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"/>

## 종가 데이터만 추출 후 정규화(Normalization)



```python
# 종가(Close)만 추출
close = data_['Close']

# 추출한 종가(Close) 데이터를 lineplot
# .plot(color='tomato')
plt.plot(data_['Close'].values, color='m')
plt.xticks(np.arange(len(data_['Close'])), data_['Close'].index.strftime('%Y-%m-%d'), rotation=45)
plt.show()
```

<img src="data:image/png;base64,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"/>

기준(`startdate:enddate`)이 되는 종가 데이터에 대하여 정규화(Normalization)을 진행합니다.



정규화를 하여 기간에 대한 같은 가격 데이터의 스케일을 가질 수 있도록 해야하는데, 이렇게 정규화를 해야 패턴 찾을 때 주가의 **오르내림** 패턴을 인지할 수 있습니다.



정규화를 적용한 뒤 기준 값이라는 의미의 `base` 변수에 대입합니다.



```python
# 종가(Close)에 대한 정규화
base = (close - close.min()) / (close.max() - close.min())
base
```

<pre>
Date
2021-09-01    0.932432
2021-09-02    0.959459
2021-09-03    1.000000
2021-09-06    0.972973
2021-09-07    0.932432
2021-09-08    0.513514
2021-09-09    0.243243
2021-09-10    0.283784
2021-09-13    0.135135
2021-09-14    0.121622
2021-09-15    0.081081
2021-09-16    0.054054
2021-09-17    0.000000
Name: Close, dtype: float64
</pre>
## 윈도우 범위(window_size) 만큼의 유사 패턴 찾기 (cosine 유사도)


먼저, 윈도우 사이즈는 `base` 변수에 담긴 기준 데이터의 개수로 지정합니다.



```python
# 윈도우 사이즈
window_size = len(base)
```

예측 기간은 향후 **N일간**의 주간을 예측하도록 합니다.



사실 예측이라기 보다는 과거 데이터에서 찾은 **코사인 유사도가 가장 높았던 패턴의 향후 5일 주가 추이**를 그대로 가져와서 보여주는 것입니다.



```python
# 예측 기간
next_date = 5
```

반목문 횟수를 계산합니다.



**처음부터 윈도우 사이즈만큼 이동하면서 끝까지 탐색**합니다.



모든 탐색에 대해서는 **코사인 유사도를 계산하여 index로 저장**합니다.



```python
# 검색 횟수
moving_cnt = len(data) - window_size - next_date - 1
```

코사인 유사도를 구하는 공식을 활용하여 **다음과 같이 함수로 구현**합니다.



혹은 **scipy**에서 제공하는 `cosine`을 사용하여 구할 수도 있습니다.



[scipy.spatial.distance.cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)



```python
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
```

다음은 데이터의 처음부터 끝까지 `window_size`만큼 순회하면서 `base`와의 코사인 유사도를 모두 계산하여 `sim_list` 리스트에 추가합니다.



나중에 `sim_list`에서 코사인 유사도를 기준으로 내림차순 정렬을 하여 유사도가 높은 인덱스를 확인하겠습니다



```python
# 코사인 유사도를 계산하여 저장해줄 리스트를 생성합니다
sim_list = []

for i in range(moving_cnt):
    # i 번째 인덱스 부터 i+window_size 만큼의 범위를 가져와 target 변수에 대입합니다
    target = data['Close'].iloc[i:i+window_size]
    
    # base와 마찬가지로 정규화를 적용하여 스케일을 맞춰 줍니다
    target = (target - target.min()) / (target.max() - target.min())
    
    # 코사인 유사도를 계산합니다
    cos_similarity = cosine_similarity(base, target)
    
    # 계산된 코사인 유사도를 추가합니다
    sim_list.append(cos_similarity)
```

## 유사도 계산 결과 확인


계산된 코사인 유사도는 `sim_list`에 저장되어 있으며, 이를 내림차순 정렬하여 상위 20개를 출력하면 다음과 같습니다



```python
pd.Series(sim_list).sort_values(ascending=False).head(20)
```

<pre>
5384    1.000000
3276    0.994382
1801    0.989912
1026    0.989041
3196    0.986168
1989    0.985774
4011    0.985743
4538    0.985353
3210    0.985147
4659    0.985074
3528    0.984875
3211    0.984738
592     0.984496
995     0.984276
2952    0.984207
3078    0.984094
2065    0.983622
996     0.982743
3557    0.982710
2154    0.982579
dtype: float64
</pre>
**5384** 인덱스는 자기 자신 인덱스이므로 코사인 유사도가 1이 나왔습니다.



그다음으로 유사도가 높게 나온 **3276** 인덱스의 주가와 `base` 주가를 **동시에 시각화하여 향후 주가를 예측**해 봅니다.



```python
# 높은 유사도를 기록한 인덱스 대입
idx=3276

# target 변수에 종가 데이터의 [기준 인덱스] 부터 [기준 인덱스 + window_size + 예측(5일)] 데이터를 추출합니다
target = data['Close'].iloc[idx:idx+window_size+5]

# 정규화를 적용합니다
target = (target - target.min()) / (target.max() - target.min())

# 결과를 시각화합니다
plt.plot(base.values, label='base', color='grey')
plt.plot(target.values, label='target', color='orangered')
plt.xticks(np.arange(len(target)), list(target.index.strftime('%Y-%m-%d')), rotation=45)
plt.axvline(x=len(base)-1, c='grey', linestyle='--')
plt.axvspan(len(base.values)-1, len(target.values)-1, facecolor='ivory', alpha=0.7)
plt.legend()
plt.show()
```

<img src="data:image/png;base64,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"/>

**결과 정리**



1. **2013년 2월 6일**부터 **2013년 2월 23일**까지의 데이터가 **가장 유사도가 높게** 나왔음을 확인할 수 있습니다.

2. 코사인 유사도는 **0.994382**를 기록하였습니다.

3. 과거 패턴에서는 향후 주가가 횡보 혹은 추가 하락이 있었습니다.

