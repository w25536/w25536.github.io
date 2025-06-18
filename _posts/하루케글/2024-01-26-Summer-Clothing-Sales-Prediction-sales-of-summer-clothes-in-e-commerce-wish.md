---
title: "Summer Clothing Sales Prediction sales of summer clothes in e-commerce wish"
date: 2024-01-26
last_modified_at: 2024-01-26
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Summer Clothing Sales Prediction sales of summer clothes in e-commerce wish 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jmmvutu/summer-products-and-sales-in-ecommerce-wish")

print("Path to dataset files:", path)
```

    /Users/jeongho/Desktop/w25536-kaggle/kaggle/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
      warnings.warn(


    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.4)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/jmmvutu/summer-products-and-sales-in-ecommerce-wish/versions/5



```python
path
```




    '/Users/jeongho/.cache/kagglehub/datasets/jmmvutu/summer-products-and-sales-in-ecommerce-wish/versions/5'




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import os

df = pd.read_csv(
    os.path.join(path, "summer-products-with-rating-and-performance_2020-08.csv")
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
      <th>title</th>
      <th>title_orig</th>
      <th>price</th>
      <th>retail_price</th>
      <th>currency_buyer</th>
      <th>units_sold</th>
      <th>uses_ad_boosts</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>rating_five_count</th>
      <th>...</th>
      <th>merchant_rating_count</th>
      <th>merchant_rating</th>
      <th>merchant_id</th>
      <th>merchant_has_profile_picture</th>
      <th>merchant_profile_picture</th>
      <th>product_url</th>
      <th>product_picture</th>
      <th>product_id</th>
      <th>theme</th>
      <th>crawl_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020 Summer Vintage Flamingo Print  Pajamas Se...</td>
      <td>2020 Summer Vintage Flamingo Print  Pajamas Se...</td>
      <td>16.00</td>
      <td>14</td>
      <td>EUR</td>
      <td>100</td>
      <td>0</td>
      <td>3.76</td>
      <td>54</td>
      <td>26.0</td>
      <td>...</td>
      <td>568</td>
      <td>4.128521</td>
      <td>595097d6a26f6e070cb878d1</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5e9ae51d43d6a96e303acdb0</td>
      <td>https://contestimg.wish.com/api/webimage/5e9ae...</td>
      <td>5e9ae51d43d6a96e303acdb0</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SSHOUSE Summer Casual Sleeveless Soirée Party ...</td>
      <td>Women's Casual Summer Sleeveless Sexy Mini Dress</td>
      <td>8.00</td>
      <td>22</td>
      <td>EUR</td>
      <td>20000</td>
      <td>1</td>
      <td>3.45</td>
      <td>6135</td>
      <td>2269.0</td>
      <td>...</td>
      <td>17752</td>
      <td>3.899673</td>
      <td>56458aa03a698c35c9050988</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/58940d436a0d3d5da4e95a38</td>
      <td>https://contestimg.wish.com/api/webimage/58940...</td>
      <td>58940d436a0d3d5da4e95a38</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020 Nouvelle Arrivée Femmes Printemps et Été ...</td>
      <td>2020 New Arrival Women Spring and Summer Beach...</td>
      <td>8.00</td>
      <td>43</td>
      <td>EUR</td>
      <td>100</td>
      <td>0</td>
      <td>3.57</td>
      <td>14</td>
      <td>5.0</td>
      <td>...</td>
      <td>295</td>
      <td>3.989831</td>
      <td>5d464a1ffdf7bc44ee933c65</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5ea10e2c617580260d55310a</td>
      <td>https://contestimg.wish.com/api/webimage/5ea10...</td>
      <td>5ea10e2c617580260d55310a</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hot Summer Cool T-shirt pour les femmes Mode T...</td>
      <td>Hot Summer Cool T Shirt for Women Fashion Tops...</td>
      <td>8.00</td>
      <td>8</td>
      <td>EUR</td>
      <td>5000</td>
      <td>1</td>
      <td>4.03</td>
      <td>579</td>
      <td>295.0</td>
      <td>...</td>
      <td>23832</td>
      <td>4.020435</td>
      <td>58cfdefdacb37b556efdff7c</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5cedf17ad1d44c52c59e4aca</td>
      <td>https://contestimg.wish.com/api/webimage/5cedf...</td>
      <td>5cedf17ad1d44c52c59e4aca</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Femmes Shorts d'été à lacets taille élastique ...</td>
      <td>Women Summer Shorts Lace Up Elastic Waistband ...</td>
      <td>2.72</td>
      <td>3</td>
      <td>EUR</td>
      <td>100</td>
      <td>1</td>
      <td>3.10</td>
      <td>20</td>
      <td>6.0</td>
      <td>...</td>
      <td>14482</td>
      <td>4.001588</td>
      <td>5ab3b592c3911a095ad5dadb</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5ebf5819ebac372b070b0e70</td>
      <td>https://contestimg.wish.com/api/webimage/5ebf5...</td>
      <td>5ebf5819ebac372b070b0e70</td>
      <td>summer</td>
      <td>2020-08</td>
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
      <th>1568</th>
      <td>Nouvelle Mode Femmes Bohême Pissenlit Imprimer...</td>
      <td>New Fashion Women Bohemia Dandelion Print Tee ...</td>
      <td>6.00</td>
      <td>9</td>
      <td>EUR</td>
      <td>10000</td>
      <td>1</td>
      <td>4.08</td>
      <td>1367</td>
      <td>722.0</td>
      <td>...</td>
      <td>5316</td>
      <td>4.224605</td>
      <td>5b507899ab577736508a0782</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5d5fadc99febd9356cbc52ee</td>
      <td>https://contestimg.wish.com/api/webimage/5d5fa...</td>
      <td>5d5fadc99febd9356cbc52ee</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>10 couleurs femmes shorts d'été lacent ceintur...</td>
      <td>10 Color Women Summer Shorts Lace Up Elastic W...</td>
      <td>2.00</td>
      <td>56</td>
      <td>EUR</td>
      <td>100</td>
      <td>1</td>
      <td>3.07</td>
      <td>28</td>
      <td>11.0</td>
      <td>...</td>
      <td>4435</td>
      <td>3.696054</td>
      <td>54d83b6b6b8a771e478558de</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5eccd22b4497b86fd48f16b4</td>
      <td>https://contestimg.wish.com/api/webimage/5eccd...</td>
      <td>5eccd22b4497b86fd48f16b4</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>Nouveautés Hommes Siwmwear Beach-Shorts Hommes...</td>
      <td>New Men Siwmwear Beach-Shorts Men Summer Quick...</td>
      <td>5.00</td>
      <td>19</td>
      <td>EUR</td>
      <td>100</td>
      <td>0</td>
      <td>3.71</td>
      <td>59</td>
      <td>24.0</td>
      <td>...</td>
      <td>210</td>
      <td>3.961905</td>
      <td>5b42da1bf64320209fc8da69</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5e74be96034d613d42b52dfe</td>
      <td>https://contestimg.wish.com/api/webimage/5e74b...</td>
      <td>5e74be96034d613d42b52dfe</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>Mode femmes d'été sans manches robes col en V ...</td>
      <td>Fashion Women Summer Sleeveless Dresses V Neck...</td>
      <td>13.00</td>
      <td>11</td>
      <td>EUR</td>
      <td>100</td>
      <td>0</td>
      <td>2.50</td>
      <td>2</td>
      <td>0.0</td>
      <td>...</td>
      <td>31</td>
      <td>3.774194</td>
      <td>5d56b32c40defd78043d5af9</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5eda07ab0e295c2097c36590</td>
      <td>https://contestimg.wish.com/api/webimage/5eda0...</td>
      <td>5eda07ab0e295c2097c36590</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>Pantalon de yoga pour femmes à la mode Slim Fi...</td>
      <td>Fashion Women Yoga Pants Slim Fit Fitness Runn...</td>
      <td>7.00</td>
      <td>6</td>
      <td>EUR</td>
      <td>100</td>
      <td>1</td>
      <td>4.07</td>
      <td>14</td>
      <td>8.0</td>
      <td>...</td>
      <td>7023</td>
      <td>4.235939</td>
      <td>5a409cf87b584e7951b2e25f</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5e857321f53c3d2d8f25e7ed</td>
      <td>https://contestimg.wish.com/api/webimage/5e857...</td>
      <td>5e857321f53c3d2d8f25e7ed</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
  </tbody>
</table>
<p>1573 rows × 43 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1573 entries, 0 to 1572
    Data columns (total 43 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   title                         1573 non-null   object 
     1   title_orig                    1573 non-null   object 
     2   price                         1573 non-null   float64
     3   retail_price                  1573 non-null   int64  
     4   currency_buyer                1573 non-null   object 
     5   units_sold                    1573 non-null   int64  
     6   uses_ad_boosts                1573 non-null   int64  
     7   rating                        1573 non-null   float64
     8   rating_count                  1573 non-null   int64  
     9   rating_five_count             1528 non-null   float64
     10  rating_four_count             1528 non-null   float64
     11  rating_three_count            1528 non-null   float64
     12  rating_two_count              1528 non-null   float64
     13  rating_one_count              1528 non-null   float64
     14  badges_count                  1573 non-null   int64  
     15  badge_local_product           1573 non-null   int64  
     16  badge_product_quality         1573 non-null   int64  
     17  badge_fast_shipping           1573 non-null   int64  
     18  tags                          1573 non-null   object 
     19  product_color                 1532 non-null   object 
     20  product_variation_size_id     1559 non-null   object 
     21  product_variation_inventory   1573 non-null   int64  
     22  shipping_option_name          1573 non-null   object 
     23  shipping_option_price         1573 non-null   int64  
     24  shipping_is_express           1573 non-null   int64  
     25  countries_shipped_to          1573 non-null   int64  
     26  inventory_total               1573 non-null   int64  
     27  has_urgency_banner            473 non-null    float64
     28  urgency_text                  473 non-null    object 
     29  origin_country                1556 non-null   object 
     30  merchant_title                1573 non-null   object 
     31  merchant_name                 1569 non-null   object 
     32  merchant_info_subtitle        1572 non-null   object 
     33  merchant_rating_count         1573 non-null   int64  
     34  merchant_rating               1573 non-null   float64
     35  merchant_id                   1573 non-null   object 
     36  merchant_has_profile_picture  1573 non-null   int64  
     37  merchant_profile_picture      226 non-null    object 
     38  product_url                   1573 non-null   object 
     39  product_picture               1573 non-null   object 
     40  product_id                    1573 non-null   object 
     41  theme                         1573 non-null   object 
     42  crawl_month                   1573 non-null   object 
    dtypes: float64(9), int64(15), object(19)
    memory usage: 528.6+ KB



```python
df.isnull().sum()
```




    title                              0
    title_orig                         0
    price                              0
    retail_price                       0
    currency_buyer                     0
    units_sold                         0
    uses_ad_boosts                     0
    rating                             0
    rating_count                       0
    rating_five_count                 45
    rating_four_count                 45
    rating_three_count                45
    rating_two_count                  45
    rating_one_count                  45
    badges_count                       0
    badge_local_product                0
    badge_product_quality              0
    badge_fast_shipping                0
    tags                               0
    product_color                     41
    product_variation_size_id         14
    product_variation_inventory        0
    shipping_option_name               0
    shipping_option_price              0
    shipping_is_express                0
    countries_shipped_to               0
    inventory_total                    0
    has_urgency_banner              1100
    urgency_text                    1100
    origin_country                    17
    merchant_title                     0
    merchant_name                      4
    merchant_info_subtitle             1
    merchant_rating_count              0
    merchant_rating                    0
    merchant_id                        0
    merchant_has_profile_picture       0
    merchant_profile_picture        1347
    product_url                        0
    product_picture                    0
    product_id                         0
    theme                              0
    crawl_month                        0
    dtype: int64




```python
print(df.columns)
```

    Index(['title', 'title_orig', 'price', 'retail_price', 'currency_buyer',
           'units_sold', 'uses_ad_boosts', 'rating', 'rating_count',
           'rating_five_count', 'rating_four_count', 'rating_three_count',
           'rating_two_count', 'rating_one_count', 'badges_count',
           'badge_local_product', 'badge_product_quality', 'badge_fast_shipping',
           'tags', 'product_color', 'product_variation_size_id',
           'product_variation_inventory', 'shipping_option_name',
           'shipping_option_price', 'shipping_is_express', 'countries_shipped_to',
           'inventory_total', 'has_urgency_banner', 'urgency_text',
           'origin_country', 'merchant_title', 'merchant_name',
           'merchant_info_subtitle', 'merchant_rating_count', 'merchant_rating',
           'merchant_id', 'merchant_has_profile_picture',
           'merchant_profile_picture', 'product_url', 'product_picture',
           'product_id', 'theme', 'crawl_month'],
          dtype='object')



```python
columns_to_drop = [
    "title",
    "title_orig",
    "currency_buyer",
    "shipping_option_name",
    "urgency_text",
    "merchant_title",
    "merchant_name",
    "merchant_info_subtitle",
    "merchant_id",
    "merchant_profile_picture",
    "product_url",
    "product_picture",
    "product_id",
    "tags",
    "has_urgency_banner",
    "theme",
    "crawl_month",
    "origin_country",
]
```


```python
df.drop(columns_to_drop, axis=1, inplace=True)
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
      <th>price</th>
      <th>retail_price</th>
      <th>units_sold</th>
      <th>uses_ad_boosts</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>rating_five_count</th>
      <th>rating_four_count</th>
      <th>rating_three_count</th>
      <th>rating_two_count</th>
      <th>...</th>
      <th>product_color</th>
      <th>product_variation_size_id</th>
      <th>product_variation_inventory</th>
      <th>shipping_option_price</th>
      <th>shipping_is_express</th>
      <th>countries_shipped_to</th>
      <th>inventory_total</th>
      <th>merchant_rating_count</th>
      <th>merchant_rating</th>
      <th>merchant_has_profile_picture</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.00</td>
      <td>14</td>
      <td>100</td>
      <td>0</td>
      <td>3.76</td>
      <td>54</td>
      <td>26.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>white</td>
      <td>M</td>
      <td>50</td>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>50</td>
      <td>568</td>
      <td>4.128521</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.00</td>
      <td>22</td>
      <td>20000</td>
      <td>1</td>
      <td>3.45</td>
      <td>6135</td>
      <td>2269.0</td>
      <td>1027.0</td>
      <td>1118.0</td>
      <td>644.0</td>
      <td>...</td>
      <td>green</td>
      <td>XS</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>17752</td>
      <td>3.899673</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.00</td>
      <td>43</td>
      <td>100</td>
      <td>0</td>
      <td>3.57</td>
      <td>14</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>leopardprint</td>
      <td>XS</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>36</td>
      <td>50</td>
      <td>295</td>
      <td>3.989831</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.00</td>
      <td>8</td>
      <td>5000</td>
      <td>1</td>
      <td>4.03</td>
      <td>579</td>
      <td>295.0</td>
      <td>119.0</td>
      <td>87.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>black</td>
      <td>M</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>23832</td>
      <td>4.020435</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.72</td>
      <td>3</td>
      <td>100</td>
      <td>1</td>
      <td>3.10</td>
      <td>20</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>yellow</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35</td>
      <td>50</td>
      <td>14482</td>
      <td>4.001588</td>
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
      <th>1568</th>
      <td>6.00</td>
      <td>9</td>
      <td>10000</td>
      <td>1</td>
      <td>4.08</td>
      <td>1367</td>
      <td>722.0</td>
      <td>293.0</td>
      <td>185.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>navyblue</td>
      <td>S</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>5316</td>
      <td>4.224605</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>2.00</td>
      <td>56</td>
      <td>100</td>
      <td>1</td>
      <td>3.07</td>
      <td>28</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>lightblue</td>
      <td>S</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>50</td>
      <td>4435</td>
      <td>3.696054</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>5.00</td>
      <td>19</td>
      <td>100</td>
      <td>0</td>
      <td>3.71</td>
      <td>59</td>
      <td>24.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>white</td>
      <td>SIZE S</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>50</td>
      <td>210</td>
      <td>3.961905</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>13.00</td>
      <td>11</td>
      <td>100</td>
      <td>0</td>
      <td>2.50</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>white</td>
      <td>Size S.</td>
      <td>36</td>
      <td>3</td>
      <td>0</td>
      <td>29</td>
      <td>50</td>
      <td>31</td>
      <td>3.774194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>7.00</td>
      <td>6</td>
      <td>100</td>
      <td>1</td>
      <td>4.07</td>
      <td>14</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>red</td>
      <td>S</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>7023</td>
      <td>4.235939</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1573 rows × 25 columns</p>
</div>




```python
df.isnull().sum()
```




    price                            0
    retail_price                     0
    units_sold                       0
    uses_ad_boosts                   0
    rating                           0
    rating_count                     0
    rating_five_count               45
    rating_four_count               45
    rating_three_count              45
    rating_two_count                45
    rating_one_count                45
    badges_count                     0
    badge_local_product              0
    badge_product_quality            0
    badge_fast_shipping              0
    product_color                   41
    product_variation_size_id       14
    product_variation_inventory      0
    shipping_option_price            0
    shipping_is_express              0
    countries_shipped_to             0
    inventory_total                  0
    merchant_rating_count            0
    merchant_rating                  0
    merchant_has_profile_picture     0
    dtype: int64




```python
## Encoding


def ordinal_encode(data, column, orderding):
    return data[column].apply(lambda x: orderding.index(x) if x in orderding else None)
```


```python
df["product_variation_size_id"].unique()
```




    array(['M', 'XS', 'S', 'Size-XS', 'M.', 'XXS', 'L', 'XXL', nan, 'S.', 's',
           'choose a size', 'XS.', '32/L', 'Suit-S', 'XXXXXL', 'EU 35', '4',
           'Size S.', '1m by 3m', '3XL', 'Size S', 'XL', 'Women Size 36',
           'US 6.5 (EU 37)', 'XXXS', 'SIZE XS', '26(Waist 72cm 28inch)',
           'Size XXS', '29', '1pc', '100 cm', 'One Size', 'SIZE-4XL', '1',
           'S/M(child)', '2pcs', 'XXXL', 'S..', '30 cm', '5XL', '33',
           'Size M', '100 x 100cm(39.3 x 39.3inch)', '100pcs', '2XL', '4XL',
           'SizeL', 'SIZE XXS', 'XXXXL', 'Base & Top & Matte Top Coat',
           'size S', '35', '34', 'SIZE-XXS', 'S(bust 88cm)',
           'S (waist58-62cm)', 'S(Pink & Black)', '20pcs', 'US-S',
           'Size -XXS', 'X   L', 'White', '25', 'Size-S', 'Round',
           'Pack of 1', '1 pc.', 'S Diameter 30cm', '6XL',
           'AU plug Low quality', '5PAIRS', '25-S', 'Size/S', 'S Pink',
           'Size-5XL', 'daughter 24M', '2', 'Baby Float Boat', '10 ml', '60',
           'Size-L', 'US5.5-EU35', '10pcs', '17', 'Size-XXS', 'Women Size 37',
           '3 layered anklet', '4-5 Years', 'Size4XL', 'first  generation',
           '80 X 200 CM', 'EU39(US8)', 'L.', 'Base Coat', '36', '04-3XL',
           'pants-S', 'Floating Chair for Kid', '20PCS-10PAIRS', 'B',
           'Size--S', '5', '1 PC - XL', 'H01', '40 cm', 'SIZE S'],
          dtype=object)




```python
df["product_variation_size_id"].value_counts()
```




    product_variation_size_id
    S                      641
    XS                     356
    M                      200
    XXS                    100
    L                       49
                          ... 
    6XL                      1
    AU plug Low quality      1
    XXXL                     1
    25-S                     1
    SIZE S                   1
    Name: count, Length: 106, dtype: int64




```python
size_ordering = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
```


```python
df["product_variation_size_id"] = ordinal_encode(
    df, "product_variation_size_id", size_ordering
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
      <th>price</th>
      <th>retail_price</th>
      <th>units_sold</th>
      <th>uses_ad_boosts</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>rating_five_count</th>
      <th>rating_four_count</th>
      <th>rating_three_count</th>
      <th>rating_two_count</th>
      <th>...</th>
      <th>product_color</th>
      <th>product_variation_size_id</th>
      <th>product_variation_inventory</th>
      <th>shipping_option_price</th>
      <th>shipping_is_express</th>
      <th>countries_shipped_to</th>
      <th>inventory_total</th>
      <th>merchant_rating_count</th>
      <th>merchant_rating</th>
      <th>merchant_has_profile_picture</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.00</td>
      <td>14</td>
      <td>100</td>
      <td>0</td>
      <td>3.76</td>
      <td>54</td>
      <td>26.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>white</td>
      <td>3.0</td>
      <td>50</td>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>50</td>
      <td>568</td>
      <td>4.128521</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.00</td>
      <td>22</td>
      <td>20000</td>
      <td>1</td>
      <td>3.45</td>
      <td>6135</td>
      <td>2269.0</td>
      <td>1027.0</td>
      <td>1118.0</td>
      <td>644.0</td>
      <td>...</td>
      <td>green</td>
      <td>1.0</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>17752</td>
      <td>3.899673</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.00</td>
      <td>43</td>
      <td>100</td>
      <td>0</td>
      <td>3.57</td>
      <td>14</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>leopardprint</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>36</td>
      <td>50</td>
      <td>295</td>
      <td>3.989831</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.00</td>
      <td>8</td>
      <td>5000</td>
      <td>1</td>
      <td>4.03</td>
      <td>579</td>
      <td>295.0</td>
      <td>119.0</td>
      <td>87.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>black</td>
      <td>3.0</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>23832</td>
      <td>4.020435</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.72</td>
      <td>3</td>
      <td>100</td>
      <td>1</td>
      <td>3.10</td>
      <td>20</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>yellow</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35</td>
      <td>50</td>
      <td>14482</td>
      <td>4.001588</td>
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
      <th>1568</th>
      <td>6.00</td>
      <td>9</td>
      <td>10000</td>
      <td>1</td>
      <td>4.08</td>
      <td>1367</td>
      <td>722.0</td>
      <td>293.0</td>
      <td>185.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>navyblue</td>
      <td>2.0</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>5316</td>
      <td>4.224605</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>2.00</td>
      <td>56</td>
      <td>100</td>
      <td>1</td>
      <td>3.07</td>
      <td>28</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>lightblue</td>
      <td>2.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>50</td>
      <td>4435</td>
      <td>3.696054</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>5.00</td>
      <td>19</td>
      <td>100</td>
      <td>0</td>
      <td>3.71</td>
      <td>59</td>
      <td>24.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>white</td>
      <td>NaN</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>50</td>
      <td>210</td>
      <td>3.961905</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>13.00</td>
      <td>11</td>
      <td>100</td>
      <td>0</td>
      <td>2.50</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>white</td>
      <td>NaN</td>
      <td>36</td>
      <td>3</td>
      <td>0</td>
      <td>29</td>
      <td>50</td>
      <td>31</td>
      <td>3.774194</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>7.00</td>
      <td>6</td>
      <td>100</td>
      <td>1</td>
      <td>4.07</td>
      <td>14</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>red</td>
      <td>2.0</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>7023</td>
      <td>4.235939</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1573 rows × 25 columns</p>
</div>




```python
def onehot_encode(data, column):
    dummies = pd.get_dummies(data[column], dtype=int)
    data = pd.concat([data, dummies], axis=1)
    data.drop(column, axis=1, inplace=True)
    return data
```


```python
df1 = onehot_encode(df, "product_color")
```


```python
df1
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
      <th>price</th>
      <th>retail_price</th>
      <th>units_sold</th>
      <th>uses_ad_boosts</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>rating_five_count</th>
      <th>rating_four_count</th>
      <th>rating_three_count</th>
      <th>rating_two_count</th>
      <th>...</th>
      <th>white &amp; black</th>
      <th>white &amp; green</th>
      <th>white &amp; red</th>
      <th>whitefloral</th>
      <th>whitestripe</th>
      <th>wine</th>
      <th>wine red</th>
      <th>winered</th>
      <th>winered &amp; yellow</th>
      <th>yellow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.00</td>
      <td>14</td>
      <td>100</td>
      <td>0</td>
      <td>3.76</td>
      <td>54</td>
      <td>26.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>1</th>
      <td>8.00</td>
      <td>22</td>
      <td>20000</td>
      <td>1</td>
      <td>3.45</td>
      <td>6135</td>
      <td>2269.0</td>
      <td>1027.0</td>
      <td>1118.0</td>
      <td>644.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>2</th>
      <td>8.00</td>
      <td>43</td>
      <td>100</td>
      <td>0</td>
      <td>3.57</td>
      <td>14</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>3</th>
      <td>8.00</td>
      <td>8</td>
      <td>5000</td>
      <td>1</td>
      <td>4.03</td>
      <td>579</td>
      <td>295.0</td>
      <td>119.0</td>
      <td>87.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>4</th>
      <td>2.72</td>
      <td>3</td>
      <td>100</td>
      <td>1</td>
      <td>3.10</td>
      <td>20</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <th>1568</th>
      <td>6.00</td>
      <td>9</td>
      <td>10000</td>
      <td>1</td>
      <td>4.08</td>
      <td>1367</td>
      <td>722.0</td>
      <td>293.0</td>
      <td>185.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>1569</th>
      <td>2.00</td>
      <td>56</td>
      <td>100</td>
      <td>1</td>
      <td>3.07</td>
      <td>28</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>1570</th>
      <td>5.00</td>
      <td>19</td>
      <td>100</td>
      <td>0</td>
      <td>3.71</td>
      <td>59</td>
      <td>24.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>1571</th>
      <td>13.00</td>
      <td>11</td>
      <td>100</td>
      <td>0</td>
      <td>2.50</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
      <th>1572</th>
      <td>7.00</td>
      <td>6</td>
      <td>100</td>
      <td>1</td>
      <td>4.07</td>
      <td>14</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
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
<p>1573 rows × 125 columns</p>
</div>




```python
(df1.dtypes == "object").sum()
```




    0




```python
pd.set_option("display.max_rows", None)
```


```python
df1.isnull().sum()
```




    price                             0
    retail_price                      0
    units_sold                        0
    uses_ad_boosts                    0
    rating                            0
    rating_count                      0
    rating_five_count                45
    rating_four_count                45
    rating_three_count               45
    rating_two_count                 45
    rating_one_count                 45
    badges_count                      0
    badge_local_product               0
    badge_product_quality             0
    badge_fast_shipping               0
    product_variation_size_id       195
    product_variation_inventory       0
    shipping_option_price             0
    shipping_is_express               0
    countries_shipped_to              0
    inventory_total                   0
    merchant_rating_count             0
    merchant_rating                   0
    merchant_has_profile_picture      0
    Army green                        0
    Black                             0
    Blue                              0
    Pink                              0
    RED                               0
    Rose red                          0
    White                             0
    applegreen                        0
    apricot                           0
    army                              0
    army green                        0
    armygreen                         0
    beige                             0
    black                             0
    black & blue                      0
    black & green                     0
    black & stripe                    0
    black & white                     0
    black & yellow                    0
    blackwhite                        0
    blue                              0
    blue & pink                       0
    brown                             0
    brown & yellow                    0
    burgundy                          0
    camel                             0
    camouflage                        0
    claret                            0
    coffee                            0
    coolblack                         0
    coralred                          0
    darkblue                          0
    darkgreen                         0
    denimblue                         0
    dustypink                         0
    floral                            0
    fluorescentgreen                  0
    gold                              0
    gray                              0
    gray & white                      0
    green                             0
    grey                              0
    greysnakeskinprint                0
    ivory                             0
    jasper                            0
    khaki                             0
    lakeblue                          0
    leopard                           0
    leopardprint                      0
    light green                       0
    lightblue                         0
    lightgray                         0
    lightgreen                        0
    lightgrey                         0
    lightkhaki                        0
    lightpink                         0
    lightpurple                       0
    lightred                          0
    lightyellow                       0
    mintgreen                         0
    multicolor                        0
    navy                              0
    navy blue                         0
    navyblue                          0
    navyblue & white                  0
    nude                              0
    offblack                          0
    offwhite                          0
    orange                            0
    orange & camouflage               0
    orange-red                        0
    pink                              0
    pink & black                      0
    pink & blue                       0
    pink & grey                       0
    pink & white                      0
    prussianblue                      0
    purple                            0
    rainbow                           0
    red                               0
    red & blue                        0
    rose                              0
    rosegold                          0
    rosered                           0
    silver                            0
    skyblue                           0
    star                              0
    tan                               0
    violet                            0
    watermelonred                     0
    white                             0
    white & black                     0
    white & green                     0
    white & red                       0
    whitefloral                       0
    whitestripe                       0
    wine                              0
    wine red                          0
    winered                           0
    winered & yellow                  0
    yellow                            0
    dtype: int64




```python

```


```python
for column in null_columns:
    df[column] = = df[column].
```


      Cell In[27], line 2
        df[column] = = df[column].
                     ^
    SyntaxError: invalid syntax


