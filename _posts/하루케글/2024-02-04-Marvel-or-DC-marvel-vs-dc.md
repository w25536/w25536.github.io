---
title: "Marvel or DC marvel vs dc"
date: 2024-02-04
last_modified_at: 2024-02-04
categories:
  - 하루케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Marvel or DC marvel vs dc 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("leonardopena/marvel-vs-dc")

print("Path to dataset files:", path)
```

    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/leonardopena/marvel-vs-dc/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


df = pd.read_csv(os.path.join(path, "db.csv"), encoding="latin-1")
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
      <th>Original Title</th>
      <th>Company</th>
      <th>Rate</th>
      <th>Metascore</th>
      <th>Minutes</th>
      <th>Release</th>
      <th>Budget</th>
      <th>Opening Weekend USA</th>
      <th>Gross USA</th>
      <th>Gross Worldwide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Iron Man</td>
      <td>Marvel</td>
      <td>7.9</td>
      <td>79</td>
      <td>126</td>
      <td>2008</td>
      <td>140000000</td>
      <td>98618668</td>
      <td>318604126</td>
      <td>585366247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>The Incredible Hulk</td>
      <td>Marvel</td>
      <td>6.7</td>
      <td>61</td>
      <td>112</td>
      <td>2008</td>
      <td>150000000</td>
      <td>55414050</td>
      <td>134806913</td>
      <td>263427551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Iron Man 2</td>
      <td>Marvel</td>
      <td>7.0</td>
      <td>57</td>
      <td>124</td>
      <td>2010</td>
      <td>200000000</td>
      <td>128122480</td>
      <td>312433331</td>
      <td>623933331</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Thor</td>
      <td>Marvel</td>
      <td>7.0</td>
      <td>57</td>
      <td>115</td>
      <td>2011</td>
      <td>150000000</td>
      <td>65723338</td>
      <td>181030624</td>
      <td>449326618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Captain America: The First Avenger</td>
      <td>Marvel</td>
      <td>6.9</td>
      <td>66</td>
      <td>124</td>
      <td>2011</td>
      <td>140000000</td>
      <td>65058524</td>
      <td>176654505</td>
      <td>370569774</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>The Avengers</td>
      <td>Marvel</td>
      <td>8.0</td>
      <td>69</td>
      <td>143</td>
      <td>2012</td>
      <td>220000000</td>
      <td>207438708</td>
      <td>623357910</td>
      <td>1518812988</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Iron Man Three</td>
      <td>Marvel</td>
      <td>7.2</td>
      <td>62</td>
      <td>130</td>
      <td>2013</td>
      <td>200000000</td>
      <td>174144585</td>
      <td>409013994</td>
      <td>1214811252</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Thor: The Dark World</td>
      <td>Marvel</td>
      <td>6.9</td>
      <td>54</td>
      <td>112</td>
      <td>2013</td>
      <td>170000000</td>
      <td>85737841</td>
      <td>206362140</td>
      <td>644783140</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Captain America: The Winter Soldier</td>
      <td>Marvel</td>
      <td>7.7</td>
      <td>70</td>
      <td>136</td>
      <td>2014</td>
      <td>170000000</td>
      <td>95023721</td>
      <td>259766572</td>
      <td>714421503</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Guardians of the Galaxy</td>
      <td>Marvel</td>
      <td>8.0</td>
      <td>76</td>
      <td>121</td>
      <td>2014</td>
      <td>170000000</td>
      <td>94320883</td>
      <td>333176600</td>
      <td>772776600</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Avengers: Age of Ultron</td>
      <td>Marvel</td>
      <td>7.3</td>
      <td>66</td>
      <td>141</td>
      <td>2015</td>
      <td>250000000</td>
      <td>191271109</td>
      <td>459005868</td>
      <td>1402805868</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Ant-Man</td>
      <td>Marvel</td>
      <td>7.3</td>
      <td>64</td>
      <td>117</td>
      <td>2015</td>
      <td>130000000</td>
      <td>57225526</td>
      <td>180202163</td>
      <td>519311965</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Captain America: Civil War</td>
      <td>Marvel</td>
      <td>7.8</td>
      <td>75</td>
      <td>147</td>
      <td>2016</td>
      <td>250000000</td>
      <td>179139142</td>
      <td>408084349</td>
      <td>1153296293</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Doctor Strange</td>
      <td>Marvel</td>
      <td>7.5</td>
      <td>72</td>
      <td>115</td>
      <td>2016</td>
      <td>165000000</td>
      <td>85058311</td>
      <td>232641920</td>
      <td>677718395</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Guardians of the Galaxy Vol. 2</td>
      <td>Marvel</td>
      <td>7.6</td>
      <td>67</td>
      <td>136</td>
      <td>2017</td>
      <td>200000000</td>
      <td>146510104</td>
      <td>389813101</td>
      <td>863756051</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Spider-Man: Homecoming</td>
      <td>Marvel</td>
      <td>7.4</td>
      <td>73</td>
      <td>133</td>
      <td>2017</td>
      <td>175000000</td>
      <td>117027503</td>
      <td>334201140</td>
      <td>880166924</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Thor:Ragnarok</td>
      <td>Marvel</td>
      <td>7.9</td>
      <td>74</td>
      <td>130</td>
      <td>2017</td>
      <td>180000000</td>
      <td>122744989</td>
      <td>315058289</td>
      <td>853977126</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Black Panther</td>
      <td>Marvel</td>
      <td>7.3</td>
      <td>88</td>
      <td>134</td>
      <td>2018</td>
      <td>200000000</td>
      <td>202003951</td>
      <td>700059566</td>
      <td>1346913161</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Avengers: Infinity War</td>
      <td>Marvel</td>
      <td>8.5</td>
      <td>68</td>
      <td>149</td>
      <td>2018</td>
      <td>321000000</td>
      <td>257698183</td>
      <td>678815482</td>
      <td>2048359754</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Ant-Man and the Wasp</td>
      <td>Marvel</td>
      <td>7.1</td>
      <td>70</td>
      <td>118</td>
      <td>2018</td>
      <td>162000000</td>
      <td>75812205</td>
      <td>216648740</td>
      <td>622674139</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Captain Marve</td>
      <td>Marvel</td>
      <td>6.9</td>
      <td>64</td>
      <td>123</td>
      <td>2019</td>
      <td>175000000</td>
      <td>153433423</td>
      <td>426829839</td>
      <td>1128274794</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Avengers: Endgame</td>
      <td>Marvel</td>
      <td>8.5</td>
      <td>78</td>
      <td>181</td>
      <td>2019</td>
      <td>356000000</td>
      <td>357115007</td>
      <td>858373000</td>
      <td>2797800564</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>Spider-Man: Far from Home</td>
      <td>Marvel</td>
      <td>7.6</td>
      <td>69</td>
      <td>129</td>
      <td>2019</td>
      <td>160000000</td>
      <td>92579212</td>
      <td>390532085</td>
      <td>1131927996</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Catwoman</td>
      <td>DC</td>
      <td>3.3</td>
      <td>27</td>
      <td>104</td>
      <td>2004</td>
      <td>100000000</td>
      <td>16728411</td>
      <td>40202379</td>
      <td>82102379</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>Batman Begins</td>
      <td>DC</td>
      <td>8.2</td>
      <td>70</td>
      <td>140</td>
      <td>2005</td>
      <td>150000000</td>
      <td>48745440</td>
      <td>206852432</td>
      <td>373413297</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>Superman Returns</td>
      <td>DC</td>
      <td>6.0</td>
      <td>72</td>
      <td>154</td>
      <td>2006</td>
      <td>270000000</td>
      <td>52535096</td>
      <td>200081192</td>
      <td>391081192</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>The Dark Knight</td>
      <td>DC</td>
      <td>9.0</td>
      <td>84</td>
      <td>152</td>
      <td>2008</td>
      <td>185000000</td>
      <td>158411483</td>
      <td>535234033</td>
      <td>1004934033</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>Watchmen</td>
      <td>DC</td>
      <td>7.6</td>
      <td>56</td>
      <td>162</td>
      <td>2009</td>
      <td>130000000</td>
      <td>55214334</td>
      <td>107509799</td>
      <td>185258983</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Jonah Hex</td>
      <td>DC</td>
      <td>4.7</td>
      <td>33</td>
      <td>81</td>
      <td>2010</td>
      <td>47000000</td>
      <td>5379365</td>
      <td>10547117</td>
      <td>10903312</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Green Lantern</td>
      <td>DC</td>
      <td>5.5</td>
      <td>39</td>
      <td>114</td>
      <td>2011</td>
      <td>200000000</td>
      <td>53174303</td>
      <td>116601172</td>
      <td>219851172</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>The Dark Knight Rises</td>
      <td>DC</td>
      <td>8.4</td>
      <td>78</td>
      <td>164</td>
      <td>2012</td>
      <td>250000000</td>
      <td>160887295</td>
      <td>448139099</td>
      <td>1081041287</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>Man of Steel</td>
      <td>DC</td>
      <td>7.1</td>
      <td>55</td>
      <td>143</td>
      <td>2013</td>
      <td>225000000</td>
      <td>116619362</td>
      <td>291045518</td>
      <td>668045518</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>Batman v Superman: Dawn of Justice</td>
      <td>DC</td>
      <td>6.5</td>
      <td>44</td>
      <td>151</td>
      <td>2016</td>
      <td>250000000</td>
      <td>166007347</td>
      <td>330360194</td>
      <td>873634919</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Suicide Squad</td>
      <td>DC</td>
      <td>6.0</td>
      <td>40</td>
      <td>123</td>
      <td>2016</td>
      <td>175000000</td>
      <td>133682248</td>
      <td>325100054</td>
      <td>746846894</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>Wonder Woman</td>
      <td>DC</td>
      <td>7.4</td>
      <td>76</td>
      <td>141</td>
      <td>2017</td>
      <td>149000000</td>
      <td>103251471</td>
      <td>412563408</td>
      <td>821847012</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>Justice League</td>
      <td>DC</td>
      <td>6.4</td>
      <td>45</td>
      <td>120</td>
      <td>2017</td>
      <td>300000000</td>
      <td>93842239</td>
      <td>229024295</td>
      <td>657924295</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>Aquaman</td>
      <td>DC</td>
      <td>7.0</td>
      <td>55</td>
      <td>143</td>
      <td>2018</td>
      <td>160000000</td>
      <td>67873522</td>
      <td>335061807</td>
      <td>1148161807</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>Shazam!</td>
      <td>DC</td>
      <td>7.1</td>
      <td>71</td>
      <td>132</td>
      <td>2019</td>
      <td>100000000</td>
      <td>53505326</td>
      <td>140371656</td>
      <td>364571656</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>Joker</td>
      <td>DC</td>
      <td>8.7</td>
      <td>59</td>
      <td>122</td>
      <td>2019</td>
      <td>55000000</td>
      <td>96202337</td>
      <td>333204580</td>
      <td>1060504580</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39 entries, 0 to 38
    Data columns (total 11 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Unnamed: 0           39 non-null     int64  
     1   Original Title       39 non-null     object 
     2   Company              39 non-null     object 
     3   Rate                 39 non-null     float64
     4   Metascore            39 non-null     int64  
     5   Minutes              39 non-null     object 
     6   Release              39 non-null     int64  
     7   Budget               39 non-null     object 
     8   Opening Weekend USA  39 non-null     int64  
     9   Gross USA            39 non-null     int64  
     10  Gross Worldwide      39 non-null     int64  
    dtypes: float64(1), int64(6), object(4)
    memory usage: 3.5+ KB



```python
df1 = df.drop(["Unnamed: 0", "Original Title"], axis=1)
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
      <th>Company</th>
      <th>Rate</th>
      <th>Metascore</th>
      <th>Minutes</th>
      <th>Release</th>
      <th>Budget</th>
      <th>Opening Weekend USA</th>
      <th>Gross USA</th>
      <th>Gross Worldwide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marvel</td>
      <td>7.9</td>
      <td>79</td>
      <td>126</td>
      <td>2008</td>
      <td>140000000</td>
      <td>98618668</td>
      <td>318604126</td>
      <td>585366247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marvel</td>
      <td>6.7</td>
      <td>61</td>
      <td>112</td>
      <td>2008</td>
      <td>150000000</td>
      <td>55414050</td>
      <td>134806913</td>
      <td>263427551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Marvel</td>
      <td>7.0</td>
      <td>57</td>
      <td>124</td>
      <td>2010</td>
      <td>200000000</td>
      <td>128122480</td>
      <td>312433331</td>
      <td>623933331</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marvel</td>
      <td>7.0</td>
      <td>57</td>
      <td>115</td>
      <td>2011</td>
      <td>150000000</td>
      <td>65723338</td>
      <td>181030624</td>
      <td>449326618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marvel</td>
      <td>6.9</td>
      <td>66</td>
      <td>124</td>
      <td>2011</td>
      <td>140000000</td>
      <td>65058524</td>
      <td>176654505</td>
      <td>370569774</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Marvel</td>
      <td>8.0</td>
      <td>69</td>
      <td>143</td>
      <td>2012</td>
      <td>220000000</td>
      <td>207438708</td>
      <td>623357910</td>
      <td>1518812988</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Marvel</td>
      <td>7.2</td>
      <td>62</td>
      <td>130</td>
      <td>2013</td>
      <td>200000000</td>
      <td>174144585</td>
      <td>409013994</td>
      <td>1214811252</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Marvel</td>
      <td>6.9</td>
      <td>54</td>
      <td>112</td>
      <td>2013</td>
      <td>170000000</td>
      <td>85737841</td>
      <td>206362140</td>
      <td>644783140</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Marvel</td>
      <td>7.7</td>
      <td>70</td>
      <td>136</td>
      <td>2014</td>
      <td>170000000</td>
      <td>95023721</td>
      <td>259766572</td>
      <td>714421503</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Marvel</td>
      <td>8.0</td>
      <td>76</td>
      <td>121</td>
      <td>2014</td>
      <td>170000000</td>
      <td>94320883</td>
      <td>333176600</td>
      <td>772776600</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Marvel</td>
      <td>7.3</td>
      <td>66</td>
      <td>141</td>
      <td>2015</td>
      <td>250000000</td>
      <td>191271109</td>
      <td>459005868</td>
      <td>1402805868</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Marvel</td>
      <td>7.3</td>
      <td>64</td>
      <td>117</td>
      <td>2015</td>
      <td>130000000</td>
      <td>57225526</td>
      <td>180202163</td>
      <td>519311965</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Marvel</td>
      <td>7.8</td>
      <td>75</td>
      <td>147</td>
      <td>2016</td>
      <td>250000000</td>
      <td>179139142</td>
      <td>408084349</td>
      <td>1153296293</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Marvel</td>
      <td>7.5</td>
      <td>72</td>
      <td>115</td>
      <td>2016</td>
      <td>165000000</td>
      <td>85058311</td>
      <td>232641920</td>
      <td>677718395</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Marvel</td>
      <td>7.6</td>
      <td>67</td>
      <td>136</td>
      <td>2017</td>
      <td>200000000</td>
      <td>146510104</td>
      <td>389813101</td>
      <td>863756051</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Marvel</td>
      <td>7.4</td>
      <td>73</td>
      <td>133</td>
      <td>2017</td>
      <td>175000000</td>
      <td>117027503</td>
      <td>334201140</td>
      <td>880166924</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Marvel</td>
      <td>7.9</td>
      <td>74</td>
      <td>130</td>
      <td>2017</td>
      <td>180000000</td>
      <td>122744989</td>
      <td>315058289</td>
      <td>853977126</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Marvel</td>
      <td>7.3</td>
      <td>88</td>
      <td>134</td>
      <td>2018</td>
      <td>200000000</td>
      <td>202003951</td>
      <td>700059566</td>
      <td>1346913161</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Marvel</td>
      <td>8.5</td>
      <td>68</td>
      <td>149</td>
      <td>2018</td>
      <td>321000000</td>
      <td>257698183</td>
      <td>678815482</td>
      <td>2048359754</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Marvel</td>
      <td>7.1</td>
      <td>70</td>
      <td>118</td>
      <td>2018</td>
      <td>162000000</td>
      <td>75812205</td>
      <td>216648740</td>
      <td>622674139</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Marvel</td>
      <td>6.9</td>
      <td>64</td>
      <td>123</td>
      <td>2019</td>
      <td>175000000</td>
      <td>153433423</td>
      <td>426829839</td>
      <td>1128274794</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Marvel</td>
      <td>8.5</td>
      <td>78</td>
      <td>181</td>
      <td>2019</td>
      <td>356000000</td>
      <td>357115007</td>
      <td>858373000</td>
      <td>2797800564</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Marvel</td>
      <td>7.6</td>
      <td>69</td>
      <td>129</td>
      <td>2019</td>
      <td>160000000</td>
      <td>92579212</td>
      <td>390532085</td>
      <td>1131927996</td>
    </tr>
    <tr>
      <th>23</th>
      <td>DC</td>
      <td>3.3</td>
      <td>27</td>
      <td>104</td>
      <td>2004</td>
      <td>100000000</td>
      <td>16728411</td>
      <td>40202379</td>
      <td>82102379</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DC</td>
      <td>8.2</td>
      <td>70</td>
      <td>140</td>
      <td>2005</td>
      <td>150000000</td>
      <td>48745440</td>
      <td>206852432</td>
      <td>373413297</td>
    </tr>
    <tr>
      <th>25</th>
      <td>DC</td>
      <td>6.0</td>
      <td>72</td>
      <td>154</td>
      <td>2006</td>
      <td>270000000</td>
      <td>52535096</td>
      <td>200081192</td>
      <td>391081192</td>
    </tr>
    <tr>
      <th>26</th>
      <td>DC</td>
      <td>9.0</td>
      <td>84</td>
      <td>152</td>
      <td>2008</td>
      <td>185000000</td>
      <td>158411483</td>
      <td>535234033</td>
      <td>1004934033</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DC</td>
      <td>7.6</td>
      <td>56</td>
      <td>162</td>
      <td>2009</td>
      <td>130000000</td>
      <td>55214334</td>
      <td>107509799</td>
      <td>185258983</td>
    </tr>
    <tr>
      <th>28</th>
      <td>DC</td>
      <td>4.7</td>
      <td>33</td>
      <td>81</td>
      <td>2010</td>
      <td>47000000</td>
      <td>5379365</td>
      <td>10547117</td>
      <td>10903312</td>
    </tr>
    <tr>
      <th>29</th>
      <td>DC</td>
      <td>5.5</td>
      <td>39</td>
      <td>114</td>
      <td>2011</td>
      <td>200000000</td>
      <td>53174303</td>
      <td>116601172</td>
      <td>219851172</td>
    </tr>
    <tr>
      <th>30</th>
      <td>DC</td>
      <td>8.4</td>
      <td>78</td>
      <td>164</td>
      <td>2012</td>
      <td>250000000</td>
      <td>160887295</td>
      <td>448139099</td>
      <td>1081041287</td>
    </tr>
    <tr>
      <th>31</th>
      <td>DC</td>
      <td>7.1</td>
      <td>55</td>
      <td>143</td>
      <td>2013</td>
      <td>225000000</td>
      <td>116619362</td>
      <td>291045518</td>
      <td>668045518</td>
    </tr>
    <tr>
      <th>32</th>
      <td>DC</td>
      <td>6.5</td>
      <td>44</td>
      <td>151</td>
      <td>2016</td>
      <td>250000000</td>
      <td>166007347</td>
      <td>330360194</td>
      <td>873634919</td>
    </tr>
    <tr>
      <th>33</th>
      <td>DC</td>
      <td>6.0</td>
      <td>40</td>
      <td>123</td>
      <td>2016</td>
      <td>175000000</td>
      <td>133682248</td>
      <td>325100054</td>
      <td>746846894</td>
    </tr>
    <tr>
      <th>34</th>
      <td>DC</td>
      <td>7.4</td>
      <td>76</td>
      <td>141</td>
      <td>2017</td>
      <td>149000000</td>
      <td>103251471</td>
      <td>412563408</td>
      <td>821847012</td>
    </tr>
    <tr>
      <th>35</th>
      <td>DC</td>
      <td>6.4</td>
      <td>45</td>
      <td>120</td>
      <td>2017</td>
      <td>300000000</td>
      <td>93842239</td>
      <td>229024295</td>
      <td>657924295</td>
    </tr>
    <tr>
      <th>36</th>
      <td>DC</td>
      <td>7.0</td>
      <td>55</td>
      <td>143</td>
      <td>2018</td>
      <td>160000000</td>
      <td>67873522</td>
      <td>335061807</td>
      <td>1148161807</td>
    </tr>
    <tr>
      <th>37</th>
      <td>DC</td>
      <td>7.1</td>
      <td>71</td>
      <td>132</td>
      <td>2019</td>
      <td>100000000</td>
      <td>53505326</td>
      <td>140371656</td>
      <td>364571656</td>
    </tr>
    <tr>
      <th>38</th>
      <td>DC</td>
      <td>8.7</td>
      <td>59</td>
      <td>122</td>
      <td>2019</td>
      <td>55000000</td>
      <td>96202337</td>
      <td>333204580</td>
      <td>1060504580</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df1["Company"]
X = df1.drop(["Company"], axis=1)
```


```python
scaler = StandardScaler()
std_X = scaler.fit_transform(X)

sns.heatmap(X.corr(), annot=True)
```




    <Axes: >




    
![png](035_Marvel_or_DC_marvel_vs_dc_files/035_Marvel_or_DC_marvel_vs_dc_7_1.png)
    



```python
corr = X.corr()

corr["Rate"].sort_values(ascending=True)
```




    Budget                 0.265655
    Release                0.331977
    Opening Weekend USA    0.521689
    Gross Worldwide        0.565348
    Minutes                0.583813
    Gross USA              0.609582
    Metascore              0.786901
    Rate                   1.000000
    Name: Rate, dtype: float64




```python
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
X_train, X_test, y_train, y_test = train_test_split(std_X, y, train_size=0.7)
```


```python
model = LogisticRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)
```




    0.9166666666666666




```python
y_preds = model.predict(X_test)

pd.DataFrame(y)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
