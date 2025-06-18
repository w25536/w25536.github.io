---
title: "Health Insurance Interest Prediction health insurance cross sell prediction"
date: 2024-02-07
last_modified_at: 2024-02-07
categories:
  - 1일1케글
tags:
  - 머신러닝
  - 데이터사이언스
  - kaggle
excerpt: "Health Insurance Interest Prediction health insurance cross sell prediction 프로젝트"
use_math: true
classes: wide
---
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("anmolkumar/health-insurance-cross-sell-prediction")

print("Path to dataset files:", path)
```

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.5)
    Path to dataset files: /Users/jeongho/.cache/kagglehub/datasets/anmolkumar/health-insurance-cross-sell-prediction/versions/1



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```


```python
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))
sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
```


```python
print(test_df.shape)
print(train_df.shape)
```

    (127037, 11)
    (381109, 12)



```python
df = pd.concat([test_df, train_df], axis=0)
```


```python
def binary_encode(df, column, postive_label):
    df = df.copy()
    df.apply(lambda x: 1 if x == postive_label else 0)
    return df


def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df


def onehot_encode(df, column):
    dummies = pd.get_dummies(df[column])
    df = df.concat([dummies, df], axis=1)
    df = df.drop(column, axis=1)
    return df
```


```python
cat_cols = list(df.select_dtypes("object").columns)
```


```python
dict = {}

for col in cat_cols:
    dict[col] = {col: list(train_df[col].unique())}

print(dict)
```

    {'Gender': {'Gender': ['Male', 'Female']}, 'Vehicle_Age': {'Vehicle_Age': ['> 2 Years', '1-2 Year', '< 1 Year']}, 'Vehicle_Damage': {'Vehicle_Damage': ['Yes', 'No']}}



```python
def preprocess_input(df):

    df = df.copy()
    # nominal feature
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Vehicle_Age"] = df["Vehicle_Age"].map(
        {"> 2 Years": 2, "1-2 Year": 1, "< 1 Year": 0}
    )
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map({"Yes": 1, "No": 0})

    test_df = df.iloc[:127037]

    test_ids = list(test_df["id"].iloc[:127037])

    df = df.drop(["id"], axis=1)

    test_df = df.iloc[:127037]
    train_df = df.iloc[127037:]

    y = train_df["Response"]
    X = train_df.drop(["Response"], axis=1)
    test_df = test_df.drop(["Response"], axis=1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.fit_transform(test_df)

    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    test_df = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

    return X, y, test_df, test_ids
```


```python
X, y, test_df, test_ids = preprocess_input(df)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
```


```python
y.sum() / len(y)
```




    0.12256336113815208




```python
# X.plot(kind = 'box', figsize=(10, 10),  logy=True)
X.shape
```




    (381109, 10)




```python
inputs = tf.keras.Input(shape=(10,))  # X.shape 10 features
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

batch_size = 64
epochs = 25

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],
    verbose=0,
)
```


```python
import plotly.express as px

fig = px.line(
    history.history,
    y=["loss", "val_loss"],
    labels={"index": "Epoch", "value": "Loss"},
    title="Training History",
)
fig.show()
```




```python
plt.figure(figsize=(14, 10))
epochs_range = range(1, epochs + 1)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
```


    
![png](038_Health_Insurance_Interest_Prediction_health_insurance_cross_sell_prediction_files/038_Health_Insurance_Interest_Prediction_health_insurance_cross_sell_prediction_15_0.png)
    



```python
np.argmin(val_loss)
```




    20




```python
model.evaluate(X_test, y_test)
```

    [1m3573/3573[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 578us/step - auc: 0.8529 - loss: 0.2685





    [0.26833873987197876, 0.8527363538742065]




```python
preds = model.predict(test_df)
```

    [1m   1/3970[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:47[0m 27ms/step

    [1m3970/3970[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 307us/step



```python
preds
```




    array([[8.3298240e-05],
           [2.7377096e-01],
           [2.9221359e-01],
           ...,
           [9.5657604e-05],
           [2.0201905e-05],
           [5.1094493e-04]], dtype=float32)




```python
preds >= 0.5
```




    array([[False],
           [False],
           [False],
           ...,
           [False],
           [False],
           [False]])




```python
preds = list(map(lambda x: int(x[0]), preds >= 0.5))

test_ids
```




    [381110,
     381111,
     381112,
     381113,
     381114,
     381115,
     381116,
     381117,
     381118,
     381119,
     381120,
     381121,
     381122,
     381123,
     381124,
     381125,
     381126,
     381127,
     381128,
     381129,
     381130,
     381131,
     381132,
     381133,
     381134,
     381135,
     381136,
     381137,
     381138,
     381139,
     381140,
     381141,
     381142,
     381143,
     381144,
     381145,
     381146,
     381147,
     381148,
     381149,
     381150,
     381151,
     381152,
     381153,
     381154,
     381155,
     381156,
     381157,
     381158,
     381159,
     381160,
     381161,
     381162,
     381163,
     381164,
     381165,
     381166,
     381167,
     381168,
     381169,
     381170,
     381171,
     381172,
     381173,
     381174,
     381175,
     381176,
     381177,
     381178,
     381179,
     381180,
     381181,
     381182,
     381183,
     381184,
     381185,
     381186,
     381187,
     381188,
     381189,
     381190,
     381191,
     381192,
     381193,
     381194,
     381195,
     381196,
     381197,
     381198,
     381199,
     381200,
     381201,
     381202,
     381203,
     381204,
     381205,
     381206,
     381207,
     381208,
     381209,
     381210,
     381211,
     381212,
     381213,
     381214,
     381215,
     381216,
     381217,
     381218,
     381219,
     381220,
     381221,
     381222,
     381223,
     381224,
     381225,
     381226,
     381227,
     381228,
     381229,
     381230,
     381231,
     381232,
     381233,
     381234,
     381235,
     381236,
     381237,
     381238,
     381239,
     381240,
     381241,
     381242,
     381243,
     381244,
     381245,
     381246,
     381247,
     381248,
     381249,
     381250,
     381251,
     381252,
     381253,
     381254,
     381255,
     381256,
     381257,
     381258,
     381259,
     381260,
     381261,
     381262,
     381263,
     381264,
     381265,
     381266,
     381267,
     381268,
     381269,
     381270,
     381271,
     381272,
     381273,
     381274,
     381275,
     381276,
     381277,
     381278,
     381279,
     381280,
     381281,
     381282,
     381283,
     381284,
     381285,
     381286,
     381287,
     381288,
     381289,
     381290,
     381291,
     381292,
     381293,
     381294,
     381295,
     381296,
     381297,
     381298,
     381299,
     381300,
     381301,
     381302,
     381303,
     381304,
     381305,
     381306,
     381307,
     381308,
     381309,
     381310,
     381311,
     381312,
     381313,
     381314,
     381315,
     381316,
     381317,
     381318,
     381319,
     381320,
     381321,
     381322,
     381323,
     381324,
     381325,
     381326,
     381327,
     381328,
     381329,
     381330,
     381331,
     381332,
     381333,
     381334,
     381335,
     381336,
     381337,
     381338,
     381339,
     381340,
     381341,
     381342,
     381343,
     381344,
     381345,
     381346,
     381347,
     381348,
     381349,
     381350,
     381351,
     381352,
     381353,
     381354,
     381355,
     381356,
     381357,
     381358,
     381359,
     381360,
     381361,
     381362,
     381363,
     381364,
     381365,
     381366,
     381367,
     381368,
     381369,
     381370,
     381371,
     381372,
     381373,
     381374,
     381375,
     381376,
     381377,
     381378,
     381379,
     381380,
     381381,
     381382,
     381383,
     381384,
     381385,
     381386,
     381387,
     381388,
     381389,
     381390,
     381391,
     381392,
     381393,
     381394,
     381395,
     381396,
     381397,
     381398,
     381399,
     381400,
     381401,
     381402,
     381403,
     381404,
     381405,
     381406,
     381407,
     381408,
     381409,
     381410,
     381411,
     381412,
     381413,
     381414,
     381415,
     381416,
     381417,
     381418,
     381419,
     381420,
     381421,
     381422,
     381423,
     381424,
     381425,
     381426,
     381427,
     381428,
     381429,
     381430,
     381431,
     381432,
     381433,
     381434,
     381435,
     381436,
     381437,
     381438,
     381439,
     381440,
     381441,
     381442,
     381443,
     381444,
     381445,
     381446,
     381447,
     381448,
     381449,
     381450,
     381451,
     381452,
     381453,
     381454,
     381455,
     381456,
     381457,
     381458,
     381459,
     381460,
     381461,
     381462,
     381463,
     381464,
     381465,
     381466,
     381467,
     381468,
     381469,
     381470,
     381471,
     381472,
     381473,
     381474,
     381475,
     381476,
     381477,
     381478,
     381479,
     381480,
     381481,
     381482,
     381483,
     381484,
     381485,
     381486,
     381487,
     381488,
     381489,
     381490,
     381491,
     381492,
     381493,
     381494,
     381495,
     381496,
     381497,
     381498,
     381499,
     381500,
     381501,
     381502,
     381503,
     381504,
     381505,
     381506,
     381507,
     381508,
     381509,
     381510,
     381511,
     381512,
     381513,
     381514,
     381515,
     381516,
     381517,
     381518,
     381519,
     381520,
     381521,
     381522,
     381523,
     381524,
     381525,
     381526,
     381527,
     381528,
     381529,
     381530,
     381531,
     381532,
     381533,
     381534,
     381535,
     381536,
     381537,
     381538,
     381539,
     381540,
     381541,
     381542,
     381543,
     381544,
     381545,
     381546,
     381547,
     381548,
     381549,
     381550,
     381551,
     381552,
     381553,
     381554,
     381555,
     381556,
     381557,
     381558,
     381559,
     381560,
     381561,
     381562,
     381563,
     381564,
     381565,
     381566,
     381567,
     381568,
     381569,
     381570,
     381571,
     381572,
     381573,
     381574,
     381575,
     381576,
     381577,
     381578,
     381579,
     381580,
     381581,
     381582,
     381583,
     381584,
     381585,
     381586,
     381587,
     381588,
     381589,
     381590,
     381591,
     381592,
     381593,
     381594,
     381595,
     381596,
     381597,
     381598,
     381599,
     381600,
     381601,
     381602,
     381603,
     381604,
     381605,
     381606,
     381607,
     381608,
     381609,
     381610,
     381611,
     381612,
     381613,
     381614,
     381615,
     381616,
     381617,
     381618,
     381619,
     381620,
     381621,
     381622,
     381623,
     381624,
     381625,
     381626,
     381627,
     381628,
     381629,
     381630,
     381631,
     381632,
     381633,
     381634,
     381635,
     381636,
     381637,
     381638,
     381639,
     381640,
     381641,
     381642,
     381643,
     381644,
     381645,
     381646,
     381647,
     381648,
     381649,
     381650,
     381651,
     381652,
     381653,
     381654,
     381655,
     381656,
     381657,
     381658,
     381659,
     381660,
     381661,
     381662,
     381663,
     381664,
     381665,
     381666,
     381667,
     381668,
     381669,
     381670,
     381671,
     381672,
     381673,
     381674,
     381675,
     381676,
     381677,
     381678,
     381679,
     381680,
     381681,
     381682,
     381683,
     381684,
     381685,
     381686,
     381687,
     381688,
     381689,
     381690,
     381691,
     381692,
     381693,
     381694,
     381695,
     381696,
     381697,
     381698,
     381699,
     381700,
     381701,
     381702,
     381703,
     381704,
     381705,
     381706,
     381707,
     381708,
     381709,
     381710,
     381711,
     381712,
     381713,
     381714,
     381715,
     381716,
     381717,
     381718,
     381719,
     381720,
     381721,
     381722,
     381723,
     381724,
     381725,
     381726,
     381727,
     381728,
     381729,
     381730,
     381731,
     381732,
     381733,
     381734,
     381735,
     381736,
     381737,
     381738,
     381739,
     381740,
     381741,
     381742,
     381743,
     381744,
     381745,
     381746,
     381747,
     381748,
     381749,
     381750,
     381751,
     381752,
     381753,
     381754,
     381755,
     381756,
     381757,
     381758,
     381759,
     381760,
     381761,
     381762,
     381763,
     381764,
     381765,
     381766,
     381767,
     381768,
     381769,
     381770,
     381771,
     381772,
     381773,
     381774,
     381775,
     381776,
     381777,
     381778,
     381779,
     381780,
     381781,
     381782,
     381783,
     381784,
     381785,
     381786,
     381787,
     381788,
     381789,
     381790,
     381791,
     381792,
     381793,
     381794,
     381795,
     381796,
     381797,
     381798,
     381799,
     381800,
     381801,
     381802,
     381803,
     381804,
     381805,
     381806,
     381807,
     381808,
     381809,
     381810,
     381811,
     381812,
     381813,
     381814,
     381815,
     381816,
     381817,
     381818,
     381819,
     381820,
     381821,
     381822,
     381823,
     381824,
     381825,
     381826,
     381827,
     381828,
     381829,
     381830,
     381831,
     381832,
     381833,
     381834,
     381835,
     381836,
     381837,
     381838,
     381839,
     381840,
     381841,
     381842,
     381843,
     381844,
     381845,
     381846,
     381847,
     381848,
     381849,
     381850,
     381851,
     381852,
     381853,
     381854,
     381855,
     381856,
     381857,
     381858,
     381859,
     381860,
     381861,
     381862,
     381863,
     381864,
     381865,
     381866,
     381867,
     381868,
     381869,
     381870,
     381871,
     381872,
     381873,
     381874,
     381875,
     381876,
     381877,
     381878,
     381879,
     381880,
     381881,
     381882,
     381883,
     381884,
     381885,
     381886,
     381887,
     381888,
     381889,
     381890,
     381891,
     381892,
     381893,
     381894,
     381895,
     381896,
     381897,
     381898,
     381899,
     381900,
     381901,
     381902,
     381903,
     381904,
     381905,
     381906,
     381907,
     381908,
     381909,
     381910,
     381911,
     381912,
     381913,
     381914,
     381915,
     381916,
     381917,
     381918,
     381919,
     381920,
     381921,
     381922,
     381923,
     381924,
     381925,
     381926,
     381927,
     381928,
     381929,
     381930,
     381931,
     381932,
     381933,
     381934,
     381935,
     381936,
     381937,
     381938,
     381939,
     381940,
     381941,
     381942,
     381943,
     381944,
     381945,
     381946,
     381947,
     381948,
     381949,
     381950,
     381951,
     381952,
     381953,
     381954,
     381955,
     381956,
     381957,
     381958,
     381959,
     381960,
     381961,
     381962,
     381963,
     381964,
     381965,
     381966,
     381967,
     381968,
     381969,
     381970,
     381971,
     381972,
     381973,
     381974,
     381975,
     381976,
     381977,
     381978,
     381979,
     381980,
     381981,
     381982,
     381983,
     381984,
     381985,
     381986,
     381987,
     381988,
     381989,
     381990,
     381991,
     381992,
     381993,
     381994,
     381995,
     381996,
     381997,
     381998,
     381999,
     382000,
     382001,
     382002,
     382003,
     382004,
     382005,
     382006,
     382007,
     382008,
     382009,
     382010,
     382011,
     382012,
     382013,
     382014,
     382015,
     382016,
     382017,
     382018,
     382019,
     382020,
     382021,
     382022,
     382023,
     382024,
     382025,
     382026,
     382027,
     382028,
     382029,
     382030,
     382031,
     382032,
     382033,
     382034,
     382035,
     382036,
     382037,
     382038,
     382039,
     382040,
     382041,
     382042,
     382043,
     382044,
     382045,
     382046,
     382047,
     382048,
     382049,
     382050,
     382051,
     382052,
     382053,
     382054,
     382055,
     382056,
     382057,
     382058,
     382059,
     382060,
     382061,
     382062,
     382063,
     382064,
     382065,
     382066,
     382067,
     382068,
     382069,
     382070,
     382071,
     382072,
     382073,
     382074,
     382075,
     382076,
     382077,
     382078,
     382079,
     382080,
     382081,
     382082,
     382083,
     382084,
     382085,
     382086,
     382087,
     382088,
     382089,
     382090,
     382091,
     382092,
     382093,
     382094,
     382095,
     382096,
     382097,
     382098,
     382099,
     382100,
     382101,
     382102,
     382103,
     382104,
     382105,
     382106,
     382107,
     382108,
     382109,
     ...]




```python
submission_df = pd.concat([pd.Series(test_ids), pd.Series(preds)], axis=1)
submission_columns = ["id", "Response"]
```


```python
submission = pd.DataFrame(submission_df, columns=submission_columns)
submission
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
      <th>id</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>127032</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>127033</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>127034</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>127035</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>127036</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>127037 rows × 2 columns</p>
</div>




```python
sample_submission.shape == submission.shape
```




    True




```python
submission.to_csv("./submission.csv")
```
