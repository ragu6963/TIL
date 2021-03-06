# [전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)

### 데이터

건물번호 / 시간 / `전력사용량(target)` / 기온 / 풍속 / 습도 / 일조 / 비전기냉방설비운영 / 태양광보유

### 데이터 불러오기 및 전처리

#### 라이브러리 및 데이터 불러오기

```python
"""
Google Colab 
"""
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings

warnings.filterwarnings(action='ignore') 

# Train 데이터셋 불러오기
train_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/전력사용량예측/datasets/train.csv", encoding='CP949') 
# Test 데이터셋 불러오기
test_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/전력사용량예측/datasets/test.csv", encoding='CP949')
```

#### 컬럼 이름 변경

```python
train_df.columns = ['건물번호','날짜시간','전력','기온','풍속','습도','강수량','일조','냉방','태양광']
test_df.columns = ['건물번호','날짜시간','기온','풍속','습도','강수량','일조','냉방','태양광']
```

#### Test 데이터 냉방 태양광 시설 결측치 채우기

```python
ice={}
hot={}
count=0
# 0번부터 60번까지 탐색
for i in range(0, len(train_df), len(train_df)//60):
    count +=1
    ice[count]=train_df.loc[i,'냉방']
    hot[count]=train_df.loc[i,'태양광']
    """
    {건물번호 : 시설유무}
    """

for i in range(len(test_df)):
    test_df.loc[i, '냉방']=ice[test_df['건물번호'][i]]
    test_df.loc[i, '태양광']=hot[test_df['건물번호'][i]]
    """
    해당 건물번호의 시설유무 값을 할당
    """
```

#### Test 데이터 결측치 보간

```python
# pandas 보간법 이용
test_df = test_df.interpolate(method='values')
```

#### 시간 요일 추가

```python
# 시간변수
def time(x):
    return int(x[-2:])

train_df['시간']=train_df['날짜시간'].apply(lambda x: time(x))
test_df['시간']=test_df['날짜시간'].apply(lambda x: time(x))

# 요일 변수
# 월요일 = 0, 일요일 = 6
def weekday(x):
    return pd.to_datetime(x[:10]).weekday()

train_df['요일']=train_df['날짜시간'].apply(lambda x :weekday(x))
test_df['요일']=test_df['날짜시간'].apply(lambda x :weekday(x))
```

#### 불쾌지수 추가

```python
train_df['불쾌지수'] = (0.81 * train_df["기온"]) + 0.01 * train_df['습도'] * (0.99 * train_df['기온'] - 14.3) + 46.3
test_df['불쾌지수'] = (0.81 * test_df["기온"]) + 0.01 * test_df['습도'] * (0.99 * test_df['기온'] - 14.3) + 46.3
```

#### 날짜시간 삭제

```python
train_df = train_df.drop(["날짜시간"],axis=1) 
test_df= test_df.drop(["날짜시간"],axis=1) 
```

#### feature target 분리

```python
X_train =  train_df.drop(["전력"],axis=1) 
y_train = train_df["전력"]
```

#### 정규화

```python
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(test_df)
```

### 손실함수 정의

```python
from keras import backend as k

# smape 정의
def smape(A, F):
    return 100/len(A) * k.sum(2 * k.abs(F - A) / (k.abs(A) + k.abs(F)))
```

### 모델

#### 1. 단순 회귀분석 모델

```python
"""
https://www.tensorflow.org/tutorials/keras/regression?hl=ko
epoch : 100
score : 43.2893627707	
"""
def build_model(): 
    model = keras.Sequential([ 
        Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
        Dense(64, activation='relu'),
        Dense(1)
    ]) 
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=smape, optimizer=optimizer)
    return model

model = build_model()

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, verbose=1,)
```

#### 2. 조금 더 복잡한 회귀분석 모델

```python
"""
epoch : 300
score : 21.2851628884
이후 계속 학습을 해봤으나 손실함수 값이 17 아래로 떨어지지 않음
"""
def build_model():  
    model = keras.Sequential([ 
        Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=smape,optimizer=optimizer)
    return model

model = build_model()
```

#### 3. LGBMRegressor

```python
""" 
LGBMRegressor 
score : 28.1132045582
"""
model=LGBMRegressor(n_estimators=100)
model.fit(X_train, y_train,verbose=100)
```

#### 4. 랜덤포레스트

> 여러개의 decision tree를 형성하고 새로운 데이터 포인트를 각 트리에 동시에 통과시키며, 각 트리가 분류한 결과에서 투표를 실시하여 가장 많이 득표한 결과를 최종 분류 결과로 선택

```python
""" 
Random Forest
score
정규화 미적용 : 7.505382484	
정규화 적용 : 7.455057382
"""
rfr = RandomForestRegressor(random_state=1)
rfr.fit(X_train, y_train)
```




### 실험해볼것

##### target 정규화

##### 건물마다 나눠서 학습시켜서 측정

##### 강수량 제외하고 학습

##### target 로그 변환



---

### 건물나눠서 예측

> 생각보다 결과가 좋지 못했다. 데이터량이 줄어들어서 성능이 안좋아진게 아닐까?

```python
train_df_groups = train_df.groupby(train_df.건물번호)
test_df_groups = test_df.groupby(test_df.건물번호)

submission = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/전력사용량예측/datasets/sample_submission.csv', encoding = 'utf-8')
submission = submission.astype({"answer" : 'float'})
start = 0
for i in range(1,61):
    # i 번째 건물 그룹 불러오기
    train = train_df_groups.get_group(i)
    test = test_df_groups.get_group(i)

    # feature target 분리
    X_train = train.drop(["전력"],axis=1) 
    y_train = train["전력"]

    # 정규화
    scaler = StandardScaler() 
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(test)

    # 학습
    rfr = RandomForestRegressor(random_state=1)
    rfr.fit(X_train, y_train)

    # 예측 
    pred = rfr.predict(X_test)
    # answer 저장 
    end = start + len(pred)
    submission.iloc[start:end]["answer"]= pred 
    start = end
```

### 로그 변환

```python
y_train_log = np.log1p(y_train) 
y_train_log

rfr = RandomForestRegressor(random_state=1)
rfr.fit(X_train, y_train_log)

pred = rfr.predict(X_test)
pred = np.exp(pred)
```

### LGBM + KFold

```python
from sklearn.model_selection import KFold
cross=KFold(n_splits=5, shuffle=True, random_state=42)

folds=[]

for train_idx, valid_idx in cross.split(X_train, y_train_log):
    folds.append((train_idx, valid_idx))
    
from lightgbm import LGBMRegressor
models={}
for fold in range(5):
    print(f'===================={fold+1}=======================')
    train_idx, valid_idx=folds[fold]
    train_x=X_train[train_idx]
    train_y=y_train_log[train_idx]
    valid_x=X_train[valid_idx]
    valid_y=y_train_log[valid_idx]
    
    model=LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train_log, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
             early_stopping_rounds=30, verbose=100)
    models[fold]=model

for i in range(5):
    pred = np.exp(models[i].predict(X_test)) / 5 
    submission['answer'] += pred

submission
```

---

seasonal - statsmodels.api

### 군집화

```python
clustering = {34:0,40:0,42:0,41:0,4:0,10:0,11:0,12:0,
              35:1,6:1,48:1,27:1,57:1,8:1,25:1,56:1,26:1,55:1,47:1,13:1,53:1,18:1,17:1,7:1,46:1,
              31:2,33:2,9:2,3:2,1:2,32:2,
              29:3,38:3,43:3,58:3,15:3,22:3,39:3,54:3,23:3,44:3,45:3,37:3,52:3,2:3,14:3,
              21:4,19:4,50:4,49:4,20:4,51:4,30:4,36:4,28:4,59:4,5:4,60:4,16:4,24:4,}

def cluster(x):
    return clustering[x]

train_df['label'] = train_df['number'].apply(lambda x:cluster(x))
test_df['label'] = test_df['number'].apply(lambda x:cluster(x))
```

