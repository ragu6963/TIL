> 도서 : 밑바닥 부터 시작하는 딥러닝 [도서](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198) [깃허브](https://github.com/WegraLee/deep-learning-from-scratch)

# 7장 합성곱 신경망(CNN)

## 7.1 전체 구조

`CNN`도 지금까지와의 신경망 같이 레고 블록처럼 계층을 조립하여 만든다. 

다만, `합성곱 계층(convolutional layer)`과 `풀링 계층(pooling layer)` 이 새롭게 등장한다.

지금까지의 신경망은 인접하는 계층의 모든 뉴런과 결합되어 있었다.

이를 `완전연결(fully-connected)`라고 하며, `Affine 계층`이라는 이름으로 구현했다.

> `Affine 계층`을 사용한 층이 5개인 완전연결 신경망

<img src="7장_합성곱 신경망(CNN).assets/fig 7-1.png">



`완전연결 신경망`은 Affine 계층 뒤에 활성화 함수 계층(ReLU or Sigmoid)이 이어진다.

>  `CNN 구조`

<img src="7장_합성곱 신경망(CNN).assets/fig 7-2.png">

`CNN 계층`은 `Conv-ReLU-(Pooling)`의 흐름으로 연결된다.(풀링 계층은 생략되기도 한다.)

그림에서 주목할 점은 출력에 가까운 층에서는 `Affine-ReLU`를 사용할 수 있고, 마지막 출력 계층에서는 `Affine-Softmax`조합을 그대로 사용한다는 점이다.

## 7.2 합성곱 계층

CNN에서는 `패딩(padding),스트라이드(strid)` 등 고유의 용어가 등장한다.

또한, 각 계층 사이에서 3차원 데이터같은 `입체적인 데이터`가 흐른다는 점이 완전연결 신경망과 다르다.

### 7.2.1 완전연결 계층의 문제점

`완전연결 계층`의 문제점은 `데이터 형상이 무시`된다는 점이다.

예를들어 이미지 데이터의 경우 `세로ㆍ가로ㆍ채널(색상)`로 구성된 `3차원 데이터`이다. 

그러나 완전연결 계층에 입력할 때는 1차원 데이터로 `평탄화(flat`)해줘야 한다.

평탄화로 인해 3차원 데이터에 있는 `공간적 정보` 가 완전연결 계층에서는 무시되기 때문에 공간적 정보를 살릴 수 없다.

> 공간적 정보 : 공간적으로 가까운 픽셀은 값이 비슷하거나, RGB의 각 채널은 서로 밀접하게 괸련되어 있거나, 거리가 먼 픽셀은 연관이 없거나

---

반면, `합성곱 계층`은 `데이터 형상을 유지`한다. 이미지는 3차원 데이터로 입력받고, 3차원 데이터를 전달한다.

> CNN 에서는 합성곱 계층의 입출력 데이터를 `특징 맵(feature map)` , 입력 데이터는 `입력 특징 맵` 출력 데이터는 `출력 특징 맵`이라고도 한다.

### 7.2.2 합성곱 연산

합성곱 계층에서` 합성곱(Convolution) 연산`을 처리한다. 이미지 처리에서는 `필터 연산`에 해당한다.

> 합성곱 연산 예

<img src="7장_합성곱 신경망(CNN).assets/fig 7-3.png">

입력 데이터와 필터는 세로 가로의 형상을 가진다. 세로와 가로는 `높이 height`, `너비 width`로 표현한다.  

> 필터는 `커널`이라고도 부른다.

예시에서는 입력 데이터는 (4,4), 필터는 (3,3), 출력은 (2,2) 이다.

---

합성곱 연산은 필터의 `윈도우`를 일정 간격으로 이동해가며 입력 데이터에 적용한다. 

> 합성곱 연산의 계산 순서, 여기서 윈도우는 그림의 회색 3 X 3이다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-4.png">

완전연결 신경망에는 가중치와 편향이 존재하는데 CNN에서는 `필터 = 가중치`이고, CNN에도 `편향`이 존재한다.

> 합성곱 연상에서의 편향

<img src="7장_합성곱 신경망(CNN).assets/fig 7-5.png">

### 7.2.3 패딩

합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정 값(0, etc...)으로 채우기도 한다. 

이것을 `패딩 padding`이라고 한다.

예를들면 아래 그림은 (4,4) 크기의 입력 데이터에 폭 1의 패딩을 적용한 것이다.

> 상하좌우 폭 1칸을 0으로 채웠다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-6.png">

(4,4) 크기에 패딩을 적용해서 (6,6)이 됐고, (3,3) 필터를 적용해서 (4,4) 크기의 출력 데이터를 얻었다.

패딩은 주로 ` 출력 데이터의 크기를 조정할 목적` 으로 사용한다. 

> 만약 (4,4) 입력에 (3,3) 필터를 적용하면 (2,2) 출력이 나온다. 
>
> 합성곱 연산을 여러번하는 신경망에서는 언젠가 크기(데이터 형상)가 1이 돼버리고, 합성곱 연산을 적용할 수 없게된다. 
>
> 이러한 현을 방지하기위해 패딩을 적용한다.

### 7.2.4 스트라이드

필터를 적용하는 위치 간격을 `스트라이드stride`라고 한다. 지금까지의 예시는 스트라이드가 모두 1이였다.

> 스트라이드가 2일 때 그림

<img src="7장_합성곱 신경망(CNN).assets/fig 7-7.png">

예시에서 스트라이드를 2로 적용 하니 출력이 (3,3)이 됐다. 이처럼 스트라이드를 키우면 출력 크기가 작아진다.

패딩을 키우면 출력 크기가 커지고, 스트라이드를 키우면 출력 크기가 작아진다.  이것을 수식화 하면 아래처럼 된다. 

> 입력 크기 (H,W) 필터 크기 (FH,FW) 출력 크기 (OH,OW) 패딩 P 스트라이드 S 

<img src="7장_합성곱 신경망(CNN).assets/e 7.1.png">

### 7.2.5 3차원 데이터의 합성곱 연산

지금까지는 2차원 데이터의 합성곱 연산이였다. 3차원 데이터의 합성곱 연산을 알아보자.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-8.png">

`채널`쪽으로 특징 맵이 여러 개라면 입력 데이터와 필터의 합성곱 연산을 채널마다 수행해서 결과를 더해서 하나의 출력을 얻는다.

3차원의 합성곱 연산에서 주의할 점은 `입력 데이터의 채널 수 = 필터의 채널 수`여야 한다는 것이다. 위 그림에서는 3개로 일치한다.

한편, 필터 자체의 크기(형상)는 원하는 값으로 설정할 수 있다.

### 7.2.6 블록으로 생각하기

<img src="7장_합성곱 신경망(CNN).assets/fig 7-10.png">

위의 그림은 3차원 합성곱 연산을 직육면체 블록으로 생각한 것이다.

예시에서는 `출력 데이터`는 한 장의 특집 맵이다. 즉, 채널이 1개인 특징 맵이다. 

만약 합성곱 연산의 출력으로 다수의 채널을 내보내려면 `필터`(가중치)를 다수 사용하면 된다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-11.png">

위 그림처럼 FN개의 필터를 적용하면 FN개의 출력 맵이 생성된다.

FN개의 맵을 모으면 (FN, OH, OW)의 형상인 블록이 완성되고, 이 블록을 다음 계층으로 넘기는 것이 CNN의 처리 흐름이다.

위 그림에서 보듯이 합성곱 연산에서는 `필터의 수`도 고려해야 한다.

그래서 필터의 가중치 데이터는 `4차원`데이터 이며 (출력 채널 수, 입력 채널 수, 높이, 너비) 순으로 쓴다. 

> 합성곱 연산에서의 편향

<img src="7장_합성곱 신경망(CNN).assets/fig 7-12.png">

`편향`은 채널 하나에 한 개의 값으로 이루어져있다. 위 그림에서 편향의 형상은 `(FN, 1, 1)`이고, 필터의 출력 형상은 항상 `(FN, OH, OW)` 이다.

`편향과 필터의 출력 블록`을 더하면` 편향의 각 값`이 `필터 출력 블록의 각 채널 원소`에 더해져서 출력 데이터를 완성한다.

### 7.2.7 배치 처리

합성곱 연산에서 배치 처리를 지원하고자 각 계층을 흐르는 데이터의 차원을 하나 늘려 `4차원`데이터로 저장한다.

구체적으로는 데이터를 `데이터 수, 채널 수, 높이, 너비`순으로 저장한다.

> 데이터가 N개일 때의 처리 흐름

<img src="7장_합성곱 신경망(CNN).assets/fig 7-13.png">

데이터는 `4차원의 형상`으로 각 계층을 타고 흐른다. 

여기서 주의할 것은 신경망에 4차원 데이터가 하나 흐를 때마다 데이터 `N`개에 대한 합성곱 연산이 이루어 진다는 것이다. 

즉, N회 분의 처리를 한 번에 수행한다는 뜻이다.

## 7.3 풀링 계층

`풀링`은 세로 가로 방향의 공간을 줄이는 연산이다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-14.png">

위 그림에서는 `2X2`영역을 원소 하나로 집약하여 공간 크기를 줄였다.

위 그림은  `2X2 최대 풀링`을 스트라이드 2로 처리한 순서이다.

`최대 풀링`은 영역에서 최댓값을 구하는 연산이고, `2X2`는 연산의 영역을 뜻한다.

즉, `2X2 최대 풀링`은 2X2 영역에서 최댓값을 구한다는 뜻이다.

스트라이드는 영역의 크기에 맞게 설정해야한다. 즉, 2X2 영역이라면 스트라이드는 2, 3X3 영역이라면 스트라이드는 3이 설정해야 한다.  

> 풀링은 평균 풀링도 존재한다. 다만, 일반적으로 최대 풀링을 많이 사용한다.

### 7.3.1 풀링 계층의 특징

__학습해야 할 매개변수가 없다.__

풀링은 영역에서 최댓값 혹은 평균을 취하는 처리이므로 학습할 필요가 없다.

__채널 수가 변하지 않는다__.

풀링 연산은 입력 데이터의 채널 수 그대로 출력데이터로 내보낸다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-15.png">

__입력의 변화에 영향을 적게 받는다.__

입력 데이터가 조금 변해도 풀링의 결과는 잘 변하지 않는다. 

<img src="7장_합성곱 신경망(CNN).assets/fig 7-16.png">

위 그림은 데이터가 오른쪽으로 1칸씩 이동했지만 결과는 변하지 않았다.

## 7.4 합성곱/풀링 계층 구현

### 7.4.1 4차원 배열

```python
import numpy as np

# 데이터가 10, 채널 1, 높이 28, 너비 28 데이터
x = np.random.rand(10, 1, 28, 28)

# 각 데이터의 형상은 동일하다.
print(x[0].shape) # 1 28 28
print(x[1].shape) # 1 28 28

# 첫 번째 데이터의 첫 채널의 공간 데이터에 접근방법
print(x[0, 0])
```

### 7.4.2 im2col로 데이터 전개하기

합성곱 연상을 구현하려면 `for`을 겹겹이 써야하지만 넘파이에서는 for 의 성능이 떨어진다.

for 대신 `im2col`이라는 편의 함수를 사용하는 방법을 구현호배자.

`im2col`은 입력 데이터를 필터링 하기 좋게 전개(펼치는) 하는 함수이다.

<img src="7장_합성곱 신경망(CNN).assets/fig 7-17.png">

그림 처럼 3차원 입력 데이터에 im2col을 적용하면 2차원 데이터로 바뀐다.(정확히는 배치 데이터까지 포함하여 4차원 데이터를 2차원으로 변환한다.)

<img src="7장_합성곱 신경망(CNN).assets/fig 7-18.png">

입력 데이터에서 `필터를 적용할 영역`(3차원 블록)을 한 줄로 늘어 놓는다.

그림에서는 스트라이드를 크게 잡아 필터의 적용 영역이 겹치지 않게 했지만, 실제 상황에서는 영역이 겹치는 경우가 대부분이다.

필터 적용 영역이 `겹치게 되면` im2col로 전개한 후의 원소 수가 원래 블록 원소 수보다 많아진다.  그래서 im2col은 `메모리를 더 많이 소비`한다는 단점이 있다.

하지만 행렬로 묶음으로써 선형 대수 라이브러를 활용해` 효율을 높일 수` 있다.

> 합성곱 연산의 필터 처리 상세 과정

<img src="7장_합성곱 신경망(CNN).assets/fig 7-19.png">

`입력 데이터`는 im2col로 전개하고, `필터`는 세로로 1열로 전개한다.

두 데이터의 햅렬 곱을 계산하고, 출력 데이터(2차원)을 변형한다.

### 7.4.3 합성곱 계층 구현

> im2col 코드

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩 값
    -------
    col : 2차원 배열
    """
    
    # 4차원 인풋데이터 형상
    N, C, H, W = input_data.shape
    # 출력데이터 높이 너비 계산
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
	
    # numpy.pad(array, pad_width, mode='constant', **kwargs)
    # 입력 데이터 패딩
    # [(0,0), (0,0), (pad, pad), (pad, pad)] -> 4차원 패딩
    # 순서데로 데이터 개수, 채널, 상하, 좌우(?) 패딩할 개수
    # mode : 값을 채우는 방식?constant : 상수로 채움, (default = 0)
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    
    # im2col 출력 결과물 저장 배열
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
	
    # 필터 순회
    for y in range(filter_h): 
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

```

> 실제 사용 결과

```python
import sys, os
import numpy as np

sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1, 3, 7, 7)  # 데이터  수, 채널 수, 높이, 너비
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # 9, 75

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # 90, 75
```

위의 코드에서는 두 가지 예를 보여주고 있다.

1. 배치 1, 채널 3, 높이 7, 너비7
2. 배치 10, 채널 3, 높이 7, 너비 7

두 경우 모두 원소수는 75개이다. 이 값은 필터의 원소 수와 동일하다.(채널 3, 높이 5, 너비 5)

> 합성곱(Colvolution) 계층 구현

```python
class Convolution:
    # 필터, 편향, 스트라이드, 패딩
    # 필터는 4차원 형상(FN, C, FH, FW)
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
	# 순전파
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # im2col로 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 필터를 reshape를 사용해 2차원 배열로 변환
        # reshpae(FN, -1) : (FN, C * FH * FW) 로 변환시켜준다.
        col_W = self.W.reshape(FN, -1).T
        # 입력 데이터와 필터 행렬 곱
        out = np.dot(col, col_W) + self.b
        # 행렬 곱 결과 reshape
        # transpose : 다차원 배열의 축 순서를 바꿔준다.
        # (N,H,W,C) -> (N,C,H,W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    # 역전파
    # Affine 계층의 역전파와 비슷하다.
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # 순전파에서 transpose 한것을 다시 돌린다.
        # (N,C,H,W) -> (0,H,W,C)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # im2col의 반대인 col2im 사용
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```

> transpose 함수 그림

<img src="7장_합성곱 신경망(CNN).assets/fig 7-20.png">

### 7.4.4 풀링 계층 구현

풀링 계층도 im2col을 사용해 입력 데이터를 전개한다. 단, 풀링의 경우엔 채널 쪽이 독립적이라는 점이 다르다.

> 2 X 2 풀링의 예

<img src="7장_합성곱 신경망(CNN).assets/fig 7-21.png">

<img src="7장_합성곱 신경망(CNN).assets/fig 7-22.png">

> 코드 구현

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    # 순전파
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 입력 데이터 im2col 적용(전개)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 역전파 때 사용할 최댓값 인덱스
        arg_max = np.argmax(col, axis=1)
        # 행별 최댓값 구하기
        out = np.max(col, axis=1)

        # 형상 reshape
        # 배열축 수정
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    # 역전파
    def backward(self, dout):
        # 배열축 복원
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[
            np.arange(self.arg_max.size), self.arg_max.flatten()
        ] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(
            dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad
        )

        return dx
```

## 7.5 CNN 구현

> 구현할 CNN 구성

<img src="7장_합성곱 신경망(CNN).assets/fig 7-23.png">

__초기화 때 받는 인수__

- input_dim : 입력 데이터(채널 수, 높이, 너비)의 차원

- conv_param : 합성곱 계층의 하이퍼파라미터(딕셔너리). 딕셔너리 정보는 아래와 같다
  - filter_num : 필터 수
  - filter_size : 필터 크기
  - stride : 스트라이드
  - pad : 패딩
  - hidden_size : 은닉층(완전연결)의 뉴런 수
  - output_size : 출력층(완전연결)의 뉴런 수
  - weight_init_std : 초기화 때의 가중치 표준편차

```python
import sys
import os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import *
from dataset.mnist import load_mnist
from common.trainer import Trainer


class SimpleConvNet:
    """
    CNN 네트워크 구성
    입력 데이터 -> [Conv -> ReLU -> Pooling] -> [Affine -> ReLU] -> [Affine -> Softmax] -> 출력
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        # 합성곱 계층의 하이퍼파라미터를 딕셔너리에서 꺼내서 저장
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        
        # 합성곱 계층의 출력 크기 계산
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        
        # 풀링 계층 출력 크기 계산
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 가중치 초기화
        self.params = {}
        # 1번째 층의 합성곱 계층 가중치와 편향
        # 배치 데이터 개수, 채널 수, 높이 너비
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)

        # 2번째 층의 완전연결 계층 가중치와 편향
        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)

        # 3번째 층의 완전연결 계층 가중치와 편향
        self.params["W3"] = weight_init_std * np.random.randn(
            hidden_size, output_size
        )
        self.params["b3"] = np.zeros(output_size)

        # CNN을 구성하는 계층을 생성
        self.layers = OrderedDict()
        # 1번째 층의 합성곱 계층
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        # 1번째 층의 활성화 함수 ReLU
        self.layers["Relu1"] = Relu()
        # 풀링 계층
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 2번째 층의 완전연결 계층
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        # 2번째 층의 활성화 함수 ReLU
        self.layers["Relu2"] = Relu()

        # 3번째 층의 완전연결 계층
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        # 출력 계층 소프트맥스 함수
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 추론 수행
        # 미리 초기화해놓은 계층을 앞에서부터 순전파 메서드 호출
        # 결과를 다음 계층으로 전달
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        # 손실함수 값 계산
        # 추론한 결과를 인수로 마지막층(출력계층 및 손실함수 계산)의 순전파 메서 호출
        # 처음부터 마지막 계층까지 순전파 처리
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # 오차역전파법으로 기울기 계산
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

# 최대 에포크 수
max_epochs = 20

# CNN 네트워크 클래스 생성
# 하이퍼파라미터 설정
network = SimpleConvNet(
    # 입력데이터 형상
    input_dim=(1, 28, 28),
    # 필터 개수, 사이즈, 패딩, 스트라이드
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    # 은닉층 개수
    hidden_size=100,
    # 출력층 개수
    output_size=10,
    weight_init_std=0.01,
)

# 학습 클래스 생성
# 네트워크 - cnn
# 최적화 방법 - Adam
# 하이퍼파라미터 - 학습율 : 0.001
# 평가 간격 - 1000epoch마다
trainer = Trainer(
    network,
    x_train,
    t_train,
    x_test,
    t_test,
    epochs=max_epochs,
    mini_batch_size=100,
    optimizer="Adam",
    optimizer_param={"lr": 0.001},
    evaluate_sample_num_per_epoch=1000,
)
# 학습시작
trainer.train()

# 그래프 그리기
markers = {"train": "o", "test": "s"}
# 최대 에포크 크기의 배열 생성
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
```

## 7.6 CNN 시각화하기

필터를 이미지로 나타내는 코드 [링크](https://github.com/youbeebee/deeplearning_from_scratch/blob/master/ch7.CNN/visualize_filter.py)

> 학습 전과 후의 1번째 층의 합성곱 계층 가중치

<img src="7장_합성곱 신경망(CNN).assets/fig 7-24.png">

학습 전 필터는 무작위로 초기화 했기 때문에 규칙이 없다.

학습 후 필터는 규칙성을 가진 이미지가 됐다.

학습 후의 필터는 에지(색상이 바뀌는 경계선)와 블롭(국소적으로 덩어리진 영역)등을 보고 있다.

예를들어 왼쪽 절반이 흰색이고, 오른쪽 절반이 검은색인 필터는 세로 방향의 에지에 반응하는 필터이다.

> 왼쪽 절반이 흰색이고, 오른쪽 절반이 검은색인 필터와 아래쪽 절반이 흰색이고 위쪽 절반이 검은색인 필터

<img src="7장_합성곱 신경망(CNN).assets/fig 7-25.png">

### 7.6.2 층 깊이에 따른 추출 정보 변화

딥러닝 시각화에 관한 연구에 따르면, 계층이 깊어질수록 추출되는 정보는 더 추상화된다는 것을 알 수 있다.

> 일반 사물 인식을 수행한 8츨의 CNN(AlexNet)

<img src="7장_합성곱 신경망(CNN).assets/fig 7-26.png">

위 그림의 네트워크는 합성곱 계층과 풀링 계층을 여러 겹 쌓고 마지막으로 완전연결 계층을 거쳐 결과를 출력하는 구조이다.

위 그림처럼 합성곱 계층을 여러 겹 쌓아서 층이 깊어 짚어지면서 더 복잡하고 추상화된 정보가 추출된다.

`단순한 에지 -> 텍스처 -> 사물의 일부 `에 반응하도록 변화하고 있다.

