# 6장 게이트가 추가된 RNN [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

5장의 RNN은 시계열 데이터에서 시간적으로 멀리 떨어진, 장기 의존 관계를 잘 학슬할 수없어서 성능이 떨어진다.

> 멀리 떨어질 수록 영향력이 떨어진다는 의미인듯

그래서 `게이트gate`라는 구조를 추가한 `LSTM` 혹은 `GRU`가 등장했다.

## 6.1 RNN의 문제점

### 6.1.1 RNN 복습

### 6.1.2 기울기 소실 또는 기울기 폭발

<img src="assets/6장 게이트가 추가된 RNN/fig 6-3.png">

> RNNLM 기울기 흐름

<img src="assets/6장 게이트가 추가된 RNN/fig 6-4.png">

위 예시와 그림에서 마지막 단어인 `?`에서 첫번째 단어`Tom`까지 역전파로 기울기를 전달하고 있다.

그런데 만약 중간에 기울기가 작아지거나`기울기 소실` 혹은 커질수있다.`기울기 폭발` 

✅ 이런 경우 가중치는 올바르게 갱신될 수 없고, `장기 의존 관계`를 학습할 수 없게 된다.

### 6.1.3 기울기 소실과 기울기 폭발의 원인

 <img src="assets/6장 게이트가 추가된 RNN/fig 6-5.png">

상류에서 내려온 기울기는 `tanh`,`+`,`MatMul` 연산을 통과하는데 `tanh`와 `MatMul`이 기울기를 변화시킨다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-6.png">

✅ `tanh`의 점선 그래프를 보면  0에서 가까울수록 커지고, 멀어질수록 작아진다. 

그런데 최댓값이 1.0 이기 때문에 기울기는 `tanh`를 통과할 때 마다 작아질 수 밖에 없다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-7.png">

✅ `MatMul` 계층은 지수적으로 증가시키거나 지수적으로 감소시킨다.

왜냐하면 `MatMul`의 역전파는 dhW~h~^T^ 로 기울기를 계산하기 때문이다. 

즉 계속해서 가중치 W~h~ 를 곱하기 때문에 W~h~가 1보다 크면 증가하고, 1보다 작으면 감소한다.

> 테스트 코드와 그래프

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch06/rnn_gradient_graph.py

import numpy as np
import matplotlib.pyplot as plt


N = 2   # 미니배치 크기
H = 3   # 은닉 상태 벡터의 차원 수
T = 20  # 시계열 데이터의 길이

dh = np.ones((N, H))

np.random.seed(3) # 재현할 수 있도록 난수의 시드 고정

Wh = np.random.randn(H, H)
#Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# 그래프 그리기
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()
```

<img src="assets/6장 게이트가 추가된 RNN/fig 6-8.png">

<img src="assets/6장 게이트가 추가된 RNN/fig 6-9.png">

### 6.1.4 기울기 폭발 대책

`기울기 클리핑 gradients clipping`라는 기법으로 기울기 폭발을 막을 수 있다.

> 기울기 클리핑 의사코드

<img src='assets/6장 게이트가 추가된 RNN/e 6-0.png'>

✅만약 기울기의 L2 노름이 임계값`문턱값`을 초과하면 수정 수식으로 기울기을 수정한다.

>여기서 g^^^ 는 모든 기울기를 하나로 모은것이다. 즉 2개의 가중치 W1 W2가 있다면 기울기 dW1 dW2를 결합한 것을 g^^^ 로 한다.

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch06/clip_grads.py

import numpy as np


dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0


def clip_grads(grads, max_norm):
    total_norm = 0
    # 전체 기울기 합
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
	
    # 임계치
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


print('before:', dW1.flatten())
clip_grads(grads, max_norm)
print('after:', dW1.flatten())
```

## 6.2 기울기 소실과 LSTM

`RNN`에서 `기울기 소실`을 해결하기 위해 등장 한것이 `게이트가 추가된 RNN`이다.

### 6.2.1 LSTM의 인터페이스

<img src="assets/6장 게이트가 추가된 RNN/fig 6-11.png">

✅`LSTM` 계층에는 `c`라는 새로운 경로가 생겼다. c 를 `기억셀 or 셀`이라하고, LSTM 전용의 기억 메커니즘이다.

`기억 셀`의 특징은 데이터를 자기 자신으로만 주고받는다는 것이다.

즉, LSTM 계층내에서 완결되고, 다른 계층으로는 출력하지 않는다.

### 6.2.2 LSTM 계층 조립하기

✅기억 셀 c~t~ 에는 과거부터 t까지의 필요한 모든 저장돼 있고, 이 정보를 바탕으로 외부 계층에 h~t~를 출력한다.

이때 h~t~는 기억 셀의 값을 tanh 함수로 변환한 값이다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-12.png">

위 그림에서 기억셀 c~t~는 입력 (c~t-1~,h~t-1~,x~t~) 로 부터 구할 수 있고, 핵심은 c~t~를 이용해서 은닉 상태 h~t~ 를 계산한다는 것이다.

> 게이트의 역할 : 데이터 흐름을 제어한다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-13.png">

<img src="assets/6장 게이트가 추가된 RNN/fig 6-14.png">

✅ 중요한 것은 `게이트의 여는 정도`도 데이터로 부터 학습한다는 것이다.



### 6.2.3 output 게이트

h~t~ = tanh(c~t~) 인데, 이때, tanh(c~t~)에 **`게이트`**를 적용한다는 것은 
각 원소에 대해**`다음 시각의 은닉 상태에 얼마나 중요한가`**를 조정한다는 뜻이다.

이때, 이 게이트에 대해 다음 은닉 상태의 출력을 담당하는 게이트 이므로 **`ouput 게이트`** 라고 한다.

---

output 게이트의 열림 상태는 입력 x~t~와 이전 은닉 상태h~t-1~로 부터 구한다.

> **O** : output 게이트, **σ()** : 시그모이드 함수, x~t~ : 입력, h~t-1~ : 이전 은닉상태, W : 가중치, b : 편향

<img src="assets/6장 게이트가 추가된 RNN/e 6-1.png">

이 **O**와 **tanh(c~t~)** 의 원소별 곱을 h~t~로 출력한다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-15.png">

✅ 위 그림처럼 output 게이트에서 수행하는 계산을 **σ**로 표기하고, 출력을 **O** 라고하면 **h~t~**는 **tanh(c~t~)**의 곱으로 계산된다. 여기서 말하는 곱은 원소별 곱이며, 이것을 **아다마르 곱**이라고도 한다.

아다마르 곱은 기호로 **Θ** 나타타낸다.

<img src='assets/6장 게이트가 추가된 RNN/e 6-2.png'>

### 6.2.4 forget 게이트

✅ 기억 셀을 기억만 하는 것이 아니라 불필요한 기억을 잊게 할 수도 있다.

c~t-1~의 기억 중에서 한 기억을 잊게 해주는 게이트를 **forget 게이트**라고 한다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-16.png">

> forget 게이트의 수식

<img src="assets/6장 게이트가 추가된 RNN/e 6-3.png">

forget 게이트의 출력 **f**와 이전 기억 셀 **c~t-1~**을 원소별로 곱해서 기억셀 **c~t~**를 구한다.

즉 **c~t~ = f Θ c~t-1~**이다.

### 6.2.5 새로운 기억 셀

✅ 새로 기억해야할 정보를 기억 셀에 추가하기 위한 게이트가 필요하다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-17.png">

<img src="assets/6장 게이트가 추가된 RNN/e 6-4.png">

ouput 게이트와 forget 게이트와 다른점은 시그모이드 함수가 아닌 **tanh 함수**를 사용했다는 점이다.

#### **c~t~ = f Θ c~t-1~ + g **

### 6.2.6 input 게이트

✅**input 게이트**는 g의 각 원소가 새로 추가되는 정보로써의 가치의 크기를 판단한다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-18.png">

<img src="assets/6장 게이트가 추가된 RNN/e 6-5.png">

#### **c~t~ = f Θ c~t-1~ + g Θ i **

### 6.2.7 LSTM의 기울기 흐름

<img src="assets/6장 게이트가 추가된 RNN/fig 6-19.png">

기억 셀의 역전파에는 **+, x** 만 존재한다.

**+**는 값을 그대로 전달하기 때문에 기울기의 변화가 없다.

✅ **x**는 아다마르 곱(원소별 곱)을 계산하기 때문에 매 시각 다른 게이트 값을 이용해서 곱을 계산하기 때문에 곱셈의 효과가 누적 되지 않아 기울기 소실이 발생하지 않는다.

**x** 노드는 forget 게이트가 제어하는데

**`잊어야 한다`**고 판단한 기억 셀의 원소에 대해서는 기울기가 작아지고,

**`잊어서는 안 된다`**고 판단한 원소에 대해서는 기울기가 약화되지 않고 과거로 전해진다.

## 6.3 LSTM 구현

> LSTM의 계산 수식

<img src="assets/6장 게이트가 추가된 RNN/e 6-6.png">

<img src="assets/6장 게이트가 추가된 RNN/e 6-7.png">

<img src="assets/6장 게이트가 추가된 RNN/e 6-8.png">

위 식중 각 게이트의 계산 **`f, g, i, o`**를 하나의 식으로 정리할 수 있다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-20.png">

위 그림처럼 4개의 가중치와 편향을 하나로 모아서 한 번에 끝낼 수 있다.

✅ 행렬 라이브러리는 큰 행렬을 한번에 계산할 때 효율적이기 때문에 네 번 계산하는 것보다 한 번 계산하는게 더 빠르다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-21.png">

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/time_layers.py

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 입력 x에 대한 4개분의 가중치
        Wh: 은닉 상태 h에 대한 4개분의 가중치
        b: 4개분의 편향
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        
        # 순전파 중간 결과 보관 변수
        self.cache = None
        
    
    # 입력, 이전 은닉상태, 이전 기억 셀
	def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        # 미니배치 수, 은닉상태와 기억셀의 차원 수
        N, H = h_prev.shape
        
        
		# affine 계층 계산 결과(4회분)
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
	
    	# 각 게이트 계산 슬라이스
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        # 각 게이트 출력
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
		
        # 다음 기억셀과 다음 은닉상태
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
```

## 6.5 RNNLM 추가 개선

### 6.5.1 LSTM 계층 다층화

✅`LSTM` 계층을 여려겹 쌓아서 정확도를 올릴 수 있다.

첫 번째 LSTM 계층의 은닉상태가 두 번째 LSTM에 입력되는 것을 알 수 있다.

이와 같은 방식으로 여러 층을 쌓을 수 있고, 더 복잡한 패턴을 학습할 수 있다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-29.png">

### 6.5.2 드롭아웃에 의한 과적합 억제

✅층을 깊게 쌓아서 정확도를 올릴 수 있지만 `과적합`을 일으킬 수도 있다.

특히, RNN은 일반적인 피드포워드보다 쉽게 과적합을 일으킨다.

과적합을 억제하기 위한 방법으로는 `데이터 양 늘리기`, `모델의 복잡도 줄이기`, `복잡도에 페널티를 주는 정규화`,`드롭아웃`이 있다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-30.png">

✅ RNN에 드롭아웃 계층을 넣는 좋은 방법은 **깊이 방향(상하 방향)** 으로 넣는 것이다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-33.png">

이렇게하면 시간 방향(좌우 방향)에는 영향을 주지않고, 깊이 방향에만 영향을 준다.

---

최근에는 시간 방향과 깊이 방향 모두 드롭아웃을 적용하는 **변형 드롭아웃**도 만들어 졌다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-34.png">

같은 계층에 속한 드롭아웃들은 **마스크**를 공유한다. 

그림에서는 색이 같은 드롭아웃 계층들은 같은 마스크를 이용한다.

 마스크를 공유함으로써 정보를 잃게 되는 방법이 고정되므로, 정보가 지수적으로 손실되는 사태를 피할 수 있다.

### 6.5.3 가중치 공유

**`가중치 공유`**는 언어모델을 개선하는 아주 간단한 트릭이다.

<img src="assets/6장 게이트가 추가된 RNN/fig 6-35.png">

그림에서는 Embedding 계층과 Affine 계층이 가중치를 공유하고 있다.

두 계층이 가중치를 공유함으로써 학습해야할 가중치 수가 크게 줄어들고, 정확도도 향상된다.

