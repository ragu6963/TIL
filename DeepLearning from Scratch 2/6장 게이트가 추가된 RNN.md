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

