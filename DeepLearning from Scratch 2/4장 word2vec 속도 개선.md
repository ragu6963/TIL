

# 4장 word2vec 속도 개선 [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

👍앞장의 `word2vec`에 두 가지 개선을 추가해서 속도를 개선해보자

1. `Embedding` 이라는 계층을 도입한다.
2. `네거티브 샘플링`이라는 새로운 손실 함수를 도입한다.

## 4.1 word2vec 개선 1

`CBOW`는 거대한 말뭉치를 다루게 되면 몇 가지 문제가 발생한다.

<img src="assets/4장 word2vec 속도 개선/fig 4-2.png">

위 그림에서는 입력층과 출력층에 각 100만 개의 뉴런이 존재한다. 많은 뉴런 때문에 중간 계산에 `많은 시간`이 소요된다.

정확히는 다음의 두 계산이 병목이 된다.

1. 입력층의 원핫 표현과 가중치 행렬 $W~in$  의 곱 계산
2. 은닉층과 가중치 행렬 $W~out$의 곱 및 Softmax 계층의 계산

첫 번째는 입력층의 원핫 표현의 문제이다. 

어휘가 100만 개라면 한 어휘의 원핫 표현의 원소 수가 100만 개의 벡터가 된다. 상당한 메모리를 차지하고, 이 원핫 벡터와 $W~in$을 곱하면 `계산 자원을 상당히 소모`한다.

두 번째는 은닉층 이후의 계산이다.

마찬가지로 은닉층과 $W~out$의 곱의 계산량이 아주 많고, Softmax 계층에서도 계산량이 증가한다.

### 4.1.1 Embedding 계층

앞 장의 word2vec 에서는 `단어의 원핫 표현`과 `가중치 행렬`과 곱했다.

만약 어휘 수가 100만개 은닉층 뉴런이 100개라면 아래 그림과 같은 행렬곱이 발생한다.

<img src="assets/4장 word2vec 속도 개선/fig 4-3.png">

그런데 `행렬곱`이 하는 일은 단지 `가중치 행렬`에서 `특정 행`을 추출할 뿐이다. 그러므로 `행렬곱`은 사실 필요가 없다.

이처럼 `단어 ID에 해당하는 특정 행 추출`을 하는 계층을 `Embedding 계층`이라고 부른다.

### 4.1.2 Embedding 계층 구현 

```python
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        # 가중치 행렬
        W, = self.params
        
        # 단어 id
        self.idx = idx
        
        # 가중치 행렬에서 단어 id에 해당하는 행 추출
        out = W[idx]
        return out

    def backward(self, dout):
        # 가중치 기울기 행렬
        dW, = self.grads
        
        # 전체 원소 값 0으로
        dW[...] = 0
        
        # 가중치 기울기 행렬의 특정 행(단어 id)에 기울기 dout을 할당
        np.add.at(dW, self.idx, dout)
        return None
```

<img src="assets/4장 word2vec 속도 개선/fig 4-4.png">

#### 순전파

`가중치 행렬(W)`에서  `단어 ID(idx)`에 해당하는 특정 행을 추출한 후, 다음 층으로 전달한다.

#### 역전파

`가중치 기울기 행렬(dW)` 의 원소를 모두 0으로 초기화하고, `특정 행(idx)`에  이전 층에서 온 `기울기(dout)`을 `더해준다`.

> 더해주는 이유는 중복 idx가 있을 때 먼저 나온 idx 행의 값을 덮어쓰게 되고, 기울기 소실이 발생한다.

<img src="assets/4장 word2vec 속도 개선/fig 4-5.png">

## 4.2 word2vec 개선 2