``

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

`네거티브 샘플링`을 이용해서 `은닉층 이후 처리`의 병목을 해결하자.

### 4.2.1 은닉층 이후 계산의 문제점

<img src="assets/4장 word2vec 속도 개선/fig 4-6.png">

`은닉층 이후`에서 계산이 오래 걸리는 곳은 두 곳이다.

1. 은닉층의 뉴런과 가중치 행렬($W~out~$) 의 곱
2. Softmax 계층의 계산

첫 번째는 거대한 행렬을 곱하는 문제이다. 위 그림에서는 은닉층의 뉴런 벡터 크기는`100`이고, 가중치 행렬의 크기는 `100 X 100만`이다. 이 두 행렬을 곱하기 위해서는 많은 시간이 소모된다.

두 번째는 Softmax 계층은 어휘가 많아질수록 계산량이 많아진다는 점이다.

<img src="assets/4장 word2vec 속도 개선/e 4-1.png">

위 식은 k번째 단어를 타깃으로 했을 때의 Softmax 계산이다.

분모 값을 얻으려면 exp 계산을 `100만`번 수행해야한다.

### 4.2.2 다중 분류에서 이진 분류로

네거티브 샘플링의 핵심 아이디어는 `이진 분류`에 있다.

✅정확하게는 `다중 분류`를 `이진 분류`로 근사하는 것이 중요한 포인트이다.

지금까지는 `100만 개의 단어 중`에서 `하나`의 옳은 단어를 선택하는 문제였다.

이 문제를 `타깃 단어가 X 인가?`에 대한 질문에 `Yes/No`로 대답할 수 있는 신경망을 생각해야한다.

<img src="assets/4장 word2vec 속도 개선/fig 4-7.png">

`이진 분류`로 접근하면 신경망 계층을 위 그림처럼 나타낼 수 있다.

은닉층과 출력 층의 $W~out~$의 내적은

✅ `하나의 특정 단어 벡터`만 추출하고,  그 벡터와 은닉층 뉴런의 내적을 계산하면 된다.

> 그림에서의 특정 단어는  `say`

<img src="assets/4장 word2vec 속도 개선/fig 4-8.png">

위 그림처럼 $W~out~$에는 각 단어 ID의 단어 벡터가 각각의 열로 저장되어 있다.

`say`에 대한 단어 벡터를 추출해서 은닉층 뉴런과의 내적을 구하면 `say`에 대한 최종 점수인 것 이다.

### 4.2.3 시그모이드 함수와 교차 엔트로피 오차

`이진 분류`를 신경망으로 해결할려면 `시그모이드 함수`를 적용해 확률로 변환하고, `교차 엔트로피 오차`를 손실 함수로 사용한다.

> 시그모이드 공식과 계층, 그래프

<img src="assets/4장 word2vec 속도 개선/e 4-2.png">

<img src="assets/4장 word2vec 속도 개선/fig 4-9.png">

✅`시그모이드 함수`를 적용해 `확률 y`를 얻으면, 이 `교차 에트로피 오차`를 사용해 확률 y로부터 손실을 구한다.

> 교차 엔트로피 오차 공식
>
> y : 시그모이드 함수 출력, t : 정답 레이블(1 또는 0)

<img src="assets/4장 word2vec 속도 개선/e 4-3-1616505942043.png">

<img src="assets/4장 word2vec 속도 개선/fig 4-10-1616505959526.png">



위 그림에서 주목할 것은 `역전파의 y-t`이다.

만약, 정답 레이블`t`이 1이라면, 확률`y`가 1에 가까워질수록 오차가 줄어든다는 뜻이다.

반대로 확률`y`가 1로부터 멀어지면 오차가 커진다.

오차가 `작다면 작게` 학습하고, `크다면 크게` 학습할 것이다.

### 4.2.4 다중 분류에서 이진 분류로 구현

> 다중 분류망과 이진 분류망

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch04/negative_sampling_layer.py
import numpy as np
import collections 
    
    
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        # 가중치 행렬에서 단어 id에 해당하는 행 추출
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 
        np.add.at(dW, self.idx, dout)
        return None
    

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        
        # 시그모이드 함수 결과
        self.y = 1 / (1 + np.exp(-x))
		
        # 크로스 엔트로피 오차
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx
    
    
class EmbeddingDot:
    def __init__(self, W):
        # 임베디드 계층
        self.embed = Embedding(W)
        # 가중치
        self.params = self.embed.params
        # 기울기
        self.grads = self.embed.grads
        # 순전파 계산 결과 임시 저장 변수
        self.cache = None
	
    # h : 은닉층 뉴런, idx : 타겟 단어ID 배열(미니배치)
    def forward(self, h, idx):
        # 타겟 단어
        target_W = self.embed.forward(idx)
        # 내적 계산 및 행 합계 게산
        out = np.sum(target_W * h, axis=1)
        
		# 계산 결과 임시 저장
        self.cache = (h, target_W)
        return out
    

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

```

<img src="assets/4장 word2vec 속도 개선/fig 4-14.png">

### 4.2.5 네거티브 샘플링

지금까지는 긍정적인 예(정답)에 대해서만 학습했다. 그렇다면 부정적인 예(오답)을 입력하면 어떤 결과가 나올까?

<img src="assets/4장 word2vec 속도 개선/fig 4-16.png">

위 그림처럼 부정적인 예를 입력했을 때 출력이 `0`에 가깝게 해주는 `가중치`가 필요하다.

✅하지만 모든 부정적인 예를 대상으로 이진 분류를 학습시킬 수는 없다. 이진분류를 사용하는 의미가 사라지기 때문이다.

그래서 근사적인 해법으로 부정적 예를 몇 개만 선택한다. 이것이 바로 `네거티브 샘플링`기법의 의미이다.

✅`네거티브 샘플링`은 긍정적 예에 대한 손실을 구하고, 몇 개의 부정적 예에 대한 손실을 구해서 더한 값을 최종 손실`긍적적 예의 손실 + 부적적 예에 대한 손실 합`로 정한다.

 <img src="assets/4장 word2vec 속도 개선/fig 4-17.png" style="zoom:50%;" >

긍적적 예`say`와 부정적 예 `hello, I`의 손실을 모두 더해서 최종손실로 사용하고 있다.

### 4.2.6 네거티브 샘플링의 샘플링 기법

✅`네거티브 샘플링`에서 부정적 예를 샘플링하는 좋은 방법은 말뭉치의 통계 데이터를 기초로 샘플링 하는 것이다.  

말뭉치에서 등장 빈도가 높은 단어를 많이 추출하고, 반대의 경우에는 적게 추출하는 것이다.

<img src="assets/4장 word2vec 속도 개선/fig 4-18.png">

> 샘플링 구현 예

<img src="assets/4장 word2vec 속도 개선/image-20210323223828065.png">

> 샘플링 공식, 기본 확률분포에 0.75 를 제곱한다

<img src="assets/4장 word2vec 속도 개선/e 4-4.png">

위 식처럼 0.75 를 제곱하는 이유는 `확률이 낮은 단어`의 확률을 살쩍 높히기 위해서이다.

 <img src="assets/4장 word2vec 속도 개선/image-20210323224124028.png">

위 예에서 보이는 것처럼 0.75 를 제곱함으로써 확률이 `낮은 단어는 높아졌고`, `높은 단어는 낮아졌다.`

### 4.2.7 네거티브 샘플링 구현

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch04/negative_sampling_layer.py
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        # 단어 등장 빈도수 계산
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        
        # 등장 확률 계산
        self.word_p = np.zeros(vocab_size) # 등장 확률 저장 변수
        for i in range(vocab_size):
            self.word_p[i] = counts[i] 
            
		
        self.word_p = np.power(self.word_p, power) # 분자 계산
        self.word_p /= np.sum(self.word_p) # 분모 계산
	
    # target id를 제외하고, 네거티브 샘플링을 한다.
    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                # target의 확률을 0으로 한다.
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        # 샘플링 사이즈
        self.sample_size = sample_size
        # 샘플링
        self.sampler = UnigramSampler(corpus, power, sample_size)
        # 출력함수 및 손실함수 저장 리스트
        # self.loss_layers[0] -> 긍정적 예(타깃)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        # 임베디드 계층 저장 리스트
        # self.embed_dot_layers[0] -> 긍정적 예(타깃)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        # 네거티브 샘플링 단어
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적 예 순전파
        # 임베디드 계층 순전파 -> 점수 출력
        score = self.embed_dot_layers[0].forward(h, target)
        # 긍정적 예 정답 레이블 -> 1
        correct_label = np.ones(batch_size, dtype=np.int32)
        # 손실함수
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        # 부정적 예 정답 레이블 -> 0
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            # 임베디드 계층 순전파 -> 점수 출력
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        # 역전파 계층을 반대로 수행
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh
```

## 4.3 개선판 word2vec 학습

 ### 4.3.1 CBOW 모델 구현

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch04/cbow.py
import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    # 어휘수, 은닉층 뉴런 수, 윈도우 사이즈, 단어 ID 목록
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer) # 계층 모으기
            
       	# 네거티브 샘플링 계층 사용
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in
	
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
```

