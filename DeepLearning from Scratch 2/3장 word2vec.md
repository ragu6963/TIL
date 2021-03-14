

# 3장 word2vec [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

## 3.1 추론 기반 기법과 신경망

### 3.1.1 통계 기반 기법의 문제점

현업에서 다루는 말뭉치의 어휘 수는 100만7을 훌쩍 넘는다고 한다.

어휘가 100만 개라면, `통계 기반 기법`에서는 `100만 X 100만` 이라는 거대한 행렬이 만들어 진다.

이는 현실적으로 `불가능`하다.

💡 반면에 `추론 기반 기법`은 신경망을 이용해서 미니배치로 학습하는 것이 일반적이기 때문에 말뭉치가 거대하더라도 학습시킬 수 있다.

게다가 여러 머신과 여러 GPU를 이용한 병렬 계산으로 학습 속도도 높일 수 있다.

<img src="assets/fig 3-1.png">

### 3.1.2 추론 기밥 기법 개요

💡 추론 기반 기법에서는 `추론`이 주된 작업이다.

`추론`이란 아래 그림처럼 맥락이 주어졌을 때 "?"에 들어갈 단어를 추측하는 작업이다.

<img src="assets/fig 3-2.png">

---

아래 그림처럼 추론 기반 기법에는 어떤 모델이 등장한다. 이 모델로 신경망을 사용한다.

모델은 맥랑 정보를 입력받으면 단어의 출현 확률을 보여준다.

<img src="assets/fig 3-3.png">

### 3.1.3 신경망에서의 단어처리

신경망은 "you"나 "say" 같은 단어를 있는 그대로 처리할 수 없기 때문에 `고정 길이 벡터`로 변환해야 한다.

💡대표적인 방법이 단어를 `원핫 벡터`로 변환하는 것이다.

아래 그림처럼 원핫 벡터는 벡터의 원소 중 하나만 1이고 나머지는 0인 벡터를 말한다.

<img src="assets/fig 3-4.png">

---

이처럼 단어를 고정 길이 벡터로 변환하면 신경망의 입력층은 아래 그림처럼 뉴런의 수를 `고정`할 수 있다.

<img src="assets/fig 3-5.png">

위 그림에서 입력층의 뉴런은 총 7개이고, 7개의 뉴런은 7개의 각각의 단어와 대응한다.

---

💡 신경망의 계층은 `벡터`를 처리할 수 있고, 단어를 `벡터`로 나타낼 수 있기 때문에 신경망을 통해 단어를 처리할 수 있다는 뜻이다.

즉, 아래 그림처럼 완전연결계층을 만들 수 있다.

<img src="assets/fig 3-6.png">

```python
# 완전연결계층 구현

import numpy as np

# 입력
c = np.array([[1, 0, 0, 0, 0, 0, 0]])
# 가중치
W = np.random.randn(7, 3) 
# 행렬곱
h = np.matmul(c, W) 
print(h) # [[0.42004167 0.53941578 0.31298182]]
```

위의 코드에서 c는 `원핫 표현`으로 단어 ID에 대응하는 원소만 1이고, 나머지는 0이다.

그래서 c 와 W의 행렬곱에서 하나의 행만 사용하고, 나머지는 사용하지 않는 것과 같다.

> 위 코드는 MatMul 계층으로 대체할 수 있다.

```python
import numpy as np

# MatMul 계층
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        (W,) = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = np.matmul(c)
print(h)
```



<img src="assets/fig 3-8.png">





## 3.2 단순한 word2vec

word2vec에서 제안하는 `CBOW` 모델을 사용해 신경망을 구축해보자.

### 3.2.1 CBOW 모델의 추론 처리

💡 CBOW`continuous bag of words` 모델은 맥락`주변단어`으로부터 타깃`중앙 단어`을 추측하는 신경망이다.

CBOW 모델의 입력은 "you"와 "goodbye" 같은 맥락이다.

<img src="assets/fig 3-9.png">

CBOW 모델의 입력층은 2개가 있고, 은닉층을 거쳐 출력층에 도달한다.

💡입력층에서 은닉층으로의 변환과 은닉층에서 출력층 뉴런으로의 변환은 `완전연결계층`이 처리한다.

> 위 그림에서 입력층이 2개인 이유는 맥락으로 사용할 단어를 2개로 정했기 때문이다. 만약 N개를 사용하면 입력층이 N개가 된다.

---

위의 그림에서 은닉층의 뉴런은 입력층의 완전연결계층에 의해 변환된 값이다.

만약 입력층이 여러 개이면 전체를 평균을 한다.

> 예를들어 입력층 두 개가 변화한 값 `h1, h2`가 있다면 은닉층 뉴런은 `(h1 + h2) / 2 `가 된다.

---

출력층의 뉴런은 7개인데 이 뉴런 하나 하나가 각 단어에 대응한다.

💡뉴런의 각 값은 단어에 대한 `점수(확률)`를 뜻하며, 값이 클수록 단어의 출현 확률도 높아진다. 

---

위 그림에서 입력층에서 은닉층으로의 변환은 완전연결계층에 의해서 이뤄지는데 가중이 `W_in`은 7 X 3 행렬이며 이 가중치가 `단어의 분산 표현`이다.

<img src="assets/fig 3-10.png">

💡가중치 W_in의 각행에는 해당` 단어의 분산 표현`이 저장돼 있다고 볼 수 있다.

따라서 학습을 할수록 맥락에서 출현하는 단어를 잘 추측하는 방향으로 이 분산 표현들이 갱신된다.

> 분산 표현 : RGB = (255,123,0) 처럼 단어를 표현하는 것

---

<img src="assets/fig 3-11.png">위 그림은 CBOW 를 계층 관점에서 본 그림이다.

모델의 가장 앞단에는 2개의 MatMul 계층`행렬곱`이 있고, 다음으로 두 계층이 더한 후 0.5를 곱해서 평균을 구해준다. 

마지막 출력층에서 다른 MatMul 계층이 적용되어 `점수(확률)`을 구한다.

---

```python
import sys

sys.path.append("..")
import numpy as np
from common.layers import MatMul


# 샘플 맥락 데이터
# 입력 2개
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# MatMul 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
# 두 개의 입력에 대한 행렬곱 순전파 값
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)

# 평균 계산
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)
#[[ 0.61767154 -0.07400862 -0.00912255  0.67955864 -0.18225631 -0.10688905 -0.71444076]]
```

### 3.2.2 CBOW 모델의 학습

💡CBOW 모델을 통해 얻은 점수에 소프트 맥스 함수를 적용하면 `확률`을 얻을 수 있다.

이 `확률`은 맥락(전후 단어)가 주어졌을 때 중앙에 어떤 단어가 출현할지에 대한 확률이다.

<img src="assets/fig 3-12.png">

위의 그림에서 맥락은 `you`와 `goodbye`이고, 정답레이블(단어)은 `say`이다.

---

CBOW 의 학습은 올바른 예측을 할 수 있도록 가중치를 조정한다. 그 결과로 가중치 W 에 단어의 출현 패턴을 파악한 벡터가 학습된다. 

>CBOW 모델은 단어 출현 패턴을 사용한 말뭉치로부터 배운다. 말뭉치가 다르면 얻게되는 분산 표현도 달라진다는 의미이다.
>
>즉, 스포츠 기사를 사용한 경우와 경제 기사를 사용했을 때 얻게되는 분산 표현이 크게 다를 것이다.

CBOW 모델은 `다중 클래스 분류`를 수행하는 신경망이다.

이 신경망을 학습하려면 `소프트맥스`와 `교차 엔트로피 오차`만 이용하면 된다.

소프트맥스는 `점수를 확률`로 변환하고, 확률과 정답 레이블로부터 교차 엔트로피 오차를 구한 후 `손실함수 값`을 사용해 학습을 진행한다.

<img src="assets/fig 3-13.png">

> `Softmax`와 `Cross Entropy Error` 계층을 합친 모습

<img src="assets/fig 3-14.png">

### 3.2.3 word2vec의 가중치와 분산표현

word2vec에는 두 가지 가중치가 있다.

`입력 층 완전연결계층`의 가중치와 `출력 층 완전연결계층`의 가중치 이다.

- 입력 측 가중치의 각 행이 각 단어의 분사 표현에 해당한다.

- 출력 측 가중치에는 단어의 의미가 인코딩된 벡터가 저장 되어있다. 다만, 각 닫어의 분산 표현이 열 방향으로 저장 된다.

---

여기서 분산 표현으로 가중치를 선택하는 방법이 세 가지가 있다.

A. 입력 측의 가중치만 이용

B. 출력 층의 가중치만 이용

C. 둘다 이용

word2vec은 두 가지 가중치 중 `입력 측의 가중치`만 이용하는게 가장 대중적인 선택이다.

## 3.3 학습 데이터 준비

### 3.3.1 맥락과 타깃

💡word2vec 신경망의 입력은 `맥락`이고, 정답 레이블은 맥락에 둘러싸인 중앙 단어`타깃`이다.

<img src="assets/fig 3-16.png">

> 단어 ID  목록 생성

```python
import numpy as np

# 말뭉치 텍스트 단어 ID로 변환 함수
def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)  # [0 1 2 3 4 1 5 6]
print(id_to_word) 
# {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
```

<img src="assets/fig 3-17.png">

> 단어 ID 목록 -> 맥락과 타깃을 반환하는 함수

```python
def create_contexts_target(corpus, window_size=1):
    """
    맥락과 타깃 생성
    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    """
    # 좌우 윈도우 사이즈만큼 뺀 데이터
    # 예) window_size = 1, corpus[1:-1] -> 두 번째부터 뒤에서 두 번째까지 데이터
    target = corpus[window_size:-window_size]

    # 전체 맥락 저장 리스트
    contexts = []

    # 좌 우 윈도우크기 만큼 뺀만큼 인덱스 순회
    for idx in range(window_size, len(corpus) - window_size):
        # 타깃 단어에 대한 맥락 리스트
        cs = []
        # 좌 우 윈도우 크기만큼 탐색
        for t in range(-window_size, window_size + 1):
            # t == 0 -> 타깃 단어
            if t == 0:
                continue
                
            # 타깃단어에 대한 맥락 리스트에 데이터 추가
            cs.append(corpus[idx + t])

        # 전체 맥락 리스트에 데이터 추가
        contexts.append(cs)

    return np.array(contexts), np.array(target)


conexts, target = create_contexts_target(corpus, 1)
print(conexts)
"""
맥락 목록
[[0 2]
[1 3]
[2 4]
[3 1]
[4 5]
[1 6]]
"""
```

### 3.3.2 원핫 표현으로 변환

<img src="assets/fig 3-18.png">

맥락과 타깃의 형상이 원핫 표현으로 바뀌면서 `말뭉치의 단어 수`만큼 추가된다.

```python
def convert_one_hot(corpus, vocab_size):
    """
    원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    """
    # 말뭉치 수
    N = corpus.shape[0]

    if corpus.ndim == 1:
        # 0으로 채운 2차원 행렬 생성
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        # 0으로 채운 3차원 행렬 생성
        # N : 말뭉치 수 , C : 2 * 윈도우 사이즈 , vocab_sizxe : 어휘 수
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)

        # 맥락의 id 인덱스를 1로 변환
        # 예) [0, 2 ] => [[1 0 0 0 0 ...] ,[0 0 1 0 0 ...]]
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

one_hot = convert_one_hot(contexts, len(word_to_id))
print(one_hot)
"""
[[[1 0 0 0 0 0 0]
  [0 0 1 0 0 0 0]]

 [[0 1 0 0 0 0 0]
  [0 0 0 1 0 0 0]]

 [[0 0 1 0 0 0 0]
  [0 0 0 0 1 0 0]]

 [[0 0 0 1 0 0 0]
  [0 1 0 0 0 0 0]]

 [[0 0 0 0 1 0 0]
  [0 0 0 0 0 1 0]]

 [[0 1 0 0 0 0 0]
  [0 0 0 0 0 0 1]]]
"""
```

## 3.4 CBOW 모델 구현

<img src="assets/fig 3-19.png">

> CBOW 구현

```python
import sys

sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        # 어휘 수, 은닉층의 뉴런 수 
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        # 계층 생성
        # 입력 1 합성곱 계층
        self.in_layer0 = MatMul(W_in)
        # 입력 2 합성곱 계층
        self.in_layer1 = MatMul(W_in)
        # 출력 합성곱 계층
        self.out_layer = MatMul(W_out)
        # 손실함수 계층
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    # 순전파
    def forward(self, contexts, target):
        # 원핫인코딩된 데이터 입력
        # h0 : 첫번째 입력
        # h1 : 두번째 입력
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        # 평균 계산
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    # 역전파
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
```

### 3.4.1 학습 코드 구현

```python
import sys

sys.path.append("..")  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# 말뭉치
text = "You say goodbye and I say hello."
# 단어 ID 목록 생성
corpus, word_to_id, id_to_word = preprocess(text)

# 단어 수
vocab_size = len(word_to_id)
# 말뭉치에서 맥락과 타깃 만들기
contexts, target = create_contexts_target(corpus, window_size)
# 타깃 원핫 인코딩
target = convert_one_hot(target, vocab_size)
# 맥락 원핫 인코딩
contexts = convert_one_hot(contexts, vocab_size)

# 모델 생성
model = SimpleCBOW(vocab_size, hidden_size)
# 최적화 방법 Adam
optimizer = Adam()
# train 인스턴스 생성
trainer = Trainer(model, optimizer)

# 학습
trainer.fit(contexts, target, max_epoch, batch_size)
# 학습 결과 그래프 그리기
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
```

## 3.5 word2vec 보충

### 3.5.1 CBOW 모델과 확률

<img src="assets/fig 3-22.png">

위 그림처럼 `W_(t-1)`과 `W_(t+1)` 이 맥락일 때  `W_t`가 티깃일 확률의 수식

> P(A|B) : 사후확률. 즉 B라는 정보가 주어졌을 때 A가 일어날 확률

<img src="assets/e 3-1.png">

위 식은 `W_(t-1)과 W_(t+1)이 발생한 후 W_t가 발생할 확률`이다. 즉, CBOW는 위 식을 모델링하고 있는 것이다.

위 식을 이용하면 CBOW 모델의 손실 함수도 간결하게 표현할 수 있다.

교차 엔트로피 오차를 적용하면 아래의 식을 유도할 수 있다.

엔트로피 식에서 `y_k`는 `k번째에 해당하는 사건이 일어날 확률`을 말한다. 그리고, `t_k`는 정답 레이블이며 원핫 벡터로 표현된다.

CBOW에서 `W_t`는 `t_k`가 해당된다.

> 교차 엔트로피 오차 공식

<img src="assets/e 1-7.png">

<img src="assets/e 3-2.png">

> 위 식을 말뭉치 전체로 확장한 식

<img src="assets/e 3-3.png">

### 3.5.2 skip-gram 모델

word2vec는 2개의 모델을 제안하고 있는데 CBOW모델과 skip-gram 모델이다.

<img src="assets/fig 3-23.png">

skip-gram 모델은 `타깃`으로 `맥락`을 추측한다.

 <img src="assets/fig 3-24.png">

그림처럼 skip-gram 모델의 입력층은 1개이고, 출력층은 맥락의 수만큼 존재한다.

> skip-gram 모델 확률 표기, CBOW와 반대이다

<img src="assets/e 3-4.png">

위 식을 맥락의 단어들 사이에 관련성이 없다고 가정하고 아래처럼 분해한다.

<img src="assets/e 3-5.png">

다시 위 식을 교차 엔트로피 오차를 적용하면 skip-gram 모델의 손실 함수를 유도할 수 있다.

<img src="assets/e 3-6.png">

위 식을 말뭉치 전체로 확장하면

<img src="assets/e 3-7.png">

skip-gram 모델의 식과 CBOW 모델의 식을 비교하면 

💡 `skip-gram 모델`은 맥락의 수만큼 추측하기 때문에 손실 함수는 `각 맥락에서 구한 손실의 총합`이어야 한다. 

반면, `CBOW 모델`은 `타깃 하나의 손실`만 구한다.

두 가지 모두 장단점이 있다.

skip-gram 모델은 정확도가 더 좋지만 CBOW 모델은 학습 속도가 더 빠르다.



### 3.5.3 통계 기반 vs. 추론 기반

- 통계 기반 기법은 1회 학습, 추론 기반 기법은 여러번 학습
- 말뭉치에 새 단어를 추가할 때 통계 기반 기법은 처음부터 다시 학습, 추론 기반 기법은 매개변수만 다시 학습.
- 통계 기반 기법에서는 단어의 유사성이 인코딩되고, word2vec에서는 유사성은 물론, 복잡한 단어 사이의 패턴도 파악되어 인코딩 된다.

추론 기반 기법과 통계 기반 기법을 융합한 `GloVe`기법도 있다.

`Glove`는 말뭉치 전체의 통계정보를 손실 함수에 도입해 미니배치 학습을 한다.