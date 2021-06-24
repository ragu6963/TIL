# 7장 RNN을 사용한 문장 생성 [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

## 7.1 언어 모델을 사용한 문장 생성

### 7.1.1 RNN을 사용한 문장 생서의 순서

> "you say goodbye and I say hello"를 학습한 언어 모델에서 "I" 를 입력했을 때 확률분포

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-2.png">

✅언어 모델이 출력하는 단어의 확률분포에서 단어를 생성하는 방법은 두가지가 있다.

1. 확률이 가장 높은 단어를  선택하는 방법 `결정적 방법`, 선택되는 단어가 고정
2. 각 후보 단어의 확률에 맞게 선택하는 방법 `확률적 방법`, 선택되는 단어가 매번 다를 수 있다.

### 7.1.2 문장 생성 구현

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch07/rnnlm_gen.py
# 문장 생성 클래스
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm 


# Rnnlm 클래스 상속
class RnnlmGen(Rnnlm):
    # Rnnlm 클래스에서 문장 생성 메서드 생성
    # 입력 단어 ID, 샘플링 하지 않을 단어 ID 목록, 샘플링 하는 단어의 수 
    def generate(self, start_id, skip_ids=None, sample_size=100):
        # 생성 문장 단어 ID
        word_ids = [start_id]
		
        # 입력 x
        x = start_id
        while len(word_ids) < sample_size:
			# 입력 ID 행렬화 (1,1)
            x = np.array(x).reshape(1, 1)
            # 예측 수행
            score = self.predict(x)
            # 확률화
            p = softmax(score.flatten())
			
            # 단어 샘플링
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                # 입력 교체
                x = sampled
                # 결과 단어 ID 추가
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)
```

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch07/generate_text.py
# 문장 생성 코드
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

# 전처리
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

# 모델 인스턴스 생성
model = RnnlmGen()
# 가중치 불러오기
model.load_params('../ch06/Rnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

✅코드 실행 결과 어느정도 올바른 문장도 나오지만 부자연스러운 문장도 많이 발견된다

### 7.1.3 더 좋은 문장으로

6장에서 단순한 RNNLM을 개선한 RNNLM으로 더 좋은 문장을 생성할 수 있다.

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch07/rnnlm_gen.py
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax 
from ch06.better_rnnlm import BetterRnnlm 


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)
```

## 7.2 seq2seq

✅**seq2seq**는 2개의 RNN모델을 사용해서 시계열 데이터를 다른 시계열 데이터로 변환하는 방법이다.

### 7.2.1 seq2seq2의 원리

seq2seq모델을 **Encoder-Decoder**모델 이라고도 한다.

- Encoder는 입력 데이터를 인코딩(부호화)
- Decoder는 인코딩된 데이터를 디코딩(복호화)

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-5.png'>

위 그림처럼 Encoder는 "나는 고양이로소이다"를 인코딩하고, 그 정보를 Decoder에 전달한다.

Decoder는 인코딩된 문장을 디코딩해서 문장을 생성한다.

---

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-6.png'>

✅ Encoder는 시계열 데이터를 **h(은닉상태벡터)**로 변환한다.

중요한 점은 **h**는 고정 길이 벡터라는 것이다. 즉, 임의 길이 문장을 고정 길이 벡터로 변환한다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-7.png">

---

✅ Decoder는 문장 생성 신경망과 LSTM 계층이 **h**를 입력받는다는 점만 다르고 동일한 구성을 가진다.

**은닉 상태 h**를 입력받아서 문장을 생성한다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-8.png'>

> 은닉 상태 **h**는 Encoder와 Decoder사이의 **가교** 역할을 한다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-9.png">

### 7.2.2 시계열 데이터 변환용 장난감 문제

**`장난감 문제`**란 머신러닝을 평가하고자 만든 간단한 문제이다.

예를들면, "57 + 6" 와 같은 문자열을 입력하면 "62"라는 정답을 출력하게 학습시키는 것이다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-10.png">

이 문제의 핵심은 모델이 **덧셈 규칙**을 올바르게 학습할 수 있느냐이다.

---

그리고, 지금까지 언어모델에서는 문장을 **단어** 단위로 분할해왔다면

이 문제에서는 **문자** 단위로 분할할려고 한다. 즉, "57 + 5"를 ["5","7","+","5"] 로 분할한다는 의미이다.

### 7.2.3. 가변 길이 시계열 데이터

✅ 문장을 **문자** 단위로 다룰 때 주의할 점은 문제 마다 **문자수가 다르다**는 것이다.

예를들어, "57 + 5"는 4문자이고, "628 + 521"은 7문자이다.

이러한 문제의 데이터는 **가변 길이 시계열 데이터**이고, 신경망 학습 시 미니배치 처리를 하기위해서는 추가적인 기능이 필요하다.

---

**패딩(padding)**은 가변 길이 시계열 데이터를 미니배치 처리를 하기 위한 가장 단순한 방법이다.

가장 긴 문장을 기준으로 입력과 출력공간을 만들고, 남는 공간은 **공백**으로 채운다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-11.png'>

출력 데이터의 **밑줄(_)**은 입력과 출력의 구분자이다.

---

패딩을 사용했을 때 문제는 패딩용 문자까지 seq2seq가 처리하게 된다는 것이다.

그래서 정확성이 중요하다면 패딩 전용 처리를 추가해야한다.

Decoder에 패딩 데이터가 입력되면 손실의 결과에 반영하지 않도록 한다.

Encoder에 패딩 데이터가 입력되면 LSTM 계층이 이전 시각의 입력을 그대로 출력하게 한다. 즉, LSTM 계층이 패딩이 존재하지 않았던 것처럼 인코딩할 수 있다.

## 7.3 seq2seq 구현

### 7.3.1 Encoder 클래스

✅ Encoder 클래스는 Embedding 계층과 LSTM 계층으로 구성되고, 문자열을 벡터 h로 변환한다.

그리고 **`벡터 h`**가 Decoder로 전달 된다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-13.png'>



<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-14.png">

시간 방향을 한꺼번에 처리하기 위한 Time 계층은 아래와 같다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-15.png">

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch07/seq2seq.py
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Encoder:
    # 어휘 수(문자의 종류), 문자 벡터의 차원 수, 은닉 상태 벡터의 차원 수
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
		
        # Embedding
        self.embed = TimeEmbedding(embed_W)
        # LSTM 계층, stateful을 False로 설정해 계층간의 상태를 유지하지 않는다.
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        # Embedding 계층과 LSTM 계층의 순전파
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]
	
    def backward(self, dh):
        # dh : 마지막 은닉 상태에 대한 기울기
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
```

### 7.3.2 Decoder 클래스

✅ Decoder 클래스는 LSTM 계층으로 구성되고, 상태 벡터 h를 받아 문자열을 출력한다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-16.png">

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-17.png'>

**7.1** 에서 문장을 생성할 때는 확률분포를 바탕으로 샘플링을 했다. 그래서 생성되는 문장이 매번달라졌다.

하지만 **덧셈**은 확률을 배제하고, 점수가 가장 높은 문자만 고를 것이다. 즉, **'확률적'**이 아닌 **'결정적'**으로 선택한다.

---

문자열 생성 계층에서 Affine 다음 나오는 **argmax** 계층은 **최댓값을 가진 원소의 인덱스를 선택**하는 계층이다.

즉, 점수가 가장 높은 문자ID를 선택한다.

> Decoder 문자열 생성 순서

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-18.png'>

> Decoder 문자열 학습 순서

다만, Decoder 학습 시에는 **Softmax** 계층을 사용한다. 그래서 Seq2seq 클래스에서 **Softmax**에 대한 추가처리를 해준다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-19.png">

```python
# https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch-2/master/ch07/seq2seq.py
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel 

class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
		
        # Embedding, LSTM, Affine 계층
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)
		
        # Embedding, LSTM, Affine 계층 순전파
        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
	
    # 문장 생성 시
    # Encoder에서 전달 받은 상태 벡터 h, 입력 문자 ID, 생성 문자 수
    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            # # Embedding, LSTM, Affine 계층
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
			
            # argmax 처리, 가장 높은 점수의 문자 ID를 선택한다.
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled
 
```

### 7.3.3 Seq2seq 클래스

```python
# https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch-2/master/ch07/seq2seq.py
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel 

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        # decoder 학습시에 마지막에 softmax 처리를 추가한다. 
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
```

### 7.3.4 seq2seq 평가

✅ seq2seq의 학습 흐름

1. 학습 데이터에서 미니배치를 선택하고
2. 미니배치로부터 기울기를 계산하고
3. 기울기를 사용하여 매개변수를 갱신한다.





## 7.4 seq2seq 개선

### 7.4.1 입력 데이터 반전(Reverse)

✅ 입력 데이터의 순서를 반전시키는 방법이다. 이 방법은 학습 진행이 빨리지면서 최종 정확도도 좋아진다.

이 방법이 효과가 좋은 이유는 **기울기 전파**가 원활해지기 때문이라고 생각된다. 

예를들면, "나는 고양이로소이다" -> "I am a cat"의 번역문제에서 **"나"** 에서 **"I"** 까지 가기 위해서는
"는", " 고양이", "로소", "이다" 까지 총 네 단어를 거쳐야한다. 그러므로 역전파 시 "I" 에서 "나" 까지 기울기가 전해지기 까지 거리만큼 영향을 받게 된다.

그런데 입력을 반전시키면 "나"와 "I" 가 바로 옆에 배치되고, 기울기가 바로 전해진다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-23.png'>

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-24.png">

### 7.4.2 엿보기(Peeky)

✅ **엿보기**는 Encoder에서 전달받은 벡터 **h**를 Decoder의 모든 **LSTM 계층** 과 **Affine 계층**에 전달하는 것이다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-25.png">

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-26.png">

다만, LSTM 계층과 Affine 계층에 입력되는 벡터가 2개가 되기 때문에  **concat 노드**를 이용해야 한다.

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-27.png">



```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch07/peeky_seq2seq.py
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
		
        embed_W = (rn(V, D) / 100).astype('f')
        # H와 D를 더한 형상이 만들어진다.
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        # H와 H를 더한 형상이 만들어진다.
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        # h를 시계열만큼 복제해서 hs에 저장
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # concat 수행
        # embedding 출력과 hs를 합치기
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        # lstm 출려과 hs를 합치기
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
```

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-28.png'>

## 7.5 seq2seq를 이용하는 애플리케이션

- 기계번역 : '한 언어의 문장'을 '다른 언어의 문장'으로 변환
- 자동요약 : '긴 문장'을 '짧게 요약된 문장'으로 변환
- 질의응답 : '질문'을 '응답'으로 변환
- 메일 자동 응답 : '받은 메일의 문장'을 '답변 글'로 변환

### 7.5.1 챗봇

✅ **챗봇**은 사람과 컴퓨터가 텍스트로 대화를 나누는 프로그램이다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-29.png'

### 7.5.2 알고리즘 학습

✅ 파이썬 코드와 같은 소스 코드를 처리할 수 있다.

<img src='deep_learning_2_images/fig 7-30.png'>

### 7.5.3 이미지 캡셔닝

✅ 이미지를 문장으로 변환하는 기술이다.

Encoder에서 LSTM이 아닌 **CNN** 계층을 사용한다. 이 때 Encoder의 최종 출력은 특징 맵`feature map`이다.

특징 맵은 3차원 이므로 LSTM이 처리할 수 있도록, 1차원으로 평탄화한 후 Affine 계층에서 변환해야 한다.

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-31.png'>

<img src='assets/7장 RNN을 사용한 문장 생성/fig 7-31-1618113478814.png'>