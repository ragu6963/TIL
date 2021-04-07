# 7장 RNN을 사용한 문장 생성 [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

## 7.1 언어 모델을 사용한 문장 생성

### 7.1.1 RNN을 사용한 문장 생서의 순서

> "you say goodbye and I say hello"를 학습한 언어 모델에서 "I" 를 입력했을 때 확률분호

<img src="assets/7장 RNN을 사용한 문장 생성/fig 7-2.png">

✅ 다음 단어를 생성하는 방법은 두 가지가 있다.

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

## 7.3 seq2seq 구현

### 7.3.1 Encoder 클래스

