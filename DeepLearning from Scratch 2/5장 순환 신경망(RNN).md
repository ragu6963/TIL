# 5장 순환 신경망(RNN) [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

지금까지 본 신경망은 `피드포워드(feed forward)`라는 유형의 신경망이다.

`피드포워드`란 흐름이 단뱡항인 신경망이다. 

✅`시계열 데이터`를 잘 다루지 못한다는 단점이 있다.

그래서 `순환 신경망(RNN)`이 등장했다.

## 5.1 확률과 언어 모델

### 5.1.1 word2vec을 확률 관점에서 바라보다

지금까지 해온 `CBOW`는 좌우 윈도우를 맥락으로 고려했다.

맥락을 왼쪽 윈도우만 한정해보자.

<img src="assets/5장 순환 신경망(RNN)/fig 5-2.png">

위 그림에서 CBOW 출력 확률은 아래 식이 된다.

<img src="assets/5장 순환 신경망(RNN)/e 5-3.png">

CBOW 모델의 학습으로  위 식의 손실 함수를 최소화 하는 가중치를 찾는다. 그러면 CBOW 모델은 맥락으로부터 타깃을 더 정확하게 추측할 것이다.

CBOW 모델의 본래 목적은 `맥락으로부터 타깃을 정확하게 추측`하는 것이고, 학습을 진행하면 부가적으로 단어의 의미가 인코딩된 `단어의 분산 표현`을 얻을 수 있다.

그렇다면 CBOW의 본래 목적인 `맥락으로부터 타깃을 정확하게 추측`하는 것은 어디에 사용될 수 있을까?



### 5.1.2 언어모델

`언어모델`은 단어 나열에 확률을 부여한다.

✅특정한 단어의 시퀀스에 대해 그 시퀀스가 일어날 가능성이 어느 정도인지 평가한다.

예를들면, "you say goodbye"는 높은 확률을 출력하고, "you say good die"는 낮은 확률을 출력하는 것이 일종의 언어 모델이다.

이 언어모델은 기계 번역과 음성 인식에 응용된다. 

예를들면, 음성 인식 시스템은 사람의 음성으로부터 몇 개의 문장을 후보로 생성할 것이다. 그 다음 언어 모델을 사용하여 후보 문장의 `문장으로써 자연스러운지`를 기준으로 순서를 매길 수 있다.

> 언어 모델의 수식화

$w_1,...,w_m$이라는 m개 단어로 된 문장이 있다.

이때 단어가 $w_1,...,w_m$ 이라는 순서로 출현할 확률을 $P(w_1,...w_m)$으로 나타낸다.

이 확률은 여러 사건이 동시에 일어날 확률이므로 `동시 확률`이라고 한다.

동시 확률 P는 아래와 같이 분해할 수 있다.

<img src="assets/5장 순환 신경망(RNN)/e 5-4.png">

위 식은 `곱셈정리`로 유도할 수 있다.

> 곱셈정리 : A와 B가 모두 일어날 확률은 B가 일어날 확률과 B가 일어난 후 A가 일어날 확률을 곱한 값과 같다.

<img src="assets/5장 순환 신경망(RNN)/e 5-5.png">

> 곱셈정리로 m개 단어의 동시 확률 P를 사후 확률로 나타낸 식

<img src="assets/5장 순환 신경망(RNN)/e 5-6.png">

<img src="assets/5장 순환 신경망(RNN)/e 5-7.png">

위 식처럼 단어 시퀀스를 하나씩 줄여가면서 사후 확률로 분해하고, 반복하면 처음의 식을 얻을 수 있다.

식에서 보이는 것처럼 목적으로 하는 

동시 확률 $P(w_1,...,w_m)$은 사후 확률의 총곱인 $πP(w_t|w_1,...,w_t-_1)$ 로 표현할 수 있다.

여기서 사후 확률은 타깃 단어보다 왼쪽에 있는 모든 단어를 맥락으로 했을 때의 확률이다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-3.png">

✅ 우리의 목표는 $P(w_t|w_1,...,w_t-_1)$ 이라는 확률을 얻는 것이다. 그리고, 확률을 계산할 수 있다면 언어 모델의 동시확률 P를 구할 수 있다.



### 5.1.3 CBOW 모델을 언어 모델로?

CBOW 를 언어모델에 적용한다면 아래 수식으로 표현할 수 있다.

<img src="assets/5장 순환 신경망(RNN)/e 5-8.png">

위 식에서는 윈도우사이즈를 2로 했다. 윈도우사이즈는 임의로 정할수 있지만 특정 길이로 고정되게 된다.

예를들면 윈도우사이즈가 10일때, 10보다 먼 거리에 있는 단어의 정보는 무시된다. 그런데 이것이 문제가 될 떄가 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-4.png">

위 그림에서 ?에 들어갈 단어를 구하기 위해서는 왼쪽으로 18번째에 있는 "Tom"을 기억해야 한다. 

하지만 윈도우사이즈가 10이라면 이 문제를 해결할 수 없다.

✅ 그렇다고 윈도우사이즈를 무작적 키우면 맥락 안의 단어 순서가 무시된다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-5.png">

CBOW 는 왼쪽 그림처럼 단어 벡터들이 더해지므로 맥락의 단어 순서는 무시된다. 예를들어 (you, say)와 (say, you)는 똑같은 맥락이 된다.

단어 순서를 고려하기 위해서는 오른쪽 그림처럼 맥락의 단어 벡터를 은닉층에서 `연결`하는 방식을 생각할 수 있다. 하지만 연결하는 방식을 사용하면 맥락의 크기에 비례해 가중치도 늘어나게 된다.

이 문제를 해결하기 위해 등장한것이 RNN이다.

## 5.2 RNN이란

`RNN`은 직역하면 `순환하는 신경망`이다

### 5.2.1 순환하는 신경망

✅ RNN의 특징은 `순환하는 경로`가 있고, 순환 경로를 따라 데이터가 끊임없이 순환한다는 것이다. 

그로인해 `과거의 정보를 기억`하는 동시에 최신 데이터로 갱신될 수 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-6.png">

$x_t$에서 t는 시각을 뜻한다. 즉, 시계열 데이터($x_0,x_1,...,x_t,...$)가 입력된다는 것이다.

그리고 입력에 대응하여 ($h_0,h_1,...,h_t,...$)가 출력된다.

### 5.2.2 순환 구조 펼치기

위 순환 구조를 펼치면 아래 그림처럼 나타낼 수 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-8.png">

위 그림의 모든 RNN 계층은 실제로는 `같은 계층`이다.

위 그림의 수식은 아래와 같다.

<img src="assets/5장 순환 신경망(RNN)/e 5-9.png">

✅RNN 에는  $x_t$ -> $h_t$ 를 위한 $W_x$ 와  이전 $h_t$ -> 현재 $h_t$ 를 위한 $W_h$ 인 2개의 가중치가 있고, 1개의 편향 b가 있다.

식에서는 행렬 곱을 계산하고, 그 합을 tanh 함수를 이용해 $h_t$를 구하는데 

이 $h_t$는 `현재 시간의 다음 계층`으로 향하면서도` 다음 시간의 RNN계층`으로 향한다.

### 5.2.3 BPTT

<img src="assets/5장 순환 신경망(RNN)/fig 5-10.png">

✅RNN의 오차역전파법은 시간 방향으로 펼친 신경망의 오차역전파라는 뜻으로 `BPTT(Backpropagation Through Time)`라고 한다.

BPTT의 문제점이 하나있는데 긴 시계열 데이터를 학습할 때  시간 크기가 커질수록 BPTT가 소비하는 컴퓨터 자원도 증가한다는 점이다.

또한, 시간 크기가 커질수록 역전파 시의 기울기가 불안정해진다(0이 되어 소멸할수도 있다.)는 점도 있다.



### 5.2.4 Truncated BPTT

✅시간 크기가 커질 때 신경망의 연결을 적당한 길이로 끊는 방법이 `Truncated BPTT` 이다.

단, 신경망의 연결을 끊을 때 역전파의 연결만 끊어야한다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-11.png">

위 그림은 10개 단위로 역전파의 연결을 끊었는데 각각의 블록은 독립적으로 오차역전파법이 완결된다.

---

순전파의 연결은 끊어지지 않았으므로 RNN을 학습시킬 때는 데이터를 순서대로 입력해야 한다는 점을 기억해야한다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-14.png">

위 그림처럼 순전파의 연결을 유지하면서 앞 시간의 블록부터 순서대로 학습해야한다.

### 5.2.5 Truncated BPTT의 미니배치 학습

지금까지의 설명은 미니배치가 1일 때의 이야기들이다. 

✅만약 미니배치가 여러개라면 데이터를 주는 시작 위치를 각 미니배치의 시작 위치로 옮겨줘야한다.

예를들면 길이가 1000이고 배치사이즈가 500이면 2개의 미니배치가 만들어진다.

첫 번째 미니배치는 처음부터 순서대로 데이터를 제공하고, 두 번째 미니배치는 500번째의 데이터를 시작위치를 정하고, 그 위치부터 순서대로 데이터를 제공한다.(즉, 시작 위치를 500만큼 옮겨준다.)

<img src='assets/5장 순환 신경망(RNN)/fig 5-15.png'>

✅ 지금까지 설명한것 처럼 Truncated BPTT는 `데이터를 순서대로 제공하기` `미니배치별로 데이터의 시작 위치 옮기기`를 주의해야한다.

## 5.3 RNN 구현

> 펼친 RNN 계층

<img src="assets/5장 순환 신경망(RNN)/fig 5-16.png">

> 펼친 RNN 계층을 하나의 계층으로 간주한 계층
>
> (x~0~,x~1~, ... , x~T-1~)을 묶은 xs 를 입력하면 (h~0~,h~1~,...,h~T-1~)을 묶은 hs를 출력한다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-17.png">

`RNN 계층`이 한 단계의 작업을 수행한다면 `Time RNN`계층은 T개 단계의 작업을 한꺼번에 처리한다.

### 5.3.1 RNN 계층 구현

> RNN의 순전파와 행렬 형상

<img src="assets/5장 순환 신경망(RNN)/e 5-9.png">

<img src="assets/5장 순환 신경망(RNN)/fig 5-18.png">

> RNN 순전파와 역전파 계산 그래프

<img src="assets/5장 순환 신경망(RNN)/fig 5-20.png">

> RNN의 초기화 순전파 역전파

```python
class RNN:
    def __init__(self, Wx, Wh, b):
        # 가중치와 편향
        self.params = [Wx, Wh, b]
        # 기울기
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # 중간 데이터
        self.cache = None

        
    def forward(self, x, h_prev):
        # 가중치와 편향
        Wx, Wh, b = self.params
        # h_t 계산(위 식의 구현)
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    
    def backward(self, dh_next):
        # 가중치와 편향
        Wx, Wh, b = self.params
        # 저장 값
        x, h_prev, h_next = self.cache
		
        # tanh 역전파
        dt = dh_next * (1 - h_next ** 2)
        
        # 편향 역전파
        db = np.sum(dt, axis=0)
        
        # W_h 역전파
        dWh = np.dot(h_prev.T, dt)
        
        # h_prev 역전파
        dh_prev = np.dot(dt, Wh.T)
        
        # W_x 역전파
        dWx = np.dot(x.T, dt)
        
        # x 역전파
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev
```

### 5.3.2 Time RNN 계층 구현

`Time RNN`은 `RNN`을 `T`개를 연결한 신경망이다.

> RNN 계층의 은닉 상태 h를 인스턴스 변수로 유지한다고 하는데 은닉상태가 어떤 의미인지 정확하게 모르겠다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-22.png">

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/time_layers.py

class TimeRNN:
    # stateful = True, 은닉상태 유지 -> 순전파를 끊지 않고 전파
    # stateful = False, 은닉상태 해제 -> 은닉상태를 영행렬로 초기화
    def __init__(self, Wx, Wh, b, stateful=False):
        # 가중치 편향
        self.params = [Wx, Wh, b]
        # 기울기
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # 게층
        self.layers = None
		
        # 은닉상태 순전파 h, 은닉상태 역전파 h 
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        # 가중치 편향
        Wx, Wh, b = self.params
        # 하류에서 올라온 xs 형상
        # 미니배치 크기, 시계열 분량 T, 입력 벡터의 차원수
        N, T, D = xs.shape
        # 가중치 형상
        D, H = Wx.shape 
			
        self.layers = []
        # 출력값을 담을 그릇
        hs = np.empty((N, T, H), dtype='f')
		
        # h 초기값 = None, 영행렬로 초기화 
        # stateful = False, 은닉상태 해제 -> 은닉상태를 영행렬로 초기화
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
		
        # T회분의 RNN 계층 생성
        for t in range(T):
            # RNN 계층 인스턴스 생성(인자 : 가중치와 편향)
            layer = RNN(*self.params)
            
            # RNN계층 출력 계산
			# stateful 값에 따라 다음 T회분의 h 계산시에 마지막 계산의 h 가 사용될 수도 있다.
            # stateful = False, 영행렬로 초기화
            # stateful = True, 다음 순전파 계산 때 마지막 계산 h 사용
            self.h = layer.forward(xs[:, t, :], self.h)
            # t 시각의 출력 h(은닉 상태) 저장
            hs[:, t, :] = self.h
            # 역전파를 위해 계층 저장
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        # 상류에서 내려오는 기울기 dhs
        N, T, H = dhs.shape
		# 가중치 형상
        D, H = Wx.shape
		
        # 하류로 내려보내는 기울기 dxs
        dxs = np.empty((N, T, D), dtype='f')
        # 이전 시각의 은닉상태 기울기 dh
        dh = 0
        # 기울기
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 역전파 수행
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            
            # t 시각의 기울기 dx 저장
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                # 기울기 저장
                grads[i] += grad

        for i, grad in enumerate(grads):
            # 기울기 멤버변수에 저장
            self.grads[i][...] = grad
            
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None
```

## 5.4 시계열 데이터 처리 계층 구현

✅이번장의 목표는 RNN을 사용하여 `언어 모델`을 구현하는 것이다

`시계열 데이터`를 처리하는 계층을 몇 개 더 만들어보자.

> RNN을 사용한 언어 모델은 RNN Langauge Model(RNNLM)

### 5.4.1 RNNLM의 전체 그림

<img src="assets/5장 순환 신경망(RNN)/fig 5-25.png">

- 첫 번째 층`Embedding` : 단어 ID를 단어의 분산 표현으로 변환
- 두 번째 층`RNN` : 은닉 상태를 다음 층으로 출력 및 오른쪽 RNN으로 출력
- 세 번째 층`Affine` : 은닉 상태를 `Softmax`계층으로 전달
- 네 번째 층`Softmax` :  점수를 확률로 변환

> "You say goodbye and I say hello."를 사용한 예

<img src="assets/5장 순환 신경망(RNN)/fig 5-26.png">