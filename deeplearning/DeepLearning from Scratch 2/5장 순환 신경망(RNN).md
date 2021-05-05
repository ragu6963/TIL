# 5장 순환 신경망(RNN) [github](https://github.com/WegraLee/deep-learning-from-scratch-2)

지금까지 본 신경망은 `피드포워드(feed forward)`라는 흐름이 단뱡항인 신경망이다.

✅`시계열 데이터`를 잘 다루지 못한다는 단점이 있다. 그래서 `순환 신경망(RNN)`이 등장했다.

## 5.1 확률과 언어 모델

### 5.1.1 word2vec을 확률 관점에서 바라보다

지금까지 해온 `CBOW`는 좌우 윈도우를 맥락으로 고려했다.

맥락을 `왼쪽` 윈도우만 한정해보자.

<img src="assets/5장 순환 신경망(RNN)/fig 5-2.png">

위 그림에서 CBOW 출력 확률은 아래 식이 된다.

<img src="assets/5장 순환 신경망(RNN)/e 5-3.png">

✅ CBOW 모델의 학습으로  위 식의 손실 함수를 최소화하는 가중치를 찾는다. 그러면 CBOW 모델은 맥락으로부터 타깃을 더 정확하게 추측할 것이다.

CBOW 모델의 본래 목적은 `맥락으로부터 타깃을 정확하게 추측`하는 것이고, 

✅ 학습을 진행하면서 부가적으로 단어의 의미가 인코딩된 `단어의 분산 표현`을 얻을 수 있다.

그렇다면 CBOW의 본래 목적인 `맥락으로부터 타깃을 정확하게 추측`하는 것은 어디에 사용될 수 있을까?



### 5.1.2 언어모델

`언어모델`은 단어 나열에 확률을 부여한다.

✅특정한 단어의 시퀀스에 대해 그 시퀀스가 일어날 가능성이 어느 정도인지 평가한다.

예를들면, "you say goodbye"는 높은 확률을 출력하고, "you say good die"는 낮은 확률을 출력하는 것이 일종의 언어 모델이다.

이 언어모델은 기계 번역과 음성 인식에 응용된다. 

예를들면, 음성 인식 시스템은 사람의 음성으로부터 몇 개의 문장을 후보로 생성할 것이다. 그
 다음 언어 모델을 사용하여 후보 문장의 `문장으로써 자연스러운지`를 기준으로 순서를 매길 수 있다.

> 언어 모델의 수식화

$w_1,...,w_m$이라는 m개 단어로 된 문장이 있다.

이때 단어가 $w_1,...,w_m$ 이라는 순서로 출현할 확률을 $P(w_1,...w_m)$으로 나타낸다.

이 확률은 여러 사건이 동시에 일어날 확률이므로 `동시 확률`이라고 한다.

동시 확률 `P`는 사후 확률을 사용하여 아래와 같이 분해할 수 있다.

<img src="assets/5장 순환 신경망(RNN)/e 5-4.png">

위 식은 결과는 `곱셈정리`로부터 유도할 수 있다.

> 곱셈정리 : A와 B가 모두 일어날 확률은 B가 일어날 확률과 B가 일어난 후 A가 일어날 확률을 곱한 값과 같다.

<img src="assets/5장 순환 신경망(RNN)/e 5-5.png">

> 곱셈정리로 m개 단어의 동시 확률 P를 사후 확률로 나타낸 식

<img src="assets/5장 순환 신경망(RNN)/e 5-6.png">

<img src="assets/5장 순환 신경망(RNN)/e 5-7.png">

위 식처럼 단어 시퀀스를 `하나씩 줄여가면서` 사후 확률로 분해하고, 반복하면 처음의 식을 얻을 수 있다.

식에서 보이는 것처럼 목적으로 하는 

동시 확률 $P(w_1,...,w_m)$은 사후 확률의 총 곱인 πP(w~t~|w~1~,...,w~t-1~) 로 표현할 수 있다.

여기서 사후 확률은 타깃 단어보다 왼쪽에 있는 모든 단어를 맥락으로 했을 때의 확률이다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-3.png">

✅  P(w~t~|w~1~,...,w~t-1~) 이라는 확률을 얻는것이 목표이고, 그 확률을 계산할 수 있다면 언어 모델의 동시확률 `P`를 구할 수 있다.

### 5.1.3 CBOW 모델을 언어 모델로?

CBOW 를 언어모델에 적용한다면 아래 수식으로 표현할 수 있다.

<img src="assets/5장 순환 신경망(RNN)/e 5-8.png">

위 식에서는 윈도우사이즈를 2로 했다. 윈도우사이즈는 임의로 정할수 있지만 특정 길이로 고정되게 된다.

✅ 예를들면 윈도우사이즈가 10일때, 10보다 먼 거리에 있는 단어의 정보는 무시된다. 그런데 이것이 문제가 될 떄가 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-4.png">

위 그림에서 ?에 들어갈 단어를 구하기 위해서는 왼쪽으로 18번째에 있는 "Tom"을 기억해야 한다. 

하지만 윈도우사이즈가 10이라면 이 문제를 해결할 수 없다.

✅ 그렇다고 윈도우사이즈를 무작정 키우면 맥락 안의 단어 순서가 무시된다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-5.png">

CBOW 는 왼쪽 그림처럼 단어 벡터들이 더해지므로 맥락의 단어 순서는 무시된다. 
예를들어 (you, say)와 (say, you)는 똑같은 맥락이 된다.

단어 순서를 고려하기 위해서는 오른쪽 그림처럼 맥락의 단어 벡터를 은닉층에서 `연결`하는 방식을 생각할 수 있다. 
하지만 연결하는 방식을 사용하면 맥락의 크기에 비례해 가중치도 늘어나게 된다.

이 문제를 해결하기 위해 등장한것이 `RNN`이다.

## 5.2 RNN이란

`RNN`은 직역하면 `순환하는 신경망`이다.

### 5.2.1 순환하는 신경망

✅ RNN의 특징은 `순환하는 경로`가 있고, 순환 경로를 따라 데이터가 끊임없이 순환한다는 것이다. 

그로인해 `과거의 정보를 기억`하는 동시에 최신 데이터로 갱신될 수 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-6.png">

$x_t$에서 t는 시각을 뜻한다. 즉, 시계열 데이터($x_0,x_1,...,x_t,...$)가 입력된다는 것이다.

그리고 입력에 대응하여 ($h_0,h_1,...,h_t,...$)가 출력된다.

### 5.2.2 순환 구조 펼치기

위 순환 구조를 펼치면 아래 그림처럼 나타낼 수 있다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-8.png">

위 그림의 모든 RNN 계층은 실제로는 `같은 계층`이다. 위 그림의 수식은 아래와 같다.

<img src="assets/5장 순환 신경망(RNN)/e 5-9.png"  >

✅RNN 에는  $x_t$ -> $h_t$ 를 위한 $W_x$ 와  이전 $h_t$ -> 현재 $h_t$ 를 위한 $W_h$ 인 2개의 가중치가 있고, 1개의 편향 b가 있다.

식에서는 행렬 곱을 계산하고, 그 합을 tanh 함수를 이용해 $h_t$를 구하는데 

이 $h_t$는 `현재 시간의 다음 계층`으로 향하면서도` 다음 시간의 RNN계층`으로 향한다.

> 여기서 h는 상태를 기억하는 값이다. 그래서 많은 문헌에서 h_t 를 `은닉 상태` 혹은 `은닉 상태 벡터`라고 표현한다.

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

예를들면 길이가 1000일때, 2개의 미니배치를 만들려면 어떻게 해야할까?

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
>
> 미니배치 크기 N, 입력의 차원수 D, 은닉 상태 벡터의 차원 수 H

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

은닉 상태`h`를 인스턴스 변수로 유지하고, 다음 시간`블록`에 사용한다.

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
        
        # 역으로 계층 수행
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
	
    # 은닉 상태를 설정하는 메서드
    def set_state(self, h):
        self.h = h

    # 은닉 상태를 초기화하는 메서드
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

첫 번째 단어인 `you` 를 입력했을 때, `say`가 가장 높은 점수를 받았다.

두 번째 단어인 `say`를 입력했을 때에는 `goodbye`와 `hello`가 높은 점수를 받았다.

여기서 중요한점은 앞 계층의 `you say`를 기억한 상태라는 것이다.

즉, 모델은 `you say goodbye` 와 `you say hello` 두 개를 높은 점수를 줬다는 뜻이다.

✅ 이처럼 `RNNLM`은 입력된 단어를 기억하고, 그것을 바탕으로 다음에 출현할 단어를 예측한다.

이러한 일을 가능하게 해주는 것이 `RNN` 계층이다.  

### 5.4.2 Time 계층 구현

<img src="assets/5장 순환 신경망(RNN)/fig 5-27.png">

`Time Affine` 계층과 `Time Embedding` 계층은 특별한 점은 없다.

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/time_layers.py
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        # 입력의 형상
        N, T = xs.shape
        # 가중치의 형상
        V, D = self.W.shape
		
        # 출력 저장 변수
        out = np.empty((N, T, D), dtype='f')
        self.layers = []
		
        # T개에 대한 Embedding 출력과 계층 저장
        for t in range(T):
            layer = Embedding(self.W)
            # 출력 저장
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        # 입력 형상
        N, T, D = x.shape
        # 가중치와 편향
        W, b = self.params
		
        # 입력 펼치기(?)
        rx = x.reshape(N*T, -1)
        # 출력 계산 xw + b
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx
```

다만, `Time Softmax`계층을 구현할 때에는 `T`개의 출력에 대한 평균 최종손실로 사용한다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-29.png">

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/time_layers.py
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        # 입력 형상
        N, T, V = xs.shape
		
        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)
		
        # 무시할 index?
        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        # 예측 점수
        xs = xs.reshape(N * T, V)
        
        # 정답 레이블
        ts = ts.reshape(N * T)
        
        mask = mask.reshape(N * T)
		
        # 점수 -> 확률
        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        
        # 손실함수 평균
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx
```

## 5.5 RNNLM 학습과 평가

### 5.5.1 RNNLM 구현

<img src="assets/5장 순환 신경망(RNN)/fig 5-30.png">

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch05/simple_rnnlm.py

import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        # 임베디드계층 가중치
        embed_W = (rn(V, D) / 100).astype('f')
        # rnn 가중치(Xavier 초깃값)와 편향
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        # affine 가중치(Xavier 초깃값)와 편향
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            # stateful=True : 이전 시각 은닉 상태 계승
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        # rnn 계층
        self.rnn_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        # Embedding -> RNN -> Affine 순서대로 호출
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        # Affine -> RNN -> Embedding 순서대로 호출
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    # 신경망의 상태 초기화
    def reset_state(self):
        self.rnn_layer.reset_state()
```

### 5.5.2 언어 모델의 평가

✅언어 모델의 예측 성능을 평가하는 척도로 `퍼플렉서티(perplexity),혼란도`를 자주이용한다.

`퍼플렉서티`를 간다히 말하면 `확률의 역수`이다.

아래 그림으로 예를들면,

모델 1은  `you`를 입력했을 때 정답 레이블인 `say`의 확률이 0.8이 나왔다. 

이 때의 퍼플렉서티는 1/0.8 = 1.25 이다.

모델 2는 0.2의 확률이 나왔고, 퍼플렉서티는 5이다.

✅ 즉, 퍼플렉서티는 작을 수로 좋다는 것이다. 그리고, 이 값은 다음에 나올 수 있는 후보 단어 수`분기 수`를 뜻한다. 즉, 1.25는 후보 단어수가 약 1개라는 것이고, 5는 약 5개나 있다는 의미이다.

<img src="assets/5장 순환 신경망(RNN)/fig 5-32.png">

> 입력이 여러 개일 때의 공식
>
> L : 신경망의 손실, N :  데이터 총 개수, t~n~  : 정답레이블, t~nk~ : n개째 데이터의 k번째 값, y~k~ : 확률분포

<img src="assets/5장 순환 신경망(RNN)/e 5-12.png">

<img src="assets/5장 순환 신경망(RNN)/e 5-13.png">

### 5.5.3 RNNLM의 학습코드

```python
# https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch05/train_custom_loop.py

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# 하이퍼파라미터 설정
batch_size = 10
wordvec_size = 100
hidden_size = 100 # RNN의 은닉 상태 벡터의 원소 수
time_size = 5     # Truncated BPTT가 한 번에 펼치는 시간 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기(전체 중 1000개만)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 입력
ts = corpus[1:]   # 출력(정답 레이블)
data_size = len(xs)
print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))

# 학습 시 사용하는 변수
# 한 뭉치를 몇번으로 나눠서 학습할지를 정하는 변수?
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
# 최적화 방법
optimizer = SGD(lr)

# 미니배치의 각 샘플의 읽기 시작 위치를 계산
"""
EXAM
corpus_size = 1000, batch_size = 2
jump = 499 -> 1번째와 500번째가 시작위치
i = [0,1] i*jump = [0, 499]
"""
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 미니배치 취득
        # 입력과 정답 레이블
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 기울기를 구하여 매개변수 갱신
        # 순전파 수행
        loss = model.forward(batch_x, batch_t)
        # 역전파 수행
        model.backward()
        # 가중치 업데이트
        optimizer.update(model.params, model.grads)
        # 손실 총 합
        total_loss += loss
        # 손실 개수
        loss_count += 1

    # 에폭마다 퍼플렉서티 평가
    ppl = np.exp(total_loss / loss_count)
    print('| 에폭 %d | 퍼플렉서티 %.2f'
          % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 그래프 그리기
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
```

