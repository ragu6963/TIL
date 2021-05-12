# Vuex

`Vuex`는  Vue의 `상태 관리` 라이브러리이다.

`상태 관리`란 여러 컴포넌트 간의 데이터 전달과 이벤트 통신을 한 곳에서 관리하는 패턴을 의미한다.

일반적인 Vue 의 통신 방식인 props, emit 을 사용하지 않고, 컴포넌트 간 데이터 통신이 가능해진다. 
이로인해 컴포넌트 간 데이터 흐름을 파악하기 쉬워진다.

### 상태 관리 패턴

상태 관리 구성요소는 크게 3가지로 나눌 수 있다.

- state : 컴포넌트 간의 공유할 `data`
- view : 데이터가 표현될 `template`
- actions : 이벤트에 반응할 `methods`

> 상태 관리는 아래와 같은 흐름으로 동작한다.

<img src="assets/6_Vuex_기초/vuex-state-one-way-data-flow.png">

### Vuex 구조

Vuex는 `state` `mutations` `action` `getters` 4가지 형태로 관리된다.

#### state

vue 컴포넌트에서의  `data` 와 동일하게 볼 수 있다. View 와 연결된 Model 이다 

#### mutations

`state` 를 변경할 수 있는 유일한 수단이다. 직접 호출 할 수 없으며 `commit()`으로 호출할 수 있다.

또한, mutations 는 `동기 작업`만 가능하다.

#### action

mutations 와 유사하나 `비동기 작업`이 가능하다.

`context` 를 첫 번째 인자로 받으며 store 의 다른 속성에 접근할 수 있다.

action 은 `dispatch` 를 통해서 호출한다.

#### getters

컴포넌트의 `computed` 와 동일하게 볼 수 있다. 즉, 특정 `state` 에 대한 연산을 하고, 결과를 반환한다. 

 