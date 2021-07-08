### useState
> useState 는 Hook 중에 하나이고, useState는 state를 함수 컴포넌트 안에서 사용할 수 있게 해준다. 출처 : [공식문서](https://ko.reactjs.org/docs/hooks-state.html)

클래스형 컴포넌트에서만 사용할 수 있었던 state 를 함수형 컴포넌트에서도 사용할 수 있게 해준다.

### 카운터 만들기
#### 기본 코드
```jsx
// App.js
import React from 'react';
import Counter from './Counter';

function App() {
  return (
    <Counter />
  );
}

export default App;
```

```jsx
// Counter.js
import React from 'react';

const Counter = () => {
  return (
    <div>
      <h1>0</h1>
      <button>+1</button>
      <button>-1</button>
    </div>
  );
}

export default Counter;
```

#### 이벤트 생성
증가ㆍ감소 함수를 만들고, 각 버튼의 클릭 이벤트에 호출되도록 설정하자. 
```jsx
// Counter.js
const Counter = () => {
  const onPlus = () => {
  }
  const onMinus = () => {
  }
  return (
    <div>
      <h1>0</h1>
      <button onClick={onPlus}>+1</button>
      <button onClick={onMinus}>-1</button>
    </div>
  );
}
```

#### useState 사용해서 동적값(state) 설정
1. `useState` 함수를 사용하기 위해서 모듈을 추가한다.
```jsx
// Counter.js
import React,{useState} from 'react';
```

2. `useState` 함수를 호출해서 `state` 와 `Setter` 를 만든다.
```jsx
const Counter = () => {
  // [state, Setter] = useState(기본값)
  const [number, setNumber] = useState(0);
  
  return (
    // ...
  );
}
```

3. 증가ㆍ감소 함수에 Setter 를 추가한다. 
Setter 로 전달한 인자로 state 가 갱신된다.
```jsx
const onPlus = () => {
  setNumber(number + 1)
}
const onMinus = () => {
  setNumber(number - 1)
}
```

4. state 할당
```<h1>{number}</h1>```

5. 전체 코드
```jsx
import React, { useState } from 'react';

const Counter = () => {
  
  const [number, setNumber] = useState(0);
  
  const onPlus = () => {
    setNumber(number + 1)
  }
  
  const onMinus = () => {
    setNumber(number - 1)
  }
  
  return (
    <div>
      <h1>{number}</h1>
      <button onClick={onPlus}>+1</button>
      <button onClick={onMinus}>-1</button>
    </div>
  );
}

export default Counter;
```

### 카운터 증가 감소 값 정하기
1씩 증가하고 감소하는 현재 코드를 수정하여 증감하는 값을 정할 수 있게 해보자.

#### 새로운 useState
증감할 값을 관리할 새로운 state
```jsx
const [value, setValue] = useState(1);
```

#### 증감값을 입력할 input 태그 
input 태그의 상태를 관리한다.
```jsx
<input type="number" onChange={onChange} value={value}/>
```

#### onChange 이벤트에 반응할 함수
`e.target`은 이벤트가 발생한 태그를 의미하고, 변화한 값으로 상태를 갱신한다는 의미이다.
```jsx
const onChange = (e) => {
  setValue(e.target.value)
}
```

<img src="assets/(6) useState/image.png">

#### 전체코드

```jsx
import React, { useState } from 'react';

const Counter = () => {
  const [number, setNumber] = useState(0);
  const [value, setValue] = useState(1);


  const onPlus = () => {
    setNumber(number + Number(value))
  }

  const onMinus = () => {
    setNumber(number - Number(value))
  }

  const onChange = (e) => {
    setValue(e.target.value)
  }

  return (
    <div>
      <h1>{number}</h1>
      <input type="number" onChange={onChange} value={value} />
      <button onClick={onPlus}>+</button>
      <button onClick={onMinus}>-</button>
    </div>
  );
}

export default Counter;
```
