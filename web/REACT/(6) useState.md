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

### 객체를 이용하여 여러개의 input 관리하기
일반적으로 `form` 태그 안에는 여러개의 `input` 이 필요하다.
그런데 각 input 의 상태 관리를 위해 `useState` 를 호출하고, `onChange` 함수를 만들면 코드의 가독성이 떨어질 수 밖에 없다.

다음 코드는 간단한 회원가입 컴포넌트이다.
3개의 `input` 을 위해 3개의 `useState` 와 `onChange` 를 만들었다.
```jsx
const Signup = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordconfirm, setPasswordConfirm] = useState("");

  const onChangeUsername = (e) => {
    setUsername(e.target.value)
  }

  const onChangePassword = (e) => {
    setPassword(e.target.value)
  }

  const onPasswordConfirm = (e) => {
    setPasswordConfirm(e.target.value)
  }

  return (
    <div>
      <form>
        <label htmlFor="username">아이디 : </label>
        <input id="username" value={username} onChange={onChangeUsername} />
        <br />
        <label htmlFor="password">비밀번호 : </label>
        <input id="password" value={password} onChange={onChangePassword} />
        <br />
        <label htmlFor="passwordconfirm">비밀번호 확인 : </label>
        <input id="passwordconfirm" value={passwordconfirm} onChange={onPasswordConfirm} />
      </form>
    </div>
  )
}
```

다음 코드는 state 를 객체로 생성해서 관리하는 코드이다.
하나의 `useState` 와 `onChange` 로 여러개의 `input` 관리할 수 있다.
```jsx
const Signup = () => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    passwordconfirm: '',
  });

  // 구조분해할당
  const { username, password, passwordconfirm } = credentials

  const onChange = (e) => {
    // 이벤트가 발생한 요소의 value 와 name 추출
    const { value, name } = e.target;

    // 상태 갱신
    setCredentials({
      ...credentials, // 스프레드 연산자, 기존 credentials 를 복사
      [name]: value // name 키를 가진 값을 value 로 갱신
    });
  };


  return (
    <div>
      <form>
        <label htmlFor="username">아이디 : </label>
        <input id="username" name="username" value={username} onChange={onChange} />
        <br />
        <label htmlFor="password">비밀번호 : </label>
        <input id="password" name="password" value={password} onChange={onChange} />
        <br />
        <label htmlFor="passwordconfirm">비밀번호 확인 : </label>
        <input id="passwordconfirm" name="passwordconfirm" value={passwordconfirm} onChange={onChange} />
      </form>
    </div>
  )
}
```

### immer.js 로 불변성 관리하기

react 에서 배열 혹은 객체 `state` 를 변경할 때에는 직접적으로 값을 수정하면 안된다.
그래서 위의 코드에서는 `credentials` 객체를 수정할 때 직접적인 수정이 아니라 새로운 객체를 생성해서 값을 변경했다.

```jsx
const onChange = (e) => { 
  const { value, name } = e.target;
 
  setCredentials({
    // 새로운 객체를 생성해서 상태를 변경한다.
    ...credentials,  
    [name]: value  
  });
};
```

`immer` 는 이러한 과정을 편하게 할 수 있게 도와준다.

#### immer 설치

`yarn add immer`

#### immer 불러오기

immer 를 사용하기 위해 불러온다. 일반적으로 `produce` 라는 이름으로 불러온다.
`import produce from 'immer';`

#### 함수형 컴포넌트에서 immer 사용하기

`produce` 함수의 첫번째 피라미터는 수정하고 싶은 배열 혹은 객체 `state`, 두 번째 피라미터는 첫 번째 `state` 를 수정하는 함수이다.

아래 코드에서 `produce` 는 수정한 객체 `credentials` 를 반환하고, `setCredentials` 의해 값이 변경된다.
```jsx
const onChange = (e) => {
  const { value, name } = e.target;
  
  setCredentials(
    // produce 는 새로운 credentials 를 반환하고, setCredentials 에 의해 상태가 변경된다.
    produce(credentials, draft => {
      draft[name] = value
    })
  ) 
  
};
```
