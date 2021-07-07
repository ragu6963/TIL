### Props
> props는 속성을 나타내는 데이터이다. 출처 : [공식문서](https://ko.reactjs.org/docs/components-and-props.html)

props 는 **properties** 의 줄임말로 컴포넌트로 값을 전달할 때 사용한다.


### 기본 사용법
#### Props 전달
 컴포넌트 태그안에 ```key="value"``` 형태로 넣어서 사용한다.
```jsx
// App.js
function App() {
  return (
    <div>
      <Hello message="Hello React World" />
    </div>
  );
}
```


#### Props 사용
컴포넌트 함수의 파라미터로 통해서 사용할 수 있다.
props 는 객체 형태로 전달된다.
```message``` 의 값을 사용하고 싶으면 ```props.message``` 로 사용하면 된다.
```jsx
// Hello.js
const Hello = (props) => {
  return (
    <div>
      <h1>
        {props.message}
      </h1>
    </div>
  )
}
```

### 여러개의 Props와 구조 분해 할당

```jsx
// App.js
function App() {
  return (
    <div>
      <Hello message="Hello React World" color="red" />
    </div>
  );
}
```

```jsx
// Hello.js
const Hello = (props) => {
  return (
    <div>
      <h1 style={{ color: props.color }}>
        {props.message}
      </h1>
    </div>
  )
}
```
#### 구조 분해 할당
> **구조 분해 할당** 구문은 배열이나 객체의 속성을 해체하여 그 값을 개별 변수에 담을 수 있게 하는 JavaScript 표현식입니다. 출처 : [MDN](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment)

props 를 사용할 때 마다 ```props.``` 를 사용하는 대신 구조 분해 할당을 이용하면 간결한 코드를 작성 할 수 있다.

```jsx
// Hello.js
const Hello = ({ message, color }) => {
// 파라미터에 props 대신 구조 분해 할당을 이용하여 변수를 할당한다.
    return (
    <div>
      <h1 style={{ color: color }}>
        {message}
      </h1>
    </div>
  )
}
```

### defaultProps
컴포넌트에 props 를 지정하지 않았을 때 기본값을 설정하고 싶을 때 사용하는 옵션이다. 
```jsx
// App.js
function App() {
  return (
    <div>
      {/* message props 를 설정하지 않았다. */}
      <Hello color="red" />
    </div>
  );
}
```

```jsx
// Hello.js
const Hello = ({ message, color }) => {
  return (
    <div>
      <h1 style={{ color: color }}>
        {message}
      </h1>
    </div>
  )
}

// defaultProps 옵션으로 default 값을 설정한다.
Hello.defaultProps = {
  message: "This default value",
}
```
