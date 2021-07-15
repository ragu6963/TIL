## Hook
> Hook은 React 버전 16.8부터 React 요소로 새로 추가되었습니다. Hook을 이용하여 기존 Class 바탕의 코드를 작성할 필요 없이 상태 값과 여러 React의 기능을 사용할 수 있습니다.
> 출처 : [공식문서](https://ko.reactjs.org/docs/hooks-intro.html)

이전 글에서 사용한 `useState` 도 Hook 의 일종이다. 이외에 어떤 Hook 이 있는 알아보자.


## Effect Hook
`side effects` 를 수행할 수 있게 해주는 Hook 이다.
클래스형 컴포넌트의 `componentDidMount` 와 `componentDidUpdate` 와 `componentWillUnmount` 를 합친 형태로도 볼 수 있다.

### 마운트(mount)에만 실행하고 싶을 때
두 번째 파라미터에 빈 배열`[]`을 넣어준다.
```jsx
useEffect(() => {
  console.log("마운트에만 실행 합니다.");
},[]);
```

### 특정 값(state, props)이 변경을 감지하고 실행하고 싶을 때
두 번째 파라미터에 감지하고 싶은 값들이 들어간 배열`[값,...]`을 넣어준다.

```jsx
const [credentials, setCredentials] = useState({
  username: '',
  password: '',
  passwordconfirm: '',
});

useEffect(() => {
  console.log("credentials가 변경되면 실행됩니다.")
  console.log(credentials)
}, [credentials])
```

### 정리(clean-up) 함수
clean-up 함수는 컴포넌트가 언마운트(unmount)되기 전, 업데이트 되기 전에 수행된다.
```jsx
useEffect(() => {
  console.log("credentials가 변경되면 실행됩니다.")
  return () => {
    console.log('cleanup 실행');
  };
}, [credentials])
```

만약, 언마운트 때만 실행하고 싶다면 두번째 파라미터에 빈 배열을 넣으면 된다.
```jsx
useEffect(() => {
  console.log("credentials가 변경되면 실행됩니다.")
  return () => {
    console.log('cleanup 실행');
  };
}, [])
```
