### JSX

Javascript XML의 약자로  리액트에서 UI를 정의하기 위한 문법이다. 

생긴건 HTML 같이 생겼지만 실제로는 JavaScript 이다.

컴포넌트에서 XML 형태로 코드를 작성하면 babel 이 JSX 를 JS 로 변환해준다.

JSX 사용이 필수는 아니지만 대부분 사용하기 때문에 필수라고 볼 수 있다.

#### 변환예시

변환 사이트 : https://babeljs.io/repl

![](assets/(3) JSX/image.png)

### JSX 규칙

#### 1. 무조건 닫는 태그가 필요하다.
Ex) ```<input/>, <br/>```

#### 2. 여러 태그들이 있으면 무조건 감싸주는 태그가 필요하다.
Ex) 틀린 코드
```jsx
return(
	<div></div>
  	<div></div>  
)
```

Ex) 맞는 코드
```jsx
return(
  <div>
      <div></div>
      <div></div>  
  </div>	
)
```
#### 3. JS 값을 사용할 때에는 중괄호{} 로 묶어서 표현한다.
```jsx
const Hello = () => {
  const message = "Hello";

  return (
    <div>
      <h1>
        This is {message} Component
      </h1>
    </div>
  )
}
```

#### 4. style 은 객체를 만들어 사용해야한다.
Ex) 틀린 코드
```jsx
return(
  <div>
    <span style="color : red;">bad</span>
  </div>
)
```

Ex) 맞는 코드
```jsx
const style ={
  color: green;
}
return(
  <div>
    <span style={style}>good</span>
  </div>	
)
```

#### 5. css 속성 중 - 로 구분되는 속성은 Camel Case 로 사용해야한다.
Ex) ```background-color``` -> ```backgorundColor```

#### 6. class 는 className 으로 사용한다.
```css
/* Hello.css */
.color-red{
  color:red;
}
```

```jsx
// Hello.js
import React from 'react'
import "./Hello.css"

const Hello = () => {
  const message = "Hello";

  return (
    <div>
      <h1 className="color-red">
        This is {message} Component
      </h1>
    </div>
  )
}

export default Hello;
```

> class 로 사용해도 작동은 하나 경고를 띄우니 꼭 className 으로 사용하자.

#### 7. 주석
```{/* Comment */}```
로 표현해야 브라우저에 노출되지 않는다.
태그 내부에도 주석이 사용가능하다.
```jsx
<h1 // Comment in Tag 
  > ... 
</h1>
```

> 제일 중요한 것은 태그는 무조건 닫아야하고, 감싸는 태그가 있어야한다는 것! 
