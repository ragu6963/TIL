# 템플릿 문법

###### 2021.05.06

### 보간법

1. 문자열 : `{{ }}` (이중 중괄호)를 사용한 텍스트 보간법이다.

```vue
<span> {{ message }} </span>
<script>
	const vm = new Vue({
        data: {
            message:"message",
        }
    })
</script>
```

2. 속성 : `v-bind 디렉티브` 를 사용하여 요소 속성을 정의할 수 있다.

```vue
<div v-bind:id="dynamicId"></div>
<!-- v-bind: 를 : 로 줄여서 사용할 수 있다. -->
<div :id="dynamicId"></div>
<script>
	const vm = new Vue({
        data: {
            dynamicId:"dynamicId",
        }
    })
</script>
```

3. Javascript 표현식 : Vue.js 는 모든 `데이터 바인딩` 내에서 JS 표현식을 지원한다.	
   - 데이터 할당은 불가능하다.	
   - 조건문은 불가능하다. 삼항 연산자를 사용해야한다.

```vue
{{ number + 1 }}

{{ ok ? 'YES' : 'NO' }}

{{ message.split('').reverse().join('') }}

<div v-bind:id="'list-' + id"></div>
```

---

### 동적 전달인자

Vue 인스턴스 속성에 따라 요소의 `속성명`을 동적으로 변경할 수 있다.

> 공식가이드에는 attributeName 를 사용하였으나 오류가 발생한다.

```vue
<div id="app">
  <a v-bind:[attributename]="url"> ... </a>
</div>

<script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
<script>
  const app = new Vue({
    el: "#app",
    data: {
      url: "https://naver.com",
      attributename: "href",
    },
  })
</script>
```

```vue
<div id="app">
  <button v-on:[eventname]="doSomething">...</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
<script>
  const app = new Vue({
    el: "#app",
    data: {
      eventname: "click",
    },
  })
</script>
```

---

### 수식어

`.(온점)` 으로 표시되는 특수 접미사이다.

> .prevent 수식어는 event.preventDefault()를 호출한다.

```html
<form v-on:submit.prevent="onSubmit"> ... </form>
```

