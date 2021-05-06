# Vue 인스턴스

###### 2021.05.06

### 인스턴스 생성

- `new Vue()` 명령어로 Vue 인스턴스를 생성한다. 

```vue
<script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
<script> 
    const vm = new Vue({
        // 옵션
    })
</script> 
```

### 데이터와 메소드

- `el`  : 인스턴스가 관리하는 DOM 엘리먼트를 선택한다.
- `data` : Vue의 반응형 시스템에 추가할 속성을 정으한다.
- `methods` : 인스턴스의 메서드들을 정의한다.

```vue
<div id="app">
    <p>{{ message }}</p>
    <button @click="clickBtn()">버튼</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
<script>
    const vm = new Vue({
        el: "#app",
        data: {
            message: "Hello! Vue.js!",
        },
        methods: {
            clickBtn: function () {
                this.message = "Click Button"
            },
        },
    })
</script>
```

### 라이프사이클 훅

- `created` : 인스턴스 생성 후 호출된다.
- `mounted` : 인스턴스가 el 옵션의 DOM 과 연결되면 호출된다.
- `updated` : 데이터가 변경되어 가상 DOM 재 렌더링 되면 호출된다.
- `destroyed` : 인스터스가 제거되면 호출된다.

> 모든 라이플사이클 훅 내에서 `this` 명령어는 vue 인스턴스를 가리킨다.

<img src="assets/Vue_인스턴스/lifecycle.png">

