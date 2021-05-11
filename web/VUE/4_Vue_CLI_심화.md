### 1. data Option

> 컴포넌트에서 data 옵션을 사용할 때에는 함수 형태로 사용해야 한다. 
>
> 함수 형태를 사용하지 않으면 각 컴포넌트의 고유한 scope를 침범한다.

```vue
<template>
  <div>
    <p>{{ msg }}</p>
  </div>
</template>

<script>
export default {
  name: "NewComponent",
  data: function () {
    return {
      msg: "Data Option",
    };
  },
};
</script>

<style>
</style>
```

---

### 2. Props

> 부모 컴포넌트에서 자식 컴포넌트로 데이터를 전달하는 수단
>
> 단방향 데이터로 자식 컴포넌트에서 부모 컴포넌트로는 영향을 줄 수 없다.

> 부모 컴포넌트

```vue
<template>
  <div>
    <h1>부모컴포넌트</h1>
    <ChildComponent :msg="msg" />
  </div>
</template>

<script>
import ChildComponent from "../components/ChildComponent.vue";
export default {
  name: "ParentComponent",
  data() {
    return {
      msg: "This is Props",
    };
  },
  components: {
    ChildComponent,
  },
};
</script>

<style>
</style>
```

> 자식 컴포넌트

```vue
<template>
  <div>
    <h2>자식컴포넌트</h2>
    <p>{{ msg }}</p>
  </div>
</template>

<script>
export default {
  name: "ChildComponent",
  props: {
    msg: String,
  },
};
</script>

<style>
</style>
```

### 3. emit

> 자식 컴포넌트에서 부모 컴포넌트로 이벤트 신호를 보내는 수단

> 부모 컴포넌트

```vue
<template>
  <div>
    <h1>부모컴포넌트</h1>
    <!-- 자식 컴포넌트의 child-button-click 가 트리거 되면 childEmit 함수가 실행된다. -->
    <ChildComponent @child-button-click="childEmit" />
    <p>{{ childMsg }}</p>
  </div>
</template>

<script>
import ChildComponent from "../components/ChildComponent.vue";
export default {
  name: "ParentComponent",
  data() {
    return {
      childMsg: "",
    };
  },
  methods: {
    // 자식 컴포넌트의 childButtonClick 함수가 실행되면 결과적으로 childEmit이 실행된다.
    childEmit(msg) {
      this.childMsg = msg;
    },
  },
  components: {
    ChildComponent,
  },
};
</script>

<style>
</style>
```

> 자식 컴포넌트

```vue
<template>
  <div>
    <h2>자식컴포넌트</h2>
    <input type="text" v-model="msg" />
    <button @click="childButtonClick">버튼</button>
    <p>{{ msg }}</p>
  </div>
</template>

<script>
export default {
  name: "ChildComponent",
  data: function () {
    return {
      msg: "",
    };
  },
  methods: {
    childButtonClick() {
      // 부모 컴포넌트에 child-button-click 이벤트를 트리거 한다.
      // this.msg 를 인자로 보낸다.
      this.$emit("child-button-click", this.msg);
    },
  },
};
</script>

<style>
</style>
```

