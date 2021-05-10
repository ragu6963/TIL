### 1. Vue 컴포넌트의 data Option

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
      msg: "aaa",
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