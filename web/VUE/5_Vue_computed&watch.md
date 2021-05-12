### computed

> 특정 `data`를 가공하는 도구

- `computed` 함수는 반환`return`이 필수이다.
- `computed` 는 대상을 저장하기 때문에 대상이 변경될 때만 함수가 실행된다.

```vue
<template>
  <div>
    <p>{{ msg }}</p>
    <p>{{ reversedMsg }}</p>
  </div>
</template>

<script>
export default {
  name: "TestComponent",
  data() {
    return {
      msg: "Message",
    };
  },
  computed: {
    reversedMsg() {
      return this.msg.split("").reverse().join("");
    },
  },
};
</script>

<style>
</style>
```



### watch

> 특정 `data` 를 감시하면서 데이터가 변경되면 함수를 실행한다.

- `computed`와 다른점은 반환`return`이 없어도 된다는 점이다.
- 특정 `data` 의 변화와 함께 액션이 필요할 때 사용한다.

```vue
<template>
  <div>
    <input type="text" v-model="msg" />
    <p>{{ reversedMsg }}</p>
  </div>
</template>

<script>
export default {
  name: "TestComponent",
  data() {
    return {
      msg: "Message",
      reversedMsg: "",
    };
  },
  watch: {
    msg(newValue) {
      this.reversedMsg = newValue.split("").reverse().join("");
    },
  },
};
</script>

<style>
</style>
```

