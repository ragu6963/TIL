# 간단한 Counter App 구현

1. vuex 설치

```bash
npm install vuex
```

2. ButtonComponet, CounterComponent 생성

> ButtonComponet.vue

```vue
<template>
  <div>
    <!-- value 만큼 counter의 값 증가/감소 -->
    <input type="number" v-model="counterSize" />
    <br />
    <!-- 클릭 한 버튼에 따라 counter의 값 증가/감소 -->
    <button @click="plusCounter">+</button>
    <button @click="minusCounter">-</button>
  </div>
</template>

<script>
export default {
  name: "ButtonComponent",
  data() {
    return {
      counterSize: 1,
    };
  },
  methods: {
    plusCounter() {
      // store 의 plusCounter action 호출
      // this.counterSize 를 인자로 전달
      this.$store.dispatch("plusCounter", this.counterSize);
    },
    minusCounter() {
      // store 의 minusCounter action 호출
      // this.counterSize 를 인자로 전달
      this.$store.dispatch("minusCounter", this.counterSize);
    },
  },
};
</script>

<style>
</style>
```

> CounterComponent.vue

```vue
<template>
  <div>
    <p>
      <!-- store 의 counter state 바인딩 -->
      {{ this.$store.state.counter }}
    </p>
  </div>
</template>

<script>
export default {
  name: "CounterComponent",
};
</script>

<style>
</style>
```

3. `store 구조` index.js 생성

```js
// src/store/index.js
import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    counter: 0,
  },
  mutations: {
    // 각 mutation 은 state 수정
    PLUS_COUNTER(state, counterSize) {
      state.counter += parseInt(counterSize)
    },
    MINUS_COUNTER(state, counterSize) {
      state.counter -= parseInt(counterSize)
    },
  },
  actions: {
    // 각 action 은 commit 메서드로 mutation 을 호출
    plusCounter({ commit }, counterSize) {
      commit('PLUS_COUNTER', counterSize);
    },
    minusCounter({ commit }, counterSize) {
      commit('MINUS_COUNTER', counterSize);
    }
  },
});
```

4. store 할당

```js
// src/main.js

import Vue from 'vue'
import App from './App.vue'
import store from './store'

Vue.config.productionTip = false

new Vue({
  store,
  render: h => h(App),
}).$mount('#app')
```

