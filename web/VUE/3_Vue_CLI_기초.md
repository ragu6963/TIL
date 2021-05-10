### 1. 설치 과정

#### - Node.js 설치

[링크](https://nodejs.org/ko/)

#### - Vue CLI 설치

```bash
npm install -g @vue/cli
```
#### - Vue CLI 프로젝트 생성
```bash
vue create my-project
```

#### - Vue CLI 프로젝트 실행

```bash
npm run serve
```

---

### 2. Vue 컴포넌트 기본 구조

```vue
<template>
  <div></div>
</template>

<script>
export default {
  name: "ComponentName",
};
</script>

<style>
</style>
```

---

### 3. 컴포넌트 안에 컴포넌트 넣기

```vue
//  src/components/SecondComponent.vue
<template>
  <div>
    <h2>자식컴포넌트</h2>
  </div>
</template>

<script>
export default {
  name: "SecondComponent",
};
</script>

<style>
</style>
```

```vue
// src/views/NewComponent.vue
<template>
  <div>
    <h1>부모컴포넌트</h1>
    <SecondComponent />
  </div>
</template>

<script>
import SecondComponent from "@/components/SecondComponent.vue";
export default {
  name: "NewComponent",
  components: {
    SecondComponent,
  },
};
</script>

<style>
</style>
```



