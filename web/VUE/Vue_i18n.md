## i18n
i18n 은 `Internationalization` 의 약어로 국제화를 의미한다.
i18n 은 `SW 국제화`를 의미한다.

## SW 국제화
`SW 국제화` 는 SW 가 특정 지역 언어에 종속되지 않고, 작동하도록 설계하고 개발하는 과정이다.

## vue-i18n
`Vue` 에서 i18n 을 쉽게 사용할 수 있게 해주는 모듈

## 사용법
### vue-i18n 설치
```bash
npm install vue-i18n
# or
yarn add vue-i18n
```

### 기본설정

`src/i18n.js`
기초적인 i18n 설정을 해주는 파일이다.
```js
import Vue from 'vue'
import VueI18n from 'vue-i18n'
import en from '@/locales/en.json'
import ko from '@/locales/ko.json'

Vue.use(VueI18n)

export default new VueI18n({ 
  locale: 'ko',
  fallbackLocale: 'ko',
  messages: { en, ko }
})
```

`src/locales/ko.json` 
사용할 언어들의 일종의 언어팩이다.`{"key":"value",...}` 형태로 작성한다.
```json
{
  "message": "안녕 i18n !!"
}
```
`src/locales/en.json`
```json
{
  "message": "hello i18n !!"
}
```
`src/main.js`
i18n 을 사용하기 위해 Vue 생성자의 인자로 넣어준다.
```js
import Vue from 'vue'
import App from './App.vue'
import i18n from './i18n'

Vue.config.productionTip = false

new Vue({
  i18n,
  render: h => h(App)
}).$mount('#app')
```

### template 에서 사용
`$t('key')` 형태로 사용한다.
```vue
<template>
  <div class="hello">
    <p>{{ $t('message') }}</p>
  </div>
</template>

<script>

export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>
```

## 언어변경
`$i18n.locale` 의 값을 변경한다.
```vue
<template>
  <div class="hello">
    <p>{{ $t('message') }}</p>
    <button @click="changeLocale">변경</button>
  </div>
</template>

<script>

export default {
  name: 'HelloWorld',
  props: {
    msg: String
  },
  methods: {
    changeLocale() {
      if (this.$i18n.locale === 'en') return (this.$i18n.locale = 'ko')
      this.$i18n.locale = 'en'
    }
  },
}
</script>
```
