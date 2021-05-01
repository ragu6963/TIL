# CSS

- 스타일, 레이아웃 등을 지정하는 언어

- 적용

  1. Inline : html tag 내에 style 속성에 적용. 테스트용
  2. 내부 참조 : `<style>` 태그 사이에 css 사용. 모든 html 파일에 적용할 수 없다.
  3. 외부 참조 : `<link>` 태그에 css 파일 경로를 추가. 유지보수 편함

- 선택자(Selector)

  - 특정한 요소`element` 를 선택하기 위해 사용

  - 기초 선택자

    - 전체 선택자`*`, 요소`element` 선택자
    - 아이디`#id` 선택자, 클래스`.class` 선택자, 속성`[속성]` 선택자

  - 고급 선택자

    - 자손 선택자 : `띄어쓰기`로 구분, 하위의 모든요소

      `article p { ... }`

    - 자식 선택자 : `>`로 구분, 바로 아래 요소

      `article > p { ... }`

    - 형제 선택자 : `~` 로 구분, 같은 계층(레벨)에 있는 요소

      `p ~ section {...}`

    - 인접형제 선택자 : `+` 로 구분, 바로 붙어있는 요소를 선택

      `selction + p { ... }`

 -  적용 순서
    1. `!important`
    2. Inline style
    3. id > class > element > 아래 코드 > 윗 코드

- css 상속

  - 상속되는 것  : `text` 관련`font,color,text-align`, opacity, visibility
  - 상속되지 않는 것 : box model 관련`w, h, p, m, border` position 관련

- css 단윈

  - px
  - % (기준 되는 사이즈에서의 배율)
  - em `상속 사이즈 기준 배율` / rem `브라우저 사이즈 기준 배율`
  - vh,vw
  - 색상 표현
    - HEX(#000, #000000)
    - RGB / RGBA
    - 색상명

- Box model

  - margin : 바깥 여백
  - border : 테두리 영역
  - padding : 내부 여백
  - contet : 글이나 이미지 요소

- box-sizing

  - content-box : 기본값, width의 너비는 content 영역을 기준으로 정한다.
  - border-box : width 의 너비는 테두리 기준으로 잡는다.

- 마진상쇄

  - 수지간의 형제 요소에서 주로 발생.
  - 둘 다 margin 을 가지고 있으면 큰 값만 적용이 된다.

  1. 큰 사이즈의 마진을 조정
  2. padding 을 이용

- Display

  - block : 가로폭 전체 차지

    - div, ul, ol, p, hr, form
    - 수평 정렬 `margin : auto;` 사용

  - inline : cotent의 너비만큼 가로폭 차지

    - width,height,margin-top,margin-bottom 지정 불가
      - line-height를 사용해 위아래 간격 조정

  - inline-block:

  - `display : none;`: 화면에서 완전히 삭제. vs `visibility : hidden;` : 화면에서 보이지는 않지만 공간은 차지 

    

- Postion
  - static(default)
    - `좌측상단`부터 배치가 된다.
    - 자식 요소라면 부모 요소를 기준으로 배치된다.
  - relative : `static` 이였을 때의 위치를 기준으로 `상대 위치` 이동
  - absolute : static이 아닌(`relative`) 가장 가까이 있는 `부모/조상` 요소를 기준으로 `절대 위치` 이동
  - fixed : 브라우저를 기준으로 `고정 위치` 이동, 스크롤을 해도 항상 같은 위치
  - sticky :  `relative` + `fixed`  기본적으로 상대 위치로 이동하지만 스크롤 이동으로 영역을 벗어나면 고정 위치