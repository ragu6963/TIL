### 객체 지향 프로그래밍(Object Oriented Programing)

프로그램을 `객체(object)` 를 기본 단위로 하고, 객체의 상호작욕으로 서술하는 방식이다.

> 객체 = 변수(특성) + 메서드(기능) 

### 특징

#### 캡슐화

변수와 메서드를 하나의 단위로 묶는 것을 의미한다. 즉 번들링(Bundling) 하는 것이다.

일반적으로 클래스(Class)로 번들링을 구현하고, 인스턴스를 통해 변수와 메서드에 접근한다.

##### 정보 은닉

내부 구현의 노출을 최소화해서 모듈의 내부를 감추는 것이다.

---

#### 상속

특정 클래스(부모 클래스)의 특성과 기능을 그대로 물려받아 사용하는 것이다. 이 때 물려받는 클래스를 자식 클래스라고 한다.

자식 클래스에서 부모 클래스의 특정 기능을 수정해서 사용하는 것을 `오버라이딩(Overriding)` 이라고 한다.

---

#### 다형성

같은 형태 다른 기능을 의미한다.

- 오버라이딩 : 부모 클래스의 메서드를 자식 클래스에서 재정의 해서 사용하는 것을 말한다.

- 오버로딩 : 메서드의 이름은 같지만 매개변수의 `갯수와 타입`이 다르게 사용하는 방법

---

#### 추상화

클래스를 정의하는 것을 추상화라고 정의할 수 있다.
즉, 공통의 속성과 기능을 묶어 이름을 정하고, 세부적인 사항은 제거하는 것.


