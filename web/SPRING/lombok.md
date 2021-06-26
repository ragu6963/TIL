## lombok 롬복

Getter, Setter, 기본생성자, toString 등을 어노테이션으로 자동생성 해주는 라이브러리

### Gradle 추가

```
plugins {
    id 'org.springframework.boot' version '2.5.2'
    id 'io.spring.dependency-management' version '1.0.11.RELEASE'
    id 'java'
}

group 'org.example'
version '1.0-SNAPSHOT'
sourceCompatibility = 1.8

repositories {
    mavenCentral()
    jcenter()
}

dependencies {
    implementation('org.springframework.boot:spring-boot-starter-web')
    implementation('org.springframework.boot:spring-boot-starter-mustache')
    testImplementation('org.springframework.boot:spring-boot-starter-test')

    // lombok
    implementation('org.projectlombok:lombok')
    annotationProcessor('org.projectlombok:lombok')
    testImplementation('org.projectlombok:lombok')
    testAnnotationProcessor('org.projectlombok:lombok')
}

test {
    useJUnitPlatform()
}
```

### 롬복 적용 - Dto 코드

> src > main > java > 패키지.web.dto > HelloResponseDto.java

```java
package org.example.springboot.web.dto;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter // 모든 필드의 get 메소드를 생성한다. 
@RequiredArgsConstructor // final 필드의 생성자를 생성한다. final 이 없으면 생성자에 포함되지 않는다.
public class HelloResponseDto {
    private final String name;
    private final int amount;
}
```

### Dto 테스트 코드

> src > test > java > 패키지.web.dto > HelloResponseDtoTest.java

```java
package org.example.springboot.web.dto;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class HelloResponseDtoTest {
    @Test
    public void 롬복_기능_테스트(){
        //given
        String name = "test";
        int amount = 1000;

        // when
        HelloResponseDto dto = new HelloResponseDto(name,amount);

        // then
        // assertThat : 검증 대상을 메소드 인자로 받는다.
        // 메소드 체이닝이 지원되는 isEqualTo와 같이 연결해서 사용할 수 있다.
        // isEqualTo : 비교 메서드
        assertThat(dto.getName()).isEqualTo(name);
        assertThat(dto.getAmount()).isEqualTo(amount);
    }
}
```

### Controller 코드

```java
@GetMapping("/hello/dto")
public HelloResponseDto hellodDto(@RequestParam("name") String name,
                                  @RequestParam("amount") int amount){
    // @RequestParam : API로 넘긴 파라미터를 가져오는 어노테이션
    // @RequestParam("name") String name : 쿼리스트링의 변수 name의 값을 문자열 변수 name에 저장한다.
    return new HelloResponseDto(name, amount);
}
```

### Controller 테스트 코드

```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.hamcrest.Matchers.is;

@ExtendWith(SpringExtension.class) // junit5
@WebMvcTest(controllers = HelloController.class)
public class HelloControllerTest
{ 
    @Test
    public void helloDto가_리턴된다() throws Exception{
        String name ="hello";
        int amount = 1000;

        // param : 쿼리 스트링 설정. 값은 String만 허용한다.
        /*
        jsonPath
            JSON 응답값을 필드별로 검증하는 메소드
            $ 를 기준으로 필드명을 명시한다.
        */
        mvc.perform(
                get("/hello/dto")
                        .param("name",name)
                        .param("amount",String.valueOf(amount)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name",is(name)))
                .andExpect(jsonPath("$.amount",is(amount)));

    }

}

```

