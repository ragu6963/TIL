## DTO(Data Transfer Object)

계층간 데이터 교환을 위한 객체

- DB > Service / Controller 등으로 보낼 때 사용하는 객체
- 로직이 없는 순수한 데이터 객체이며, getter 메서드만을 가진다.



일반적으로 각 로직에 해당하는 Dto 클래스 파일을 만들어서 관리한다.
하지만 프로젝트가 커지면 그만큼 Dto 클래스 파일이 많아지고, 관리가 어려워진다,

inner Class 로 관리하면 하나의 클래스 파일에 모든 Dto를 관리 할 수 있다.

> 기존 방법

```java
// PostsResponseDto 

import lombok.Getter;
import org.example.springboot.domain.posts.Posts;

@Getter
public class PostsResponseDto {
    private Long id;
    private String title;
    private String content;
    private String author;

    public PostsResponseDto(Posts entity){
        this.id = entity.getId();
        this.title = entity.getTitle();
        this.content = entity.getContent();
        this.author = entity.getAuthor();
    }
}
```

```java
// PostsSaveRequestDto

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.example.springboot.domain.posts.Posts;

@Getter
@NoArgsConstructor
public class PostsSaveRequestDto {
    private String title;
    private String content;
    private String author;

    @Builder
    public PostsSaveRequestDto(String title,String content, String author){
        this.title = title;
        this.content = content;
        this.author = author;
    }

    public Posts toEntity(){
        return Posts.builder()
                .title(title)
                .content(content)
                .author(author)
                .build();
    }
}
```

```java
// PostsUpdateRequestDto

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class PostsUpdateRequestDto {
    private String title;
    private String content;

    @Builder
    public PostsUpdateRequestDto(String title, String content, String author){
        this.title = title;
        this.content = content;
    }
}
```

---

> Inner Class 이용

```java
// PostsDto

package org.example.springboot.web.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor; 

import org.example.springboot.domain.posts.Posts;

public class PostsDto {

    @Getter
    public static class Response{
        private Long id;
        private String title;
        private String content;
        private String author;

        public Response(Posts entity){
            this.id = entity.getId();
            this.title = entity.getTitle();
            this.content = entity.getContent();
            this.author = entity.getAuthor();
        }
    }

    @Getter
    @NoArgsConstructor
    public static class Save{
        private String title;
        private String content;
        private String author;

        @Builder
        public Save(String title,String content, String author){
            this.title = title;
            this.content = content;
            this.author = author;
        }

        public Posts toEntity(){
            return Posts.builder()
                    .title(title)
                    .content(content)
                    .author(author)
                    .build();
        }
    }

    @Getter
    @NoArgsConstructor
    public static class Update{
        private String title;
        private String content;

        @Builder
        public Update(String title, String content, String author){
            this.title = title;
            this.content = content;
        }
    }
}
```



위의 예제처럼 하나의 클래스 파일에 Dto를 관리해서 Dto 클래스 파일이 늘어나는 것을 방지하고, 관리를 용이하게 할 수 있다



```java
// conroller
import lombok.RequiredArgsConstructor;

import org.example.springboot.service.posts.PostsService;
import org.example.springboot.web.dto.PostsDto;
import org.springframework.web.bind.annotation.*;

@RequiredArgsConstructor
@RestController
public class PostsApiContorller {
    private final PostsService postsService;

    @PostMapping("/api/v1/posts")
    public Long save(@RequestBody PostsDto.Save requestDto) {
        return postsService.save(requestDto);
    }

    @PutMapping("/api/v1/posts/{id}")
    public Long update(@PathVariable Long id, @RequestBody PostsDto.Update requestDto) {
        return postsService.update(id, requestDto);
    }

    @GetMapping("/api/v1/posts/{id}")
    public PostsDto.Response findById(@PathVariable Long id) {
        return postsService.findById(id);
    } 
}
```

```java
// srevice
import lombok.RequiredArgsConstructor;

import org.example.springboot.domain.posts.Posts;
import org.example.springboot.domain.posts.PostsRepository;
import org.example.springboot.web.dto.PostsDto;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@RequiredArgsConstructor
@Service
public class PostsService {

    private final PostsRepository postsRepository;

    @Transactional
    public Long save(PostsDto.Save requestDto){
        return postsRepository.save(requestDto.toEntity()).getId();
    }

    @Transactional
    public Long update(Long id, PostsDto.Update requestDto){
        Posts posts = postsRepository.findById(id).orElseThrow(()->
                new IllegalAccessError("해당 게시글이 없습니다. id = " +id));

        posts.update(requestDto.getTitle(),requestDto.getContent());

        return id;
    }

    public PostsDto.Response findById (Long id){
        Posts entity = postsRepository.findById(id)
                .orElseThrow(() ->  new IllegalAccessError("해당 게시글이 없습니다. id = " +id));

        return new PostsDto.Response(entity);
    }
}
```

