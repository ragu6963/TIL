### WAS

클라이언트의 동적 요청을 처리하기 위한 애플리케이션 서버(DB 조회, 로직 처리가 필요한 컨텐츠)

- 예시 : django, Spring
- django collectstaic 을 하는 이유는 정적 파일을 한 곳에 모아 Web Server(Nginx) 가 사용할 수 있게 하기 위해서이다.



### Web Server

클라이언트의 **요청(request)** 을 가장 먼저 처리한다.

- 정적(static) 응답 : html 이미지 js css 등등
- 동적(dynamic) 응답 : WAS 처리 부탁 

- 예시 : Apache, Nginx



### WSGI(Web Server Gateway Interface)

WAS(Nginx) 와 Web Server(Django)의 연결을 중계한다.

WAS와 Web Server는 서로를 모르기 때문에 서로가 알 수 있게 변환해준다. (HTTP 요청 -> Python , Python -> HTTP 응답)

- 예시 : uWSGI, Gunicorn

