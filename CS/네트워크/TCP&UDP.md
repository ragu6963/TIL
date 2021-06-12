### TCP

컴퓨터간의 데이터 통신을 하기위한 프로토콜 중 하나이며 `연결형` 프로토콜이다. 
즉, 연결된 상태에서 데이터를 주고받는 프로토콜이다.
UDP와 다르게 신뢰성을 보장하는 프로토콜이다.

##### 3-Way-Handshake : 연결설정

- Client > Server 연결 요청 SYN(Synchronization)
- Server > Client  연결 응답 및 포트를 열어달라는 요청 ACK(Acknowledgement) + SYN
- Client > Server 연결 응답 ACK

##### 4-Way-HandShake : 연결해제

- Client > Server 해제 요청 FIN
- Server > Client 응답 및 진행 중인 통신이 끝날 때 까지 대기 ACK
- Server > FIN 해제 요청 FIN
- Client > Server ACK

##### 사용 예시

- HTTP 통신
- 이메일
- 파일 전송
- 순서대로 도착해야하는 상황

### UDP

TCP와 다르게 신뢰성은 보장하지 못하지만 더 빠른 `비연결형` 프로토콜이다.

비연결성이고, 순서화 되지 않다.

##### 사용 예시

- `DNS`
- 실시간 동영상(스트리밍)
- 게임

##### DNS(Domain Name System)

IP 네트워크에서 사용하는 시스템으로 영문/한글 주소를 IP로 변환해주는 시스템.

DNS를 운영하는 서버를 네임서버(Name Server)라고 한다.

