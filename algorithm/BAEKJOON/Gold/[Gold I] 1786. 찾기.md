# [Gold I] 1786. 찾기 [작성중]

### 사용 알고리즘 : KMP 알고리즘

### 풀이

`KMP` 알고리즘 풀이 문제.

단어`P` 의 `LPS(Logest Prefix & Suffix)` 테이블을 구한다.

### 시간복잡도 : O(N+M) N : 원본 문장길이, M : 찾는 단어길이

### 코드

```python
# [Gold I] 1786. 찾기
# https://www.acmicpc.net/problem/1786
import sys

sys.stdin = open("./input.txt", "r")


def make_table():
    length = len(P)
    j = 0
    for i in range(1, length):
        while j > 0 and P[i] != P[j]:
            j = table[j - 1]
        if P[i] == P[j]:
            j += 1
            table[i] = j


def kmp():
    t_size = len(T)
    p_size = len(P)
    j = 0
    for i in range(t_size):
        while j > 0 and T[i] != P[j]:
            j = table[j - 1]

        if T[i] == P[j]:
            if j == p_size - 1:
                results.append(i - len(P) + 2)
                j = table[j]
            else:
                j += 1


results = []
T = input()
P = input()  # ABCDABD
table = [0 for _ in range(len(P))]
make_table()
kmp()
print(len(results))
for result in results:
    print(result)

```

