# [Silver I] 1697. 숨바꼭질 

### 사용 알고리즘 : BFS

### 풀이방식

완전탐색문제

bfs를 사용해 한 index에서 3가지 연산(+1 , -1, *2)을 모두 수행한다.

방문 여부를 확인해서 cost 가 가장 작을 때만 방문한다.



### PYTHON 코드

```python
import sys
from collections import deque

sys.stdin = open("./input.txt", "r")
input = sys.stdin.readline

n, k = list(map(int, input().split()))

MAX = 100001


def bfs(start, end):
    visit = set()
    visit.add(start)
    queue = deque()
    queue.append([start, 0])

    while queue:
        index, cost = queue.popleft()
        if index == end:
            return cost

        indexA = index + 1
        indexB = index - 1
        indexC = index * 2

        if indexA < MAX and indexA not in visit:
            queue.append([indexA, cost + 1])
            visit.add(indexA)

        if indexB >= 0 and indexB not in visit:
            queue.append([indexB, cost + 1])
            visit.add(indexB)

        if indexC < MAX and indexC not in visit:
            queue.append([indexC, cost + 1])
            visit.add(indexC)


cost = bfs(n, k)
print(cost)
```

