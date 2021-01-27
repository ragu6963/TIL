# [Gold V] 1916. 최소비용 구하기

### [BAEKJOON](https://www.acmicpc.net/problem/1916)

### 사용 알고리즘 : 다익스트라

### 풀이방식

평범한 다익스트라 문제

### PYTHON 코드

```python
import sys
import heapq
from collections import defaultdict

sys.stdin = open("./input.txt", "r")
input = sys.stdin.readline
INF = int(1e9)

N = int(input())
M = int(input())

graph = defaultdict(list)
distance = [INF] * (N + 1)

for _ in range(M):
    u, v, w = list(map(int, input().split()))
    graph[u].append([v, w])

start, end = list(map(int, input().split()))


def Dijkstra(start):
    heap = list()
    distance[start] = 0
    heapq.heappush(heap, [0, start])

    while heap:
        distA, indexA = heapq.heappop(heap)

        if distA > distance[indexA]:
            continue

        for indexB, distB in graph[indexA]:
            totalDist = distA + distB
            if totalDist < distance[indexB]:
                distance[indexB] = totalDist
                heapq.heappush(heap, [totalDist, indexB])


Dijkstra(start)
print(distance[end])
```

