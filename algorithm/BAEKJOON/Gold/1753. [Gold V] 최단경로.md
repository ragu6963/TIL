# [Gold V] 1753. 최단경로 

### [BAEKJOON](https://www.acmicpc.net/problem/1753)

### 사용 알고리즘 : 다익스트라

### 풀이방식

평범한 다익스트라 문제

### PYTHON 코드

```python
import sys
import heapq
from collections import defaultdict

sys.stdin = open("./input.txt", "r")

INF = 1e9

V, E = list(map(int, input().split()))
start = int(input())
distance = [INF] * (V + 1)
heap = []
graph = defaultdict(list)

for _ in range(E):
    # u -> v , 비용 w
    u, v, w = list(map(int, input().split()))
    graph[u].append([v, w])


def Dijkstra(start):
    distance[start] = 0
    heapq.heappush(heap, (0, start))
    while heap:
        dist, now = heapq.heappop(heap)

        if distance[now] < dist:
            continue

        # i는 현재 노드와 인접한 다른 노드들
        # i[0]은 다른 노드 번호
        # i[1]은 다른 노드로 가는 비용
        for i in graph[now]:
            # cost는 현재 노드까지 비용 + 다음 노드로 가는 비용
            cost = dist + i[1]
            # 기존 비용 보다 cost(현재 노드까지 비용 + 다음 노드로 가는 비용) 가 작을 때
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(heap, (cost, i[0]))


Dijkstra(start)
for index in range(1, V + 1):
    cost = distance[index]
    if cost == INF:
        print("INF")
    else:
        print(cost)

```

