# [Gold III] 1238. 파티

### 사용 알고리즘 : 다익스트라

### 풀이방식

일반적인 다익스트라 문제와 다른 점은 출발점에서 도착점으로 갔다가 돌아오는 것도 생각해야 한다는 점이다.

그래서 정방향 그래프와 역방향 그래프 , 정방향 거리 배열과 역방향 거리 배열을 생성했다.

다익스트라로 정방향과 역방향 거리를 계산해 두 거리를 더한 값 중에 가장 큰 값이 문제의 답이다.   

### PYTHON 코드

```python
import sys
import heapq
from collections import defaultdict


sys.stdin = open("./input.txt", "r")


def Dijkstra(graph, distance):
    distance[X] = 0
    heap = list()
    heapq.heappush(heap, (X, 0))

    while heap:
        indexA, distA = heapq.heappop(heap)
        if distA > distance[indexA]:
            continue

        for indexB, distB in graph[indexA]:
            distTotal = distA + distB

            if distTotal < distance[indexB]:
                distance[indexB] = distTotal
                heapq.heappush(heap, [indexB, distTotal])


input = sys.stdin.readline

INF = 1e9

N, M, X = list(map(int, input().split()))
distance = [INF] * (N + 1)
distanceReverse = [INF] * (N + 1)
graph = defaultdict(list)
graphReverse = defaultdict(list)

for _ in range(M):
    u, v, w = list(map(int, input().split()))
    graph[u].append([v, w])
    graphReverse[v].append([u, w])


Dijkstra(graph, distance)
Dijkstra(graphReverse, distanceReverse)
result = -1
for index in range(1, N + 1):
    total = distance[index] + distanceReverse[index]
    if total > result:
        result = total
print(result)

```

