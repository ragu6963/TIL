# 1767. 프로세서 연결하기

### 풀이

`조합` 알고리즘 활용

각 `core`에서 네 방향을 좌표를 벗어날 때 까지 탐색한다. 단, 다른 core 혹은 전선을 만나면 다른 방향 탐색을 한다.

한 방향에서 좌표를 벗어날 때 까지`벽에 닿을 때 까지` 탐색을 성공하면, 

다음 core, 이전 방문 배열`visited`에 현재 core의 전선 좌표`visit_list_temp`, 이전 전선 개수 + 현재 전선 개수 , 이전 core 개수 +1 를 인자로 재귀함수를 호출한다.

네 방향의 탐색을 끝내고, 아무방향도 탐색안하는`네 방향 모두 막혀있는`경우의 재귀함수를 호출한다.

모든 core를 탐색하면 2가지 경우에서 최소 전선 개수`min_line`를 갱신한다.

1. 현재 core 개수`croe_count`가 최대 코어 개수`max_core` 보다 클 때
2. 현재 core 개수와 최대 코어 개수가 같을 때, 현재 전선 개수`line_count`가 최소 전선 개수`min_line` 보다 작을 때

### 코드

```python
import sys

sys.stdin = open("input.txt", 'r')

dy = [0, 0, 1, -1]
dx = [1, -1, 0, 0]


# 좌표 범위 체크
def range_check(ny, nx):
	return 0 <= ny < N and 0 <= nx < N


# 각 core에서 4방향으로 전선 연결 해보기
# core index, 방문 표시, 전선개수, core 개수
def solve(index, visited, line_count, core_count):
	global min_line, max_core

	# 모든 전선 연결 끝
	if index == len_cores:
		# 코어 수가 최대 코어수보다 더 많을 때
		if core_count > max_core:
			max_core = core_count
			min_line = line_count

		# 코어 수가 최대 코어수랑 같을 때 전선 수가 최소 전선수 보다 작을 때
		if core_count == max_core and line_count < min_line:
			max_core = core_count
			min_line = line_count
		return

	y = cores[index][0]
	x = cores[index][1]

	# core기준 4방향 탐색
	for i in range(4):
		ny = y + dy[i]
		nx = x + dx[i]

		# 현재 core에서 방문 좌표 표시
		visit_list_temp = set()

		# 좌표가 범위 안이고, core 혹은 전선이 아닐 때
		while range_check(ny, nx) and (ny, nx) not in visited:
			visit_list_temp.add((ny, nx))
			ny = ny + dy[i]
			nx = nx + dx[i]

		# 범위를 벗어났다 -> 벽에 도달했다.
		if not range_check(ny, nx):
			line_count_temp = len(visit_list_temp)
            
			# 연결했을 때
			# 이전 방문 배열에 현재 core 방문 좌표 추가해서 재귀 호출
			solve(index + 1, visited | visit_list_temp, line_count + line_count_temp, core_count + 1)

	# 연결안했을 때
	solve(index + 1, visited, line_count, core_count)


T = int(input())
for t in range(T):
	N = int(input())
	graph = []
	cores = []
	# 최대 코어 ,최소 라인
	max_core = 0
	min_line = 10e9

	# 벽에 붙어 있는 core 개수
	wall_core = 0
	# core, 전선 방문 표시
	visited = set()

	for y in range(N):
		temp = list(map(int, input().split()))
		graph.append(temp)
		for x in range(N):
			if temp[x] == 1:
				# core 방문 추가
				visited.add((y, x))
				# 벽에 붙어있으면 위치 저장 X
				if x == 0 or x == N - 1 or y == 0 or y == N - 1:
					# 벽에 붙어있는 core 개수 증가
					wall_core += 1
					continue

				# 벽에 안 붙어있으면 위치 저장
				cores.append((y, x))

	# 코어 전체 개수
	len_cores = len(cores)

	solve(0, visited, 0, wall_core)
	print("#{} {}".format(t + 1, min_line))

```

