WHITE, RED, BLUE, OUTSIDE = 0, 1, 2, 3
RIGHT, LEFT, UP, DOWN = 1, 2, 3, 4
dirs = [[], [0, 1], [0, -1], [-1, 0], [1, 0]]

N, K = list(map(int, input().split()))


def check_end():
    for y in range(N):
        for x in range(N):
            if len(hores_on_board[y][x]) >= 4:
                return True
    return False


def is_inside(y, x):
    return -1 < y and y < N and -1 < x and x < N


def move_horse(k):
    y, x, direction = horses[k]
    index_on_board = hores_on_board[y][x].index(k)
    on_horses = hores_on_board[y][x][index_on_board:]

    ny = y + dirs[direction][0]
    nx = x + dirs[direction][1]

    for horse in on_horses:
        horses[horse][0] = ny
        horses[horse][1] = nx
        hores_on_board[y][x].remove(horse)
        hores_on_board[ny][nx].append(horse)


def reverse_order(k):
    y, x, _ = horses[k]
    index_on_board = hores_on_board[y][x].index(k)
    hores_on_board[y][x][index_on_board:] = reversed(hores_on_board[y][x][index_on_board:])


def reverse_direction(k):
    y, x, direction = horses[k]

    if direction == RIGHT:
        direction = LEFT

    elif direction == LEFT:
        direction = RIGHT

    elif direction == UP:
        direction = DOWN

    elif direction == DOWN:
        direction = UP

    horses[k] = [y, x, direction]


def check_next_color(k):
    y, x, direction = horses[k]
    ny = y + dirs[direction][0]
    nx = x + dirs[direction][1]

    if not is_inside(ny, nx):
        return OUTSIDE

    return board[ny][nx]


board = []
horses = []
hores_on_board = [[[] for _ in range(N)] for _ in range(N)]

for _ in range(N):
    board.append(list(map(int, input().split())))


for i in range(K):
    y, x, direction = list(map(int, input().split()))
    y -= 1
    x -= 1
    horses.append([y, x, direction])
    hores_on_board[y][x].append(i)

turn = 0
while True:
    turn += 1

    for k in range(K):
        next_color = check_next_color(k)

        if next_color == WHITE:
            move_horse(k)

        elif next_color == RED:
            move_horse(k)
            reverse_order(k)

        elif next_color == BLUE or next_color == OUTSIDE:
            reverse_direction(k)
            next_color = check_next_color(k)

            if next_color == WHITE:
                move_horse(k)
            elif next_color == RED:
                move_horse(k)
                reverse_order(k)

        if check_end():
            break

    if check_end():
        break

    if turn >= 1000:
        break

if turn >= 1000:
    print(-1)
else:
    print(turn)
