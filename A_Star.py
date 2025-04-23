import heapq

# 曼哈顿距离
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* 算法
def astar(grid, start, end):
    rows = len(grid)            # 矩阵行数
    cols = len(grid[0])         # 矩阵列数
    open_heap = list()        # 优先堆
    came_from = dict()      # 父节点字典
    closed = set()              # 已探索集合

    # 始点入堆
    heapq.heappush(open_heap, (0, start[0], start[1]))
    # 实际代价矩阵 g
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score[start] = 0
    # 综合代价矩阵 f = 启发式代价 h + 实际代价 g
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    f_score[start] = heuristic(start, end) + 0

    while open_heap:
        # 取出堆顶节点
        _ , x , y = heapq.heappop(open_heap)
        current = (x, y)
        # 忽略已探节点
        if current in closed: continue
        # 搜索完成，开始回溯
        if current == end:
            path = list()
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        # 继续探索
        closed.add(current)
        for neighbor in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            x, y = neighbor
            if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic((x, y), end)
                    f_score[neighbor] = tentative_g + heuristic((x, y), end)
                    heapq.heappush(open_heap, (f, x, y))
    return None

if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    path = astar(grid=grid, start=(0,0) , end=(2,2))
    print("路径:", path)
