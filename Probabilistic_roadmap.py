import math
import heapq
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# 算法配置参数
config = {
    'x_range': (0, 10),             # x 值域
    'y_range': (0, 10),             # y 值域
    'obstacles': [                   # 障碍物
        {'type': 'rect', 'x': 3, 'y': 3, 'width': 2, 'height': 2},
        {'type': 'rect', 'x': 6, 'y': 6, 'width': 2, 'height': 2}
    ],
    'num_samples': 1000,      # 随机散点数量
    'k_neighbors': 15,            # 取临近点数量
    'start': (1, 1),                    # 起始点坐标
    'goal': (9, 9),                    # 目标点坐标
    'obstacle_margin': 0.2,    # 散点生成余量
    'line_margin': 1.0,           # 碰撞检测余量
}

#################################################
#    -----------------------  散点生成聚类  -----------------------     #
#################################################

# 判断点位合法性
def point_not_in_obstacle(point, obstacles, margin=config['obstacle_margin']):
    for obs in obstacles:
        if obs['type'] == 'rect':
            x, y, w, h = obs['x'], obs['y'], obs['width'], obs['height']
            if (x - margin <= point[0] <= x + w + margin) and (y - margin <= point[1] <= y + h + margin):
                return False
    return True

# 生成随机地图散点（生成策略可优化）
def sample_free_space(num_samples, x_range, y_range, obstacles):
    samples = set()
    while len(samples) < num_samples:
        sample = (np.random.uniform(*x_range), np.random.uniform(*y_range))
        if point_not_in_obstacle(sample, obstacles):
            samples.add(sample)
    return list(samples)

######################################################
#    ---------------------    拓扑合法性判断聚类    ----------------------      #
######################################################

# 叉积计算
def ccw(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

# 判断线段AB和CD是否相交
def segments_intersect(A, B, C, D):
    return (ccw(A, B, C) * ccw(A, B, D) < 0) and (ccw(C, D, A) * ccw(C, D, B) < 0)

# 计算点P到线段CD的距离
def point_segment_distance(P, C, D):
    cd_x = D[0] - C[0]
    cd_y = D[1] - C[1]
    cp_x = P[0] - C[0]
    cp_y = P[1] - C[1]
    dp_x = P[0] - D[0]
    dp_y = P[1] - D[1]
    t = (cp_x * cd_x + cp_y * cd_y) / (cd_x ** 2 + cd_y ** 2)
    if t <= 0:         # 点P对线段CD的投影在C点之外
        return math.hypot(cp_x, cp_y)
    if t >= 1:         # 点P对线段CD的投影在D点之外
        return math.hypot(dp_x, dp_y)
    if 0 < t < 1:     # 点P对线段CD的投影在CD之间
        return math.hypot(cp_x - t * cd_x, cp_y - t * cd_y)

# 克莱姆法则求解  t  s
def compute_t_s(A, B, C, D):
    ab_x = B[0] - A[0]
    ab_y = B[1] - A[1]
    cd_x = D[0] - C[0]
    cd_y = D[1] - C[1]
    ac_x = C[0] - A[0]
    ac_y = C[1] - A[1]

    a1 = ab_x**2 + ab_y**2
    b1 = -ab_x * cd_x - ab_y * cd_y
    c1 = ac_x  * ab_x + ac_y * ab_y
    a2 = ab_x * cd_x + ab_y * cd_y
    b2 = -cd_x**2 - cd_y**2
    c2 = -ac_x  * cd_x - ac_y * cd_y

    t = (c1 * b2 - c2 * b1) / a1 * b2 - a2 * b1
    s = (a1 * c2 - a2 * c1) / a1 * b2 - a2 * b1
    return t, s

# 计算两线段最短距离
def segment_distance(A, B, C, D):
    # 四个端点到另一线段的距离
    d1 = point_segment_distance(A, C, D)
    d2 = point_segment_distance(B, C, D)
    d3 = point_segment_distance(C, A, B)
    d4 = point_segment_distance(D, A, B)
    min_d = min(d1, d2, d3, d4)
    # 线段AB与线段CD等价平行，可直接取最小距离返回
    if abs(B[0] - A[0] * D[1] - C[1] - B[1] - A[1] * D[0] - C[0]) < 1e-10: return min_d
    else: # 线段AB与线段CD不平行，推算比例参数后计算
        t, s = compute_t_s(A, B, C, D)
        if 0 <= t <= 1 and 0 <= s <= 1:
            point_ab = (A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]))
            point_cd = (C[0] + s * (D[0] - C[0]), C[1] + s * (D[1] - C[1]))
            dx = point_ab[0] - point_cd[0]
            dy = point_ab[1] - point_cd[1]
            return math.hypot(dx, dy)
        else: return min_d

# 判断线段合法性
def line_not_intersects_rect(p1, p2, rect, margin=config['line_margin']):
    # 干扰物四角坐标
    x_min, y_min, x_max, y_max = rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']
    # 宽阶段相交检测
    if max(p1[0], p2[0]) < x_min or min(p1[0], p2[0]) > x_max or max(p1[1], p2[1]) < y_min or min(p1[1], p2[1]) > y_max: return True
    # 窄阶段相交检测
    edges = [[(x_min, y_min), (x_min, y_max)], [(x_min, y_max), (x_max, y_max)], [(x_max, y_max), (x_max, y_min)], [(x_max, y_min), (x_min, y_min)]]
    for edge in edges:
        # 线段与障碍相交
        if segments_intersect(p1, p2, *edge): return False
        # 线段与障碍余量距离不足
        if segment_distance(p1, p2, *edge) < margin: return False
    return True

#################################################
#    -----------------------  算法流程聚类  -----------------------     #
#################################################

# 构建连通图
def build_roadmap(samples, obstacles, k_neighbors):
    tree = KDTree(samples) # KDTree用于快速查找neighbors
    graph = {i: set() for i in range(len(samples))} # 邻节点字典
    for idx, point in enumerate(samples):
        # 提取邻节点距离与序号
        distances, neighbor_indexs = tree.query([point], k=k_neighbors + 1)
        distances, neighbor_indexs = distances[0][1:], neighbor_indexs[0][1:]
        # 合法线段存入邻节点字典
        for neighbor_idx in neighbor_indexs:
            neighbor = samples[neighbor_idx]
            if False not in [line_not_intersects_rect(point, neighbor, obstacle) for obstacle in obstacles]:
                distance = distances[np.where(neighbor_indexs == neighbor_idx)][0]
                graph[idx].add((neighbor_idx, distance))
                graph[neighbor_idx].add((idx, distance))
    return graph

# A Star路径规划
def astar(graph, nodes, start_idx, goal_idx):
    open_heap = list()          # 优先队列
    came_from = dict()        # 父节点字典
    closed = set()                # 已探索集合

    # 始点入堆
    heapq.heappush(open_heap, (0, start_idx))
    # 实际代价矩阵g
    g_score = {i: float('inf') for i in graph}
    g_score[start_idx] = 0
    # 综合代价矩阵 f = 启发式代价 h + 实际代价 g
    f_score = {i: float('inf') for i in graph}
    f_score[start_idx] = np.linalg.norm(np.array(nodes[start_idx]) - np.array(nodes[goal_idx])) + 0

    while open_heap:
        _ , current = heapq.heappop(open_heap)
        if current in closed: continue
        # 抵达终点，路径回溯
        if current == goal_idx:
            path = list()
            while current in came_from:
                path.append(nodes[current])
                current = came_from[current]
            path.append(nodes[start_idx])
            return path[::-1]
        # 继续搜索
        closed.add(current)
        for neighbor, distance in graph[current]:
            tentative_g = g_score[current] + distance
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + np.linalg.norm(np.array(nodes[neighbor]) - np.array(nodes[goal_idx]))
                f_score[neighbor] = f
                heapq.heappush(open_heap, (f, neighbor))
    return None

if __name__ == "__main__":
    # 随机散点生成
    samples = sample_free_space(config['num_samples'], config['x_range'], config['y_range'], config['obstacles'])
    # 添加起点和终点
    samples.append(config['start'])
    samples.append(config['goal'])
    # 构建连通图
    roadmap = build_roadmap(samples, config['obstacles'], config['k_neighbors'])
    # 路径搜索
    path = astar(roadmap, samples, len(samples) - 2, len(samples) - 1)
    # 可视化
    plt.figure(figsize=(10, 10))
    # 障碍物绘制
    for obs in config['obstacles']:
        rect = plt.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], color='gray', alpha=0.5)
        plt.gca().add_patch(rect)
    # 散点绘制
    xs, ys = zip(*samples)
    plt.scatter(xs[:-2], ys[:-2], s=5, c='blue')
    plt.scatter(*config['start'], s=100, c='green', marker='o')
    plt.scatter(*config['goal'], s=100, c='red', marker='o')
    # 路径绘制
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, c='red', linewidth=2)
    # 其他标识
    plt.xlim(config['x_range'])
    plt.ylim(config['y_range'])
    plt.grid(True)
    plt.show()

