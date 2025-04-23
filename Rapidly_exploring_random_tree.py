import math
import random
import matplotlib.pyplot as plt

# 配置参数
config = {
    'start': (0, 0),                    # 起点坐标
    'goal': (10, 10),                # 终点坐标
    'max_iter': 10000,            # 最大迭代数
    'area': [-2, 12, -2, 12],      # 区域范围（x_min, x_max, y_min, y_max）
    'step_size': 1.0,                # 拓展步长
    'goal_sample_rate': 0.2,  # 采样偏置率
    'margin': 1,                    # 圆形障碍余量
    'num_obstacles': 5,        # 障碍物数量
}

# 创建随机障碍
def create_obstacles(num_obstacles):
    obstacles = list()
    while len(obstacles) < num_obstacles:
        temp_obstacle = (random.uniform(-2, 12), random.uniform(-2, 12), random.uniform(0.5,2))
        if (temp_obstacle not in obstacles  # 障碍不重复且规避起始点和终止点
            and math.hypot(temp_obstacle[0] - config['start'][0], temp_obstacle[1] - config['start'][1]) -1 > temp_obstacle[2]
            and math.hypot(temp_obstacle[0] - config['goal'][0], temp_obstacle[1] - config['goal'][1]) -1 > temp_obstacle[2]):
            obstacles.append(temp_obstacle)
    return obstacles

# 树节点
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

# RRT路径规划算法
class RRT:
    def __init__(self, start, goal, obstacles, area, max_iter, step_size, goal_sample_rate):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles
        self.area = area
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.nodes = [self.start]

    # 路径规划
    def plan(self):
        for _ in range(self.max_iter):
            # 偏置采样（策略可优化）
            if random.random() < self.goal_sample_rate: rand_node = self.goal
            else: rand_node = Node(random.uniform(self.area[0], self.area[1]), random.uniform(self.area[2], self.area[3]))
            # 寻找生成点最近节点
            nearest_node = min(self.nodes, key=lambda n: math.hypot(n.x - rand_node.x, n.y - rand_node.y))
            # 向随机点方向扩展
            theta = math.atan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)  # 拓展方向
            new_node = Node(nearest_node.x + self.step_size * math.cos(theta), nearest_node.y + self.step_size * math.sin(theta))
            new_node.parent = nearest_node
            # 新路径碰撞检测
            if not self.check_collision(nearest_node, new_node):
                self.nodes.append(new_node)
                # 检查是否到达目标区域
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.step_size:
                    final_node = Node(self.goal.x, self.goal.y, new_node)
                    if not self.check_collision(new_node, final_node, margin=0):
                        path = list()
                        while final_node:
                            path.append((final_node.x, final_node.y))
                            final_node = final_node.parent
                        return path[::-1]
        return [(0,0)]

    # 线段碰撞检测
    def check_collision(self, start, end, margin=config['margin']):
        for (x, y, r) in self.obstacles:
            # 计算线段到圆心的最短距离
            dx = end.x - start.x
            dy = end.y - start.y
            a = dx ** 2 + dy ** 2
            b = 2 * (dx * (start.x - x) + dy * (start.y - y))
            c = (start.x - x) ** 2 + (start.y - y) ** 2 - r ** 2
            if b ** 2 - 4 * a * c >= -margin: return True
        return False

if __name__ == "__main__":
    # 环境初始化
    obstacles = create_obstacles(config['num_obstacles'])
    # 算法初始化
    rrt = RRT(
        start=config['start'],
        goal=config['goal'],
        area=config['area'],
        max_iter=config['max_iter'],
        step_size=config['step_size'],
        goal_sample_rate=config['goal_sample_rate'],
        obstacles=obstacles,
    )
    # 执行路径规划
    path = rrt.plan()
    print(path)
    # ---------------  可视化  --------------- #
    plt.figure(figsize=(8, 8))
    # 绘制障碍物
    for (x, y, r) in obstacles:
        circle = plt.Circle((x, y), r, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    # 绘制树结构
    for node in rrt.nodes:
        if node.parent: plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'g-', lw=1.0)
    # 绘制路径
    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r-', lw=2)
    # 标记起终点
    plt.plot(rrt.start.x, rrt.start.y, 'bs', markersize=5)
    plt.plot(rrt.goal.x, rrt.goal.y, 'bs', markersize=5)
    # 显示属性
    plt.axis('equal')
    plt.grid(True)
    plt.show()

