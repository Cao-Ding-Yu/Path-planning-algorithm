import random
import numpy as np
import matplotlib.pyplot as plt

# 算法参数
config = {
    'start': (1,1),                     # 起始点坐标
    'goal': (30,30),                 # 目标点坐标
    'obstacles _num': 50,      # 障碍物数量
    'k_att': 5.0,                      # 引力增益系数
    'k_att_alpha': 0.05,          # 引力调节系数
    'k_rep': 500.0,                 # 斥力增益系数
    'rep_radius': 3.0,             # 斥力影响半径
    'step_size':  0.1,              # 步长距离
    'max_iters':  1000,          # 最大迭代次数
    'osc_threshold': 6,         # 震荡检测阈值
}

# 创建随机障碍
def create_obstacles(obstacles_num, start, goal):
    obstacles = list()
    while len(obstacles) < obstacles_num:
        obstacle = (random.uniform(start[0], goal[0]), random.uniform(start[1], goal[1]))
        if obstacle not in obstacles and obstacle != start and obstacle != goal:
            obstacles.append(obstacle)
    return obstacles

# 人工势场路径规划算法
class APF:
    def __init__(self, start, goal, obstacles, k_att, k_att_alpha, k_rep, rep_radius, step_size, max_iters, osc_threshold):
        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.obstacles = [np.array(obs, dtype=np.float64) for obs in obstacles]
        self.k_att = k_att
        self.k_att_alpha = k_att_alpha
        self.k_rep = k_rep
        self.rep_radius = rep_radius
        self.step_size = step_size
        self.max_iters = max_iters
        self.osc_threshold = osc_threshold

    # 引力场  F = k_att * ( d + α * ( d + |d| ) )
    def attractive_force(self, position):
        vec_to_goal = self.goal - position
        distance = np.linalg.norm(vec_to_goal)
        return self.k_att * (vec_to_goal + self.k_att_alpha * vec_to_goal * distance)

    # 斥力场  F = k_rep * ( 1/d - 1/rep_radius ) * ( 1 / d*d )
    def repulsive_force(self, position):
        rep_force = np.zeros(2)
        for obs in self.obstacles:
            vec_to_obs = position - obs
            obs_dist = np.linalg.norm(vec_to_obs)
            if obs_dist < self.rep_radius:
                if np.linalg.norm(position - self.goal) > self.rep_radius:
                    rep_term = self.k_rep * (1 / obs_dist - 1 / self.rep_radius) * (1 / obs_dist ** 2)
                    rep_force += rep_term * (vec_to_obs / obs_dist)
                else: pass # 引入AStar算法优化，防止障碍距离目标点过近斥力过大，出现无法抵达目标的现象
        return rep_force

    # 路径规划
    def plan(self):
        path = [self.start.copy()]                  # 路径集合
        current_pos = self.start.copy()         # 当前位置
        last_force = None                           # 上次受力
        oscillation_count = 0                      # 震荡次数
        for iter in range(self.max_iters):
            # 计算合力
            f_att = self.attractive_force(current_pos)
            f_rep = self.repulsive_force(current_pos)
            total_force = f_att + f_rep
            # 振荡检测
            if last_force is not None:
                dir_change = np.dot(last_force, total_force)
                if dir_change < 0: oscillation_count += 1
                else: oscillation_count = max(0, oscillation_count - 1)
            # 处理局部极小值和振荡
            if oscillation_count > self.osc_threshold:
                perturbation = random.uniform(0.5, 1.5) * np.array([np.cos(random.uniform(0, 2 * np.pi)), np.sin(random.uniform(0, 2 * np.pi))]) # 处理逻辑可结合AStar算法改进，防止方向随机撞上障碍物
                current_pos = current_pos + perturbation
                oscillation_count = 0
                path.append(current_pos.copy())
                last_force = total_force
                continue
            # 常规移动
            direction = total_force / np.linalg.norm(total_force)
            current_pos += self.step_size * direction
            path.append(current_pos.copy())
            last_force = total_force
            # 抵达终点
            if np.linalg.norm(self.goal - current_pos) <= config['step_size']:
                path.append(self.goal.copy())
                break
        return path

if __name__ == "__main__":
    # 障碍物初始化
    obstacles = create_obstacles(config['obstacles _num'], config['start'], config['goal'])
    # 算法初始化
    apf = APF(
        start=config['start'],
        goal=config['goal'],
        obstacles=obstacles,
        k_att=config['k_att'],
        k_att_alpha=config['k_att_alpha'],
        k_rep=config['k_rep'],
        rep_radius=config['rep_radius'],
        step_size=config['step_size'],
        max_iters=config['max_iters'],
        osc_threshold=config['osc_threshold'],
    )
    # 路径规划
    path = apf.plan()
    # 可视化
    plt.figure(figsize=(10, 10))
    # 绘制路径和关键点
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'b--', linewidth=2, label='Path')
    plt.scatter(*config['start'], c='green', s=150, marker='*', label='Start')
    plt.scatter(*config['goal'], c='blue', s=150, marker='*', label='Goal')
    # 绘制障碍物
    for obs in obstacles: plt.scatter(*obs, c='red', s=50, marker='s', label='Obstacle')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
