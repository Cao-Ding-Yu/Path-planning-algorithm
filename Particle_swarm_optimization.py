import numpy as np

"""
    dim                 # 变量维度（x, y, z）
    n_particles      # 粒子数量
    n_iter              # 迭代次数
    lower              # 变量下界
    upper             # 变量上界
    w                    # 惯性权重
    c1                   # 个体学习因子
    c2                   # 社会学习因子
    v_init              # 初始速度因子
    v_max            # 速度上限因子
"""

# 适应度函数 sum( |x| + |y| + |z| )
def linear_sum(x):
    return np.sum(np.abs(x), axis=1)

# 粒子群优化算法
def particle_swarm_optimization(func, dim=3, n_particles=30, n_iter=1000, lower=-5.0, upper=5.0, w=0.8, c1=0.5, c2=0.5, v_init=0.1, v_max=0.2):
    # 初始化粒子位置和初速度
    positions = np.random.uniform(low=lower, high=upper, size=(n_particles, dim))
    velocities = np.random.uniform(low=-(upper - lower), high=(upper - lower), size=(n_particles, dim)) * v_init
    # 最优位置和适应度
    pbest_pos = positions.copy()
    pbest_fitness = func(positions)
    gbest_pos = pbest_pos[np.argmin(pbest_fitness)].copy()
    gbest_fitness = pbest_fitness[np.argmin(pbest_fitness)]
    # 迭代优化
    for iteration in range(n_iter):
        # 计算速度矢量和
        velocities = w * velocities + c1 * np.random.rand(n_particles, dim) * (pbest_pos - positions) + c2 * np.random.rand(n_particles, dim) * (gbest_pos - positions)
        velocities = np.clip(velocities, -(upper - lower) * v_max, (upper - lower) * v_max)
        # 更新粒子群位置
        positions += velocities
        positions = np.clip(positions, lower, upper)
        current_fitness = func(positions)
        # 更新个体最优
        improved = current_fitness < pbest_fitness
        pbest_pos[improved] = positions[improved]
        pbest_fitness[improved] = current_fitness[improved]
        # 更新全局最优
        current_gbest_index = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_index] < gbest_fitness:
            gbest_pos = pbest_pos[current_gbest_index].copy()
            gbest_fitness = pbest_fitness[current_gbest_index]
        print(f"Iteration {iteration + 1:3d}, Best Fitness: {gbest_fitness:.6f}")
    return gbest_pos, gbest_fitness

if __name__ == '__main__':
    best_pos, best_fitness = particle_swarm_optimization(func=linear_sum)
    print("\nOptimization Results:\n" + f"Best position (x, y, z): {best_pos}\n" + f"Best fitness (x + y + z): {best_fitness}")

