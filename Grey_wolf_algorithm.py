import random
import numpy as np

"""
    obj_func: 目标函数
    lb: 变量下界
    ub: 变量上界
    dim: 变量维度
    search_agents: 狼群数量
    max_iter: 最大迭代次数
 """

# 适应度函数
def func(x):
    return sum(abs(x))

# 灰狼算法
def grey_wolf_optimizer(func, lb, ub, dim, search_agents, max_iter):
    # α狼 β狼 δ狼 ω狼群 初始化
    alpha_pos, beta_pos, delta_pos = [np.zeros(dim) for _ in range(3)]
    alpha_score, beta_score, delta_score = [float("inf") for _ in range(3)]
    positions = np.random.uniform(lb, ub,(search_agents, dim))
    # 迭代优化
    for iter in range(max_iter):
        # 更新α狼 β狼 δ狼
        for i in range(search_agents):
            fitness = func(positions[i, :])
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()
        # 线性递减系数（步长）
        a = 2 - iter * (2 / max_iter)
        # ω狼群围攻猎物
        for i in range(search_agents):
            for j in range(dim):
                a1, a2, a3 = [2 * a * random.random() - a for _ in range(3)]
                c1, c2, c3 = [2 * random.random() for _ in range(3)]
                # α向量
                alpha = abs(c1 * alpha_pos[j] - positions[i, j])
                x1 = alpha_pos[j] - a1 * alpha
                # β向量
                beta = abs(c2 * beta_pos[j] - positions[i, j])
                x2 = beta_pos[j] - a2 * beta
                # δ向量
                delta = abs(c3 * delta_pos[j] - positions[i, j])
                x3 = delta_pos[j] - a3 * delta
                # 合成向量
                positions[i, j] = (x1 + x2 + x3) / 3
                positions[i, j] = np.clip(positions[i, j], lb, ub)
        print(f"Iteration {iter}: Best Score = {alpha_score}")
    return alpha_pos, alpha_score

if __name__ == "__main__":
    best_pos, best_score = grey_wolf_optimizer(func, lb=-5.0, ub=5.0, dim=3, search_agents=20, max_iter=100)
    print("\nOptimization Results:" + f"\nBest Position: {best_pos}" + f"\nBest Score: {best_score}")