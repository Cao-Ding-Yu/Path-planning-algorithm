import numpy as np

"""
    func: 目标函数
    dim: 变量维度
    lb: 变量下界
    ub: 变量上界
    n_wolves: 狼群总数
    n_scouts: 探狼数量
    max_iter: 最大迭代次数
    step_alpha: 步长缩放因子
"""

# 适应度函数
def func(x):
    return np.sum(abs(x))

# 狼群优化算法
class WolfPackAlgorithm:
    # 算法初始化
    def __init__(self, func, dim, lb, ub, n_wolves, n_scouts, max_iter, step_alpha):
        self.func = func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.n_wolves = n_wolves
        self.n_scouts = n_scouts
        self.max_iter = max_iter
        self.step_alpha = step_alpha
        # 狼群初始化
        self.positions = np.random.uniform(lb, ub, (n_wolves, dim))
        self.fitness = np.array([func(x) for x in self.positions])
        self.leader_position = self.positions[np.argmin(self.fitness)].copy()
        self.leader_fitness = self.fitness[np.argmin(self.fitness)]

    # 更新探狼
    def update_leader(self):
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.leader_fitness:
            self.leader_position = self.positions[min_idx].copy()
            self.leader_fitness = self.fitness[min_idx]

    # 探狼游走
    def scouts_search(self, current_iter):
        step = self.step_alpha * (1 - current_iter / self.max_iter)
        for i in range(self.n_scouts):
            direction = np.random.uniform(-1, 1, self.dim)
            direction /= np.linalg.norm(direction) + 1e-8
            new_pos = self.positions[i] + step * direction
            new_pos = np.clip(new_pos, self.lb, self.ub)
            if self.func(new_pos) < self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = self.func(new_pos)

    # 猛狼围攻
    def besiege(self):
        for i in range(self.n_scouts, self.n_wolves):
            step = self.step_alpha * np.random.rand() * (self.leader_position - self.positions[i])
            new_pos = self.positions[i] + step
            new_pos = np.clip(new_pos, self.lb, self.ub)
            if self.func(new_pos) < self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = self.func(new_pos)

    # 执行迭代优化
    def optimize(self):
        for iter in range(self.max_iter):
            self.scouts_search(iter)
            self.update_leader()
            self.besiege()
            self.update_leader()
            print(f"Iteration {iter + 1}/{self.max_iter}, Best Fitness: {self.leader_fitness:.6f}")
        return self.leader_position, self.leader_fitness

if __name__ == "__main__":
    wpa = WolfPackAlgorithm(func=func, dim=3, lb=-5.0, ub=5.0, n_wolves=30, n_scouts=5, max_iter=1000, step_alpha=0.1)
    best_position, best_fitness = wpa.optimize()
    print("\nOptimization Result:" + f"\nBest Position: {best_position}" + f"\nBest Fitness: {best_fitness}")
