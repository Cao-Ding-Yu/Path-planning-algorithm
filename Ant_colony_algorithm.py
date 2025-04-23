import numpy as np
import matplotlib.pyplot as plt

# 最佳路径参数
best_path = None
best_distance = float('inf')

# 蚂蚁参数
num_ants = 20                  # 蚂蚁数量
num_iterations = 200        # 迭代次数
alpha = 1                          # 信息素因子
beta = 2                            # 启发式因子
rho = 0.1                           # 信息素挥发系数
Q = 1                                # 信息素强度常数

# 城市参数
cities_num = 50
position_min = 0
position_max = 50

# 创建不重复城市坐标
def cities_init(cities_num, min, max):
    cities = set()
    while len(cities) < cities_num:
        city = tuple(np.random.randint(min, max+1, size=2))
        cities.add(city)
    return np.array(list(cities))

# 计算城市欧氏距离矩阵
def distance_matrix(cities):
    dist_matrix = np.zeros((cities_num, cities_num))
    for i in range(cities_num):
        for j in range(cities_num):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

if __name__ == '__main__':
    # 城市初始化
    cities = cities_init(cities_num, position_min, position_max)
    dist_matrix = distance_matrix(cities)
    # 信息素矩阵初始化
    pheromone = np.ones((cities_num, cities_num)) * 0.1
    # 路径迭代
    for iteration in range(num_iterations):
        all_paths = list()
        # 蚂蚁行为
        for ant in range(num_ants):
            path = list()
            total_distance = 0
            # 随机选择起始城市
            current_city = np.random.randint(cities_num)
            path.append(current_city)
            # 蚂蚁遍历城市
            while len(path) < cities_num:
                probabilities = list()
                unvisited = [city for city in range(cities_num) if city not in path]
                for city in unvisited:
                    phe = pheromone[current_city][city] ** alpha
                    visibility = (1.0 / dist_matrix[current_city][city]) ** beta
                    probabilities.append(phe * visibility)
                probabilities = np.array(probabilities) / np.sum(probabilities)
                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                current_city = next_city
            # 统计或更新路径总长度
            for i in range(cities_num - 1):
                total_distance += dist_matrix[path[i]][path[i + 1]]
            total_distance += dist_matrix[path[-1]][path[0]]
            if total_distance < best_distance:
                best_distance = total_distance
                best_path = path.copy()
            # 记录路径信息
            all_paths.append((path, total_distance))

        # 信息素行为：τ_ij(t+1) = (1-ρ)*τ_ij(t)
        pheromone = (1 - rho) * pheromone
        for path, dist in all_paths:
            delta_pheromone = Q / dist
            for i in range(cities_num - 1):
                pheromone[path[i]][path[i + 1]] += delta_pheromone
                pheromone[path[i + 1]][path[i]] += delta_pheromone
            pheromone[path[-1]][path[0]] += delta_pheromone
            pheromone[path[0]][path[-1]] += delta_pheromone

        # 显示迭代过程
        print(f"Iteration {iteration + 1}: Best Distance = {best_distance:.2f}")

    # 路径可视化
    plt.figure(figsize=(8, 6))
    # 绘制城市点
    plt.scatter(cities[:, 0], cities[:, 1], color='blue', s=100)
    # 绘制路径
    for i in range(cities_num - 1):
        plt.plot(
            [cities[best_path[i]][0], cities[best_path[i + 1]][0]],
            [cities[best_path[i]][1], cities[best_path[i + 1]][1]],
            color='red', linestyle='-', linewidth=2
        )
    plt.plot(
        [cities[best_path[-1]][0], cities[best_path[0]][0]],
        [cities[best_path[-1]][1], cities[best_path[0]][1]],
        color='red', linestyle='-', linewidth=2
    )
    # 绘制说明
    plt.title(f"Optimal Path (Distance: {best_distance:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()