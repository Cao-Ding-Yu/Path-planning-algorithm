import random

POP_SIZE = 10                           # 种群大小
GENOME_LENGTH = 10            # 个体长度
CROSSOVER_RATE = 0.75           # 交叉率
MUTATION_RATE = 0.5             # 变异率
MAX_GENERATIONS = 100         # 迭代次数

# 适应度评估
def fitness(individual):
    x = int(individual, 2)
    return -x*x + 256

# 父代选择（轮盘赌选择法）
def select_parents(population, fitnesses):
    probs = [f / sum(fitnesses) for f in fitnesses]
    return random.choices(population, weights=probs, k=2)

# 基因交叉
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

# 基因变异
def mutate(individual):
    mutated = list()
    for bit in individual:
        if random.random() < MUTATION_RATE:
            if bit == '1': mutated.append('0')
            else: mutated.append('1')
        else: mutated.append(bit)
    return ''.join(mutated)

if __name__ == '__main__':
    # 种群初始化
    best_fitness = None
    best_individual = None
    population = [''.join(random.choices(['0', '1'], k=GENOME_LENGTH)) for _ in range(POP_SIZE)]
    # 进化开始
    for generation in range(MAX_GENERATIONS):
        # 适应度分析
        fitnesses = [fitness(i) for i in population]
        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        print(f"Generation {generation:02}: Best = {best_individual}, Fitness = {best_fitness}")
        # 繁育子代
        new_population = list()
        while len(new_population) < POP_SIZE:
            parents = select_parents(population, fitnesses)
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        # 精英保留策略
        new_fitnesses = [fitness(i) for i in new_population]
        new_population[new_fitnesses.index(min(new_fitnesses))] = best_individual
        population = new_population
    print(f"最终最优解: {best_individual} (x = {int(best_individual, 2)}, Fitness = {best_fitness})")
