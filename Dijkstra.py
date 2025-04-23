import heapq

def dijkstra(graph, start):
    # 创建字典，默认与其他节点无穷远
    distances = {node: [float('inf'), [start]] for node in graph}
    # 起始点初始化
    distances[start][0] = 0
    heap = [(0, start)]
    # 距离迭代
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        # 当前路径非更短，则跳过该节点
        if current_distance > distances[current_node][0]: continue
        # 遍历当前节点的所有邻居
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor][0]:
                distances[neighbor][0] = distance
                distances[neighbor][1] = distances[current_node][1] + [neighbor]
                heapq.heappush(heap, (distance, neighbor))
    return distances

if __name__ == "__main__":
    graph = {
        'A': [('B', 100), ('C', 30), ('D', 10), ('E', 5)],
        'B': [('A', 100), ('C', 50)],
        'C': [('A', 30), ('B', 50)],
        'D': [('A', 10), ('F', 50)],
        'E': [('A', 5), ('F', 60), ('G', 10), ('H', 20)],
        'F': [('D', 50), ('E', 60), ('G', 50), ('H', 50)],
        'G': [('E', 10), ('F', 50)],
        'H': [('E', 20), ('F', 50)],
    }
    result = dijkstra(graph, 'A')
    for node, dist in result.items():
        print(f"{node}: {dist}")
