import random
import numpy as np

# 1.0 新增：节点类
class Node:
    def __init__(self, id, x, y, type='normal'):
        self.id = id
        self.x = x
        self.y = y
        self.type = type  # 'normal', 'hotspot', 'remote'
        
        # 容器：直接知道当前节点有哪些人和车
        self.customers = [] 
        self.cars = []

    def get_traffic_factor(self):
        """根据节点类型返回拥堵/成本系数"""
        if self.type == 'hotspot':
            return 2.0  # 闹市区，进出成本加倍
        return 1.0

# 1.1 顾客类 (修改)
class Customer:
    def __init__(self, id, start_node, end_node, creation_time):
        self.id = id
        # 核心改动：不再存 (x,y)，而是存 Node 对象或 Node ID
        self.start_node = start_node 
        self.end_node = end_node     
        self.creation_time = creation_time
        self.status = 'waiting' 

# 1.2 车辆类 (修改)
class Car:
    def __init__(self, id, current_node, capacity=1):
        self.id = id
        self.current_node = current_node # 核心改动：存 Node 对象或 Node ID
        self.capacity = capacity
        self.status = 'idle' 
        self.current_customer_id = None 

# 1.3 网络类 (修改)
class Net:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.nodes = [] # 存储所有的 Node 对象
        self._init_nodes(n, m) # 初始化节点
        self.adj_matrix = self._generate_random_weights(n, m)
    
    def _init_nodes(self, n, m):
        """初始化节点对象"""
        for i in range(n * m):
            y = i // m
            x = i % m
            # 模拟：随机设置一些节点为闹市区
            node_type = 'hotspot' if random.random() < 0.1 else 'normal'
            self.nodes.append(Node(i, x, y, node_type))

    def _generate_random_weights(self, n, m):
        num_nodes = n * m
        matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(matrix, 0)
        
        for y in range(n):
            for x in range(m):
                idx = y * m + x
                current_node = self.nodes[idx]
                
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for nx, ny in neighbors:
                    if 0 <= nx < m and 0 <= ny < n:
                        n_idx = ny * m + nx
                        neighbor_node = self.nodes[n_idx]
                        
                        # 基础权重
                        base_weight = random.randint(1, 10)
                        
                        # 核心优化：利用 Node 属性调整权重
                        # 比如：进入闹市区的成本更高
                        weight = base_weight * neighbor_node.get_traffic_factor()
                        
                        matrix[idx][n_idx] = weight
                        matrix[n_idx][idx] = weight 
        return matrix
    
    def get_node(self, idx):
        return self.nodes[idx]
        
    # coord_to_index 等辅助函数可以保留，用于通过坐标查找 Node
    def coord_to_index(self, coord):
        x, y = coord
        return y * self.m + x
    
    def index_to_coord(self, index):
        return (self.nodes[index].x, self.nodes[index].y)