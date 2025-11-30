import random
import numpy as np
import math 

# Node, Customer, Car 类保持不变

class Node:
    def __init__(self, id, x, y, type='normal'):
        self.id = id
        self.x = x
        self.y = y
        self.type = type 
        
        self.customers = [] 
        self.cars = []

class Customer:
    def __init__(self, id, start_node, end_node, creation_time):
        self.id = id
        self.start_node = start_node 
        self.end_node = end_node     
        self.creation_time = creation_time
        self.status = 'waiting' 

class Car:
    def __init__(self, id, current_node, capacity=1):
        self.id = id
        self.current_node = current_node 
        self.capacity = capacity
        self.status = 'idle' 
        self.current_customer_id = None 

# 网络类 (实现多闹市区最大值权重叠加)
class Net:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.nodes = [] 
        
        # R=7.0
        self.hotspot_centers, self.hotspot_radius = self._init_nodes(n, m) 
        self.adj_matrix = self._generate_random_weights(n, m) 

    # --- 辅助方法：计算节点到所有中心点的距离及最大权重 ---
    def _calculate_hotspot_max_factor(self, x, y):
        """
        计算节点(x, y)在所有闹市区中的最大权重因子。
        实现：权重取 max(2.5 - d_norm * 1.5)
        """
        
        HOTSPOT_MAX_FACTOR = 2.5 
        HOTSPOT_MIN_FACTOR = 1.0 
        factor_range = HOTSPOT_MAX_FACTOR - HOTSPOT_MIN_FACTOR # 1.5

        max_factor = 1.0 # 默认权重，如果不在任何闹市区范围内
        
        if not self.hotspot_centers:
            return max_factor

        for cx, cy in self.hotspot_centers:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist < self.hotspot_radius:
                d_norm = dist / self.hotspot_radius
                # 线性衰减公式
                current_factor = HOTSPOT_MAX_FACTOR - d_norm * factor_range
                
                # 核心要求：取所有闹市区计算出的权重的最大值
                max_factor = max(max_factor, current_factor)
        
        return max_factor


    # --- 核心修改 1: _init_nodes (闹市区数量 3-5 个，R=7.0) ---
    def _init_nodes(self, n, m):
        num_nodes = n * m
        self.nodes = []
        
        # **** 核心修改：闹市区中心数量 3-5 个 ****
        num_hotspot_centers = random.randint(3, 5) 
        hotspot_radius = 7.0 
        base_hotspot_prob = 0.5 
        
        hotspot_centers = []
        for _ in range(num_hotspot_centers):
            center_x = random.randint(0, m - 1)
            center_y = random.randint(0, n - 1)
            hotspot_centers.append((center_x, center_y))

        for i in range(num_nodes):
            y = i // m
            x = i % m
            
            node_type = 'normal'
            
            # 检查节点是否位于任何闹市区范围内，并设置节点类型
            for cx, cy in hotspot_centers:
                 dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                 if dist < hotspot_radius:
                    # 闹市区节点类型仍基于概率（二次衰减），但权重将基于最大值
                    decay_factor = (1.0 - (dist / hotspot_radius))**2 
                    hotspot_prob = base_hotspot_prob * decay_factor
                    
                    if random.random() < hotspot_prob:
                        node_type = 'hotspot'
                        break # 一旦标记为 hotspot 就跳出，避免重复计算概率

            # 红绿灯概率 0.04
            if node_type == 'normal':
                r_traffic = random.random()
                if r_traffic < 0.04: 
                    node_type = 'traffic_light'
            
            self.nodes.append(Node(i, x, y, node_type))
            
        return hotspot_centers, hotspot_radius

    # --- 核心修改 2: _generate_random_weights (权重由 _calculate_hotspot_max_factor 决定) ---
    def _generate_random_weights(self, n, m):
        num_nodes = n * m
        matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(matrix, 0)
        
        TRAFFIC_LIGHT_FACTOR = 1.8
        
        for y in range(n):
            for x in range(m):
                idx = y * m + x
                
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] 

                for nx, ny in neighbors:
                    if 0 <= nx < m and 0 <= ny < n:
                        n_idx = ny * m + nx
                        neighbor_node = self.nodes[n_idx]
                        
                        base_weight = random.randint(1, 10)
                        traffic_factor = 1.0 
                        
                        if neighbor_node.type == 'traffic_light':
                            traffic_factor = TRAFFIC_LIGHT_FACTOR
                            
                        elif neighbor_node.type == 'hotspot':
                            # **** 核心修改：调用最大权重计算函数 ****
                            traffic_factor = self._calculate_hotspot_max_factor(neighbor_node.x, neighbor_node.y)
                            
                        weight = base_weight * traffic_factor
                        
                        matrix[idx][n_idx] = weight
                        matrix[n_idx][idx] = weight 
                        
        return matrix
    
    def get_node(self, idx):
        return self.nodes[idx]
        
    def coord_to_index(self, coord):
        x, y = coord
        return y * self.m + x
    
    def index_to_coord(self, index):
        return (self.nodes[index].x, self.nodes[index].y)