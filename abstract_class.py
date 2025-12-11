import random
import numpy as np
import math 

class Node:
    def __init__(self, id, x, y, type='normal'):
        self.id = id
        self.x = x
        self.y = y
        self.type = type # Type of node ('normal', 'traffic_light', 'hotspot', 'hotspot_affiliated')
        
class Customer:
    def __init__(self, id, start_node, end_node, creation_time):
        self.id = id
        self.start_node = start_node 
        self.end_node = end_node     
        self.creation_time = creation_time
        self.status = 'waiting' 
        self.missed_windows = 0 

class Car:
    def __init__(self, id, current_node, capacity=1):
        self.id = id
        self.current_node = current_node 
        self.capacity = capacity
        self.status = 'idle' 
        self.current_customer_id = None 

class Net:
    def __init__(self, n, m, map_config=None):
        
        self.n = n
        self.m = m
        self.nodes = [] 
        
        hotspot_coords = []
        hotspot_affiliated_coords = []
        
        if map_config:
            if 'hotspot_coords' in map_config:
                hotspot_coords = map_config['hotspot_coords']
            if 'hotspot_affiliated_coords' in map_config:
                hotspot_affiliated_coords = map_config['hotspot_affiliated_coords']
        else:
            hotspot_coords, hotspot_affiliated_coords = self._generate_random_map(n, m)

        self._generate_config_map(n, m, hotspot_coords, hotspot_affiliated_coords)

        self.weights_matrix = self._generate_weights(n, m)

    def _generate_config_map(self, n, m, hotspot_coords, hotspot_affiliated_coords):
        num_nodes = n * m
        self.nodes = []
        
        # traffic_light 概率保留
        TRAFFIC_LIGHT_PROB = 0.04 

        for i in range(num_nodes):
            y = i // m
            x = i % m
            node_type = 'normal'
            
            if (x, y) in hotspot_coords:
                node_type = 'hotspot'
            
            if (x, y) in hotspot_affiliated_coords:
                node_type = 'hotspot_affiliated'

            if node_type == 'normal':
                if random.random() < TRAFFIC_LIGHT_PROB: 
                    node_type = 'traffic_light'
                    
            self.nodes.append(Node(i, x, y, node_type))
    
    def _generate_random_map(self, n, m):
        """
        [NEW] 随机生成 Hotspot 和 Hotspot_affiliated 区域的坐标。
        """
        
        # 设定随机生成的概率和数量
        HOTSPOT_PROB = 0.05  # 节点成为 Hotspot 的概率
        AFFILIATED_DISTANCE = 2 # Hotspot_affiliated 节点与 Hotspot 的最大曼哈顿距离
        
        all_coords = [(x, y) for y in range(n) for x in range(m)]
        random.shuffle(all_coords) 
        
        hotspot_coords = set()
        
        for x, y in all_coords:
            if random.random() < HOTSPOT_PROB:
                hotspot_coords.add((x, y))

        hotspot_affiliated_coords = set()
        
        if not hotspot_coords:
            return list(hotspot_coords), list(hotspot_affiliated_coords)

        for hx, hy in hotspot_coords:
            for x, y in all_coords:
                manhattan_dist = abs(x - hx) + abs(y - hy)
               
                if 0 < manhattan_dist <= AFFILIATED_DISTANCE and (x, y) not in hotspot_coords:
                    hotspot_affiliated_coords.add((x, y))

        return list(hotspot_coords), list(hotspot_affiliated_coords)

    def _generate_weights(self, n, m):
        num_nodes = n * m
        matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(matrix, 0)
        
        TRAFFIC_LIGHT_FACTOR = 1.8
        HOTSPOT_FACTOR = 3.0  
        HOTSPOT_AFFILIATED_FACTOR = 2.0
        
        for x in range(m):
            for y in range(n):
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
                            traffic_factor = HOTSPOT_FACTOR
                            
                        elif neighbor_node.type == 'hotspot_affiliated':
                            traffic_factor = HOTSPOT_AFFILIATED_FACTOR
                        weight = base_weight * traffic_factor
                        matrix[idx][n_idx] = weight
                        matrix[n_idx][idx] = weight 
                        
        return matrix
    
    def get_node(self, idx): return self.nodes[idx]