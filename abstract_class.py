import random
import numpy as np
import math 

class Node:
    def __init__(self, id, x, y, type='normal'):
        self.id = id
        self.x = x
        self.y = y
        self.type = type 
        
class Customer:
    def __init__(self, id, start_node, end_node, creation_time):
        self.id = id
        self.start_node = start_node 
        self.end_node = end_node     
        self.creation_time = creation_time
        self.status = 'waiting' 
        # 新增：记录该顾客被“遗留”了多少个时间窗
        self.missed_windows = 0 

class Car:
    def __init__(self, id, current_node, capacity=1):
        self.id = id
        self.current_node = current_node 
        self.capacity = capacity
        self.status = 'idle' 
        self.current_customer_id = None 

class Net:
    # Net 类代码与你上传的保持完全一致，无需修改
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.nodes = [] 
        self.hotspot_centers, self.hotspot_radius = self._init_nodes(n, m) 
        self.adj_matrix = self._generate_random_weights(n, m) 

    def _calculate_hotspot_max_factor(self, x, y):
        HOTSPOT_MAX_FACTOR = 2.5 
        HOTSPOT_MIN_FACTOR = 1.0 
        factor_range = HOTSPOT_MAX_FACTOR - HOTSPOT_MIN_FACTOR 
        max_factor = 1.0 
        if not self.hotspot_centers:
            return max_factor
        for cx, cy in self.hotspot_centers:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < self.hotspot_radius:
                d_norm = dist / self.hotspot_radius
                current_factor = HOTSPOT_MAX_FACTOR - d_norm * factor_range
                max_factor = max(max_factor, current_factor)
        return max_factor

    def _init_nodes(self, n, m):
        num_nodes = n * m
        self.nodes = []
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
            for cx, cy in hotspot_centers:
                 dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                 if dist < hotspot_radius:
                    decay_factor = (1.0 - (dist / hotspot_radius))**2 
                    hotspot_prob = base_hotspot_prob * decay_factor
                    if random.random() < hotspot_prob:
                        node_type = 'hotspot'
                        break 
            if node_type == 'normal':
                r_traffic = random.random()
                if r_traffic < 0.04: 
                    node_type = 'traffic_light'
            self.nodes.append(Node(i, x, y, node_type))
        return hotspot_centers, hotspot_radius

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
                            traffic_factor = self._calculate_hotspot_max_factor(neighbor_node.x, neighbor_node.y)
                        weight = base_weight * traffic_factor
                        matrix[idx][n_idx] = weight
                        matrix[n_idx][idx] = weight 
        return matrix
    
    def get_node(self, idx): return self.nodes[idx]
    def coord_to_index(self, coord): return coord[1] * self.m + coord[0]
    def index_to_coord(self, index): return (self.nodes[index].x, self.nodes[index].y)