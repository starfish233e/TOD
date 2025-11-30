import heapq 
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math
import random 

# 2.1 A* 最短路径算法 
def shortest_path(start_node_id, end_node_id, net):
    if start_node_id == end_node_id:
        start_node = net.nodes[start_node_id]
        return 0, [(start_node.x, start_node.y)]

    def heuristic(idx):
        current_node = net.nodes[idx]
        target_node = net.nodes[end_node_id]
        return abs(current_node.x - target_node.x) + abs(current_node.y - target_node.y)

    num_nodes = net.n * net.m
    g_score = {i: float('inf') for i in range(num_nodes)}
    g_score[start_node_id] = 0
    f_score = {i: float('inf') for i in range(num_nodes)}
    f_score[start_node_id] = heuristic(start_node_id)
    open_set = [(f_score[start_node_id], start_node_id)]
    came_from = {}
    
    while open_set:
        current_f, current_id = heapq.heappop(open_set)
        if current_id == end_node_id:
            path = []
            temp = current_id
            while temp in came_from:
                node_obj = net.nodes[temp]
                path.append((node_obj.x, node_obj.y))
                temp = came_from[temp]
            start_node_obj = net.nodes[start_node_id]
            path.append((start_node_obj.x, start_node_obj.y))
            return g_score[end_node_id], path[::-1]
            
        for neighbor_id, cost in enumerate(net.adj_matrix[current_id]):
            if cost == float('inf') or cost == 0: continue
            tentative_g_score = g_score[current_id] + cost
            if tentative_g_score < g_score[neighbor_id]:
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id)
                heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

    return float('inf'), []

# 2.2 动态订单匹配方法 (适配 顾客 >= 车辆 & 等待时间成本)
def match_orders_dynamic(customers, cars, net, T_win, alpha):
    """
    使用匈牙利算法解决最优指派问题 (Customers > Cars)，
    并引入 alpha 参数平衡距离成本和等待时间成本。
    """
    num_cust = len(customers)
    num_cars = len(cars)
    
    # 距离矩阵
    # cost_matrix 的维度是 (顾客数, 车辆数)
    cost_matrix = np.zeros((num_cust, num_cars))
    distance_map = {} # 记录真实的物理距离 (空驶)
    
    for i, cust in enumerate(customers):
        for j, car in enumerate(cars):
            car_node_id = car.current_node.id 
            cust_node_id = cust.start_node.id 
            
            # 1. 计算物理距离 (空驶距离)
            dist, path = shortest_path(car_node_id, cust_node_id, net)
            distance_map[(cust.id, car.id)] = (dist, path)
            
            # 2. 计算等待时间成本
            waiting_cost = cust.missed_windows * T_win
            
            # 3. 计算最终匹配成本 Cost
            # Cost = alpha * 距离成本 + (1 - alpha) * 等待时间成本
            
            match_cost = (alpha * dist) - ((1 - alpha) * waiting_cost)
            
            cost_matrix[i, j] = match_cost

    # 4. 匈牙利算法 (scipy 自动处理 num_cust > num_cars 的情况)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment = {}
    total_match_cost = 0
    total_physical_empty_cost = 0 # 记录实际空驶距离

    for r, c in zip(row_ind, col_ind):
        # r 是顾客索引，c 是车辆索引
        cust = customers[r]
        car = cars[c]
        
        if r < num_cust and c < num_cars: 
            assignment[cust.id] = car.id
            total_match_cost += cost_matrix[r, c]
            
            # 记录真实的物理空驶距离
            physical_dist, _ = distance_map[(cust.id, car.id)]
            total_physical_empty_cost += physical_dist

    # 返回匹配结果 (顾客ID -> 车辆ID), 总物理空驶距离, 距离映射
    return assignment, total_physical_empty_cost, distance_map