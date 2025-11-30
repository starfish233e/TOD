import heapq 
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math
import random 

# 2.1 A* 最短路径算法 (保持不变)
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

# 2.2 动态订单匹配方法 (适配 顾客 > 车辆 & 等待时间成本)
def match_orders_dynamic(customers, cars, net, T_win):
    """
    输入: 
      customers: 当前所有等待的顾客列表
      cars: 当前空闲的车辆列表
      T_win: 时间窗的成本基数 (单位时间成本)
    
    逻辑:
      Cost = 接驾距离 + (顾客已滞留窗口数 * T_win)
      
      如果 顾客A 距离车 10，已等待 0 窗口，Cost = 10
      如果 顾客B 距离车 15，已等待 2 窗口 (假设T_win=5)，Cost = 15 - (2*5) = 5? 
      
      注意：我们要最小化 Cost。
      如果我们希望优先服务等待久的人，我们应该 *减少* 他们的 Cost 吗？
      不，匈牙利算法是最小化总距离。
      
      修正逻辑：
      我们希望等待久的顾客更容易被选中。
      通常做法是：Cost = Distance - (Weight * WaitingTime)
      或者：Priority Score = Distance + Penalty。
      
      这里采用简化的加法惩罚逻辑的变体（将等待视为必须被消除的紧急度）：
      我们希望“对等待久的顾客，即使车远一点也值得去接”。
      
      实际做法：
      我们不用减法（可能出现负数）。我们用“机会成本”。
      如果不接这个等待久的顾客，由于他之后会更愤怒，成本更高。
      但在标准二分图匹配中，最简单的方式是：
      Cost_ij = Distance_ij
      我们如何体现等待优先级？
      
      让我们使用你要求的逻辑：把时间成本加入成本矩阵。
      通常这意味着：Total Cost = Real Distance + Waiting Penalty.
      但在匹配时，Waiting Penalty 对于同一个 Customer 对所有 Car 都是常数。
      在 linear_sum_assignment 中，如果一行的所有元素都加上同一个常数，匹配结果（谁配谁）是不变的！
      
      **关键点**：如果只是一次匹配，且所有车对该顾客的“等待成本”都一样，那么等待时间**不会影响**该顾客是否被选中（相对于其他顾客），除非我们面临 顾客 > 车辆 的情况。
      
      当 Customers > Cars 时，scipy 会选择 Cost 和最小的 M 个配对。
      此时，如果 Customer A (wait=100) 和 Customer B (wait=0)。
      车 C 离 A 距离 20，离 B 距离 10。
      
      如果不加等待成本：选 (C, B)，Cost=10。A 落选。
      
      如果我们想让 A 被选中，我们需要让 A 的“广义成本”变得比 B 低？
      不，linear_sum_assignment 是求最小和。
      如果我们定义 Cost = Distance - (Wait_Time * Factor)。
      
      例：Factor = 10.
      A: Dist 20, Wait 5. Score = 20 - 50 = -30.
      B: Dist 10, Wait 0. Score = 10 - 0 = 10.
      算法会选最小的 (-30)，所以选 A。这是对的！
      
      **结论**：为了让等待久的顾客被优先匹配，我们需要在成本矩阵中 **减去** 与等待时间成正比的项。
    """
    
    if not customers or not cars:
        return {}, 0, {}

    num_cust = len(customers)
    num_cars = len(cars)
    
    # 距离矩阵
    cost_matrix = np.zeros((num_cust, num_cars))
    distance_map = {} # 记录真实的物理距离
    
    for i, cust in enumerate(customers):
        for j, car in enumerate(cars):
            car_node_id = car.current_node.id if hasattr(car.current_node, 'id') else car.current_node
            cust_node_id = cust.start_node.id if hasattr(cust.start_node, 'id') else cust.start_node
            
            # 1. 计算物理距离 (空驶距离)
            dist, path = shortest_path(car_node_id, cust_node_id, net)
            distance_map[(cust.id, car.id)] = (dist, path)
            
            # 2. 计算匹配权重 Cost
            # Cost = 物理距离 - (等待时间带来的“必须服务”的紧迫感)
            # 这样等待时间越长，Cost 越小 (甚至为负)，越容易被选中进入最小权匹配
            waiting_bonus = cust.missed_windows * T_win
            
            cost_matrix[i, j] = dist - waiting_bonus

    # 3. 匈牙利算法 (scipy 自动处理 num_cust > num_cars 的情况)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment = {}
    total_physical_empty_cost = 0

    for i in range(len(row_ind)):
        cust_idx = row_ind[i]
        car_idx = col_ind[i]
        
        cust = customers[cust_idx]
        car = cars[car_idx]
        
        assignment[cust.id] = car.id
        
        # 状态更新在此处不做，在主循环做，这里只负责计算
        real_dist, _ = distance_map[(cust.id, car.id)]
        total_physical_empty_cost += real_dist
        
    return assignment, total_physical_empty_cost, distance_map