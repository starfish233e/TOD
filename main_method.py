import heapq 
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math
import random # 保持完整性

# 导入抽象类以便于类型引用 (如果需要，但这里只涉及数据结构)

# --- 1. 定义颜色映射 (权重越大越红) ---
def create_detailed_colormap():
    """
    创建自定义的颜色映射。权重越大越红 (2.5)，权重越小越绿 (1.0)。
    """
    # 颜色列表 (从低因子颜色 (1.0) 到高因子颜色 (2.5))
    colors_for_cmap = [
        'darkgreen',    # 对应 1.0 (权重最小)
        'lime',
        'yellowgreen',
        'gold',
        'orange',
        'orangered',
        'red',
        'darkred'       # 对应 2.5 (权重最大)
    ]
    
    return LinearSegmentedColormap.from_list("red_to_green_traffic", colors_for_cmap)


# --- 2. 辅助函数：计算最大权重（用于可视化） ---
def _calculate_hotspot_max_factor_vis(net, x, y):
    """
    镜像 Net._calculate_hotspot_max_factor 逻辑，用于可视化所有节点的权重。
    """
    
    HOTSPOT_MAX_FACTOR = 2.5 
    HOTSPOT_MIN_FACTOR = 1.0 
    factor_range = HOTSPOT_MAX_FACTOR - HOTSPOT_MIN_FACTOR # 1.5

    max_factor = 1.0 
    
    if not net.hotspot_centers:
        return max_factor

    for cx, cy in net.hotspot_centers:
        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
        
        if dist < net.hotspot_radius:
            d_norm = dist / net.hotspot_radius
            # 线性衰减公式
            current_factor = HOTSPOT_MAX_FACTOR - d_norm * factor_range
            
            # 核心要求：取最大值
            max_factor = max(max_factor, current_factor)
    
    return max_factor


# --- 3. 可视化函数 ---
def visualize_net_weights(net, title="Map Node Traffic Factors (Grid: 20x20) - Max Factor Overlay"):
    """
    可视化地图网格节点的交通因子 (1.0 到 2.5)。
    """
    
    TRAFFIC_LIGHT_FACTOR = 1.8
    HOTSPOT_MAX_FACTOR = 2.5 
    
    num_nodes = net.n * net.m
    x_coords = [n.x for n in net.nodes]
    y_coords = [n.y for n in net.nodes]
    
    node_factors = np.zeros(num_nodes)
    
    for i, node in enumerate(net.nodes):
        factor = 1.0
        
        if node.type == 'traffic_light':
            factor = TRAFFIC_LIGHT_FACTOR
        
        # 无论节点类型是 'hotspot' 还是 'normal'，只要它位于闹市区影响范围内，
        # 都应计算其基于距离的最大权重。
        # traffic_light 权重固定，所以排除。
        if node.type != 'traffic_light':
             # **** 核心修改：所有节点权重基于最大权重计算 ****
             factor = _calculate_hotspot_max_factor_vis(net, node.x, node.y)
            
        node_factors[i] = factor

    # 2. 定义颜色映射 (Colormap) 和归一化
    cmap = create_detailed_colormap()
    min_factor = 1.0 
    max_factor = HOTSPOT_MAX_FACTOR 
    norm = Normalize(min_factor, max_factor)
    
    # 3. 绘图
    plt.figure(figsize=(10, 10)) 
    
    scatter = plt.scatter(x_coords, y_coords, 
                          c=node_factors, 
                          cmap=cmap, 
                          norm=norm, 
                          s=150, 
                          edgecolors='black', 
                          linewidths=0.5)

    # 4. 标注特殊节点类型
    traffic_light_indices = [i for i, n in enumerate(net.nodes) if n.type == 'traffic_light']
    tl_x = [x_coords[i] for i in traffic_light_indices]
    tl_y = [y_coords[i] for i in traffic_light_indices]
    
    plt.scatter(tl_x, tl_y, s=200, marker='s', color='magenta', alpha=0.8, label='Traffic Light')


    # 5. 添加颜色条 (Colorbar)
    cbar = plt.colorbar(scatter, fraction=0.04, pad=0.04)
    cbar.set_label('Node Traffic Factor (2.5 = Dark Red, 1.0 = Dark Green)', rotation=270, labelpad=20)

    # 6. 设置图表属性
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    plt.xlim(-0.5, net.m - 0.5)
    plt.ylim(-0.5, net.n - 0.5)
    plt.xticks(range(net.m)) 
    plt.yticks(range(net.n)) 
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.gca().set_aspect('equal', adjustable='box') 

    # 7. 显示图形
    plt.show()

# 2.1 A* 最短路径算法 (适配 Node 类)
def shortest_path(start_node_id, end_node_id, net):
    """
    使用 A* 算法计算两个节点之间的最短路径。
    输入: 起点和终点的 Node ID (int)
    输出: (距离, 路径列表[(x,y), ...])
    """
    # 检查输入是否合法
    if start_node_id == end_node_id:
        start_node = net.nodes[start_node_id]
        return 0, [(start_node.x, start_node.y)]

    # h(n): 启发式函数，使用节点坐标计算曼哈顿距离
    def heuristic(idx):
        # 直接从 net.nodes 列表中获取节点对象及其坐标
        current_node = net.nodes[idx]
        target_node = net.nodes[end_node_id]
        return abs(current_node.x - target_node.x) + abs(current_node.y - target_node.y)

    # 初始化分数
    num_nodes = net.n * net.m
    g_score = {i: float('inf') for i in range(num_nodes)}
    g_score[start_node_id] = 0
    
    f_score = {i: float('inf') for i in range(num_nodes)}
    f_score[start_node_id] = heuristic(start_node_id)

    # 优先队列: (f_score, node_id)
    open_set = [(f_score[start_node_id], start_node_id)]
    
    # 路径回溯记录
    came_from = {}
    
    while open_set:
        current_f, current_id = heapq.heappop(open_set)

        if current_id == end_node_id:
            # 重建路径：为了保持兼容性，我们依然返回 (x,y) 坐标列表
            # 如果后续需要分析节点类型，也可以改为返回 Node 对象列表
            path = []
            temp = current_id
            while temp in came_from:
                node_obj = net.nodes[temp]
                path.append((node_obj.x, node_obj.y))
                temp = came_from[temp]
            
            # 加上起点
            start_node_obj = net.nodes[start_node_id]
            path.append((start_node_obj.x, start_node_obj.y))
            
            return g_score[end_node_id], path[::-1] # 返回距离和路径(倒序)
            
        # 遍历邻居
        # 在邻接矩阵中，一行代表一个节点的所有出边
        # 注意：这里假设 net.adj_matrix 是稀疏的或者是全连接但由 inf 填充
        # 为了性能，如果矩阵很大，建议改为邻接表。但在网格图中，可以直接遍历可能的4个邻居
        # 这里为了通用性，依然遍历 adj_matrix 的一行 (或者利用网格特性优化)
        
        # 优化：网格结构下，没必要遍历所有 N*M 个点，只看 adj_matrix 中非无穷大的
        # 我们可以利用 net 里的逻辑或者直接根据 ID 算邻居，
        # 但既然已经有了 adj_matrix，我们依靠矩阵数据：
        
        # 获取当前节点的所有邻居 (权重不为 inf)
        # 这种写法在节点数很多时可能较慢，建议在 Net 类中加一个 get_neighbors(id)
        # 这里暂时维持原逻辑结构，但做一点剪枝
        for neighbor_id, cost in enumerate(net.adj_matrix[current_id]):
            if cost == float('inf') or cost == 0:
                continue

            tentative_g_score = g_score[current_id] + cost
            
            if tentative_g_score < g_score[neighbor_id]:
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id)
                heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

    return float('inf'), []

# 2.2 订单调度匹配方法 (适配 Node 类)
def match_orders(customers, cars, net):
    """
    使用匈牙利算法解决最优指派问题。
    """
    if len(customers) != len(cars):
        print("警告: 顾客和车数量不相等，无法使用标准匈牙利算法。")
        return None
        
    num = len(customers)
    cost_matrix = np.zeros((num, num))
    distance_map = {} 
    
    for i, cust in enumerate(customers):
        for j, car in enumerate(cars):
            # 修改点：直接获取 Node 对象的 ID
            # 假设 Customer.start_node 和 Car.current_node 已经是 Node 对象
            # 如果你存的是 ID，则直接使用；如果是对象，则用 .id
            
            car_node_id = car.current_node.id if hasattr(car.current_node, 'id') else car.current_node
            cust_node_id = cust.start_node.id if hasattr(cust.start_node, 'id') else cust.start_node
            
            # 计算最短路径
            dist, path = shortest_path(car_node_id, cust_node_id, net)
            
            cost_matrix[i, j] = dist
            distance_map[(cust.id, car.id)] = (dist, path)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment = {}
    total_cost = 0

    for i in range(num):
        cust = customers[row_ind[i]]
        car = cars[col_ind[i]]
        
        assignment[cust.id] = car.id
        
        car.status = 'en_route_to_customer'
        car.current_customer_id = cust.id
        cust.status = 'matched'
        
        dist = cost_matrix[row_ind[i], col_ind[i]]
        total_cost += dist
        
    return assignment, total_cost, distance_map