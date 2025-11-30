import random
import numpy as np
import math
import heapq 
from scipy.optimize import linear_sum_assignment 
# 以下是可视化所需的依赖：
import matplotlib.pyplot as plt 
# 注意：Normalize 是 plt.Normalize 的简写，但为清晰，我们用 plt
from matplotlib.colors import LinearSegmentedColormap, Normalize 
from abstract_class import Net, Customer, Car, Node 
from main_method import match_orders, shortest_path

def generate_mock_data(net, num_orders=20):
    """生成测试数据。"""
    customers = []
    cars = []
    
    hotspots = [n for n in net.nodes if n.type == 'hotspot']
    traffic_lights = [n for n in net.nodes if n.type == 'traffic_light'] 
    normals = [n for n in net.nodes if n.type == 'normal']
    
    print(f"地图统计: 总节点 {len(net.nodes)}, 闹市区节点 {len(hotspots)}, 红绿灯节点 {len(traffic_lights)}, 普通节点 {len(normals)}")

    for i in range(num_orders):
        if hotspots and random.random() < 0.7:
            start_node = random.choice(hotspots)
        else:
            start_node = random.choice(net.nodes)
            
        end_node = random.choice(net.nodes)
        
        while end_node.id == start_node.id:
            end_node = random.choice(net.nodes)
            
        cust = Customer(id=i, start_node=start_node, end_node=end_node, creation_time=0)
        customers.append(cust)

        car_start_node = random.choice(net.nodes)
        car = Car(id=i, current_node=car_start_node)
        cars.append(car)
        
    return customers, cars

def run_experiment():
    """
    运行完整的调度实验流程：初始化网络、生成数据、匹配订单、统计指标。
    """

    # 1. 初始化网络 (使用 20x20 网格)
    print(">>> 正在初始化网络...")
    net = Net(20, 20) 
    
    # 核心修改 2: 调用可视化函数，并短暂暂停刷新
    

    # 2. 生成数据 (20个订单，20辆车)
    print(">>> 正在生成模拟数据...")
    customers, cars = generate_mock_data(net, num_orders=20)
    
    # 3. 运行核心调度算法
    print(">>> 正在进行订单匹配 (匈牙利算法)...")
    assignment, total_empty_cost, details = match_orders(customers, cars, net) # type: ignore
    
    if assignment is None:
        print("匹配失败，请检查车辆和订单数量。")
        return

    # 4. 计算核心指标
    print(">>> 正在计算核心实验指标...")
    
    wait_times = []      
    loaded_distances = [] 
    
    for cust_id, car_id in assignment.items():
        pickup_dist, _ = details[(cust_id, car_id)]
        wait_times.append(pickup_dist)
        
        cust = next(c for c in customers if c.id == cust_id)
        
        trip_dist, _ = shortest_path(cust.start_node.id, cust.end_node.id, net)
        loaded_distances.append(trip_dist)

    # 5. 统计输出
    avg_wait_time = np.mean(wait_times)
    max_wait_time = np.max(wait_times)
    total_loaded_dist = sum(loaded_distances)
    total_total_dist = total_empty_cost + total_loaded_dist
    
    utilization_rate = total_loaded_dist / total_total_dist if total_total_dist > 0 else 0

    print("-" * 30)
    print("📊 实验结果报告")
    print("-" * 30)
    print(f"1. 平均顾客等待时间 (空驶成本): {avg_wait_time:.2f}")
    print(f"2. 最长顾客等待时间 (长尾效应): {max_wait_time:.2f}")
    print(f"3. 总空驶成本 (调度成本):       {total_empty_cost:.2f}")
    print(f"4. 载客总里程 (服务价值):       {total_loaded_dist:.2f}")
    print(f"5. 车辆里程利用率:             {utilization_rate * 100:.2f}%")
    print("-" * 30)

    if assignment:
        sample_cust_id = list(assignment.keys())[0]
        sample_car_id = assignment[sample_cust_id]
        dist, path = details[(sample_cust_id, sample_car_id)]
        print(f"\n[样例] 顾客 {sample_cust_id} 被指派给 车辆 {sample_car_id}")
        print(f"       接驾距离: {dist:.2f}")
        print(f"       接驾路径: {path}")
    
    visualize_net_weights(net, title=f"Map Weights (Grid: {net.n}x{net.m}) - Max Factor Overlay, R={net.hotspot_radius}") 
    plt.show()

# 导入抽象类以便于类型引用 (如果需要，但这里只涉及数据结构)

# --- 1. 定义颜色映射 (权重越大越红) ---
def create_detailed_colormap():
# ... (函数体不变) ...
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
# ... (函数体不变) ...
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
    # 核心修改 3: 用 plt.draw() 替代 plt.show()
    plt.draw()