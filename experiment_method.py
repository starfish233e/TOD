import random
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap, Normalize 
from abstract_class import Net, Customer, Car
from main_method import match_orders_dynamic, shortest_path # 确保只导入需要的

# ----------------- 辅助函数 -----------------

def calculate_average_edge_weight(net):
    """计算地图所有边的平均权重，作为时间窗成本的基准 T_win"""
    valid_weights = net.adj_matrix[net.adj_matrix != np.inf]
    if len(valid_weights) == 0: return 10.0
    return np.mean(valid_weights)

def generate_new_orders(net, num_new_orders, current_id_counter, time_window_index):
    """在每个时间窗生成新订单"""
    new_customers = []
    hotspots = [n for n in net.nodes if n.type == 'hotspot']
    
    for _ in range(num_new_orders):
        # 70% 概率从闹市区出发
        if hotspots and random.random() < 0.7:
            start_node = random.choice(hotspots)
        else:
            start_node = random.choice(net.nodes)
            
        end_node = random.choice(net.nodes)
        while end_node.id == start_node.id:
            end_node = random.choice(net.nodes)
            
        cust = Customer(id=current_id_counter, start_node=start_node, end_node=end_node, creation_time=time_window_index)
        new_customers.append(cust)
        current_id_counter += 1
        
    return new_customers, current_id_counter


# ----------------- 可视化函数 (新增) -----------------

def create_detailed_colormap():
    """创建自定义颜色映射：从黄色到橙色到红色，表示流量因子增加"""
    # 颜色列表: Yellow (1.0) -> Orange -> Red (2.5)
    colors = [(1, 1, 0), (1, 0.5, 0), (1, 0, 0)] 
    cmap_name = 'hotspot_traffic'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

def visualize_net_weights(net):
    """可视化网络地图，根据流量因子对节点进行颜色编码"""
    # HOTSPOT_MAX_FACTOR 应该与 Net 类中的定义一致
    HOTSPOT_MAX_FACTOR = 2.5 
    
    x_coords = [node.x for node in net.nodes]
    y_coords = [node.y for node in net.nodes]
    
    # 1. 计算每个节点的权重因子
    node_factors = np.ones(len(net.nodes))
    for i, node in enumerate(net.nodes):
        if node.type == 'hotspot':
            # 注意: _calculate_hotspot_max_factor 是 Net 的私有方法，用于计算节点权重
            factor = net._calculate_hotspot_max_factor(node.x, node.y)
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
    cbar.set_label(f'Node Traffic Factor ({HOTSPOT_MAX_FACTOR} = Max Hotspot, 1.0 = Normal)', fontsize=12)

    # 6. 设置图表属性
    plt.title('Network Grid Visualization by Traffic Factor', fontsize=16)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xticks(np.arange(net.m))
    plt.yticks(np.arange(net.n))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# ----------------- 主实验逻辑 -----------------

def run_experiment():
    # 1. 参数设置
    GRID_W, GRID_H = 20, 20
    NUM_WINDOWS = 8          # 仿真运行 8 个时间窗
    INITIAL_CARS = 15        # 只有 15 辆车
    NEW_ORDERS_PER_WIN = 20  # 每个窗口产生 20 个订单 (订单 > 车辆，必定产生积压)
    
    print(f">>> 初始化网络 ({GRID_W}x{GRID_H})...")
    net = Net(GRID_W, GRID_H)
    
    # 计算 T_win: 时间窗的权重因子 (用于平衡距离成本和等待成本)
    # 设为平均路段长度的 3 倍，意味着等待 1 个窗口相当于多跑 3 个平均路段的距离
    avg_edge = calculate_average_edge_weight(net)
    T_win = avg_edge * 3.0 
    print(f">>> 计算 T_win (时间窗权重) = {T_win:.2f} (基于地图平均边权)")

    # 2. 初始化车辆
    cars = []
    for i in range(INITIAL_CARS):
        start_node = random.choice(net.nodes)
        cars.append(Car(id=i, current_node=start_node))
    
    waiting_customers = [] # 积压池
    customer_id_counter = 0
    
    # 统计数据容器
    history_stats = []

    print(f">>> 开始仿真: {NUM_WINDOWS} 个时间窗, {INITIAL_CARS} 辆车, 每轮新增 {NEW_ORDERS_PER_WIN} 订单")
    print("-" * 60)

    # 3. 时间窗循环
    for t in range(1, NUM_WINDOWS + 1):
        print(f"\n[时间窗 {t}/{NUM_WINDOWS}]")
        
        # 3.1 生成新订单
        new_orders, customer_id_counter = generate_new_orders(net, NEW_ORDERS_PER_WIN, customer_id_counter, t)
        waiting_customers.extend(new_orders)
        
        # 3.2 筛选可用车辆 (简化：假设上一轮匹配的车这一轮都完成任务变为空闲)
        available_cars = cars # 所有车都可用 (简化模型)
        
        print(f"   当前等待顾客数: {len(waiting_customers)} | 可用车辆数: {len(available_cars)}")
        
        # 3.3 核心匹配 (带等待权重)
        assignment, total_empty_dist, details = match_orders_dynamic(waiting_customers, available_cars, net, T_win)
        
        # 3.4 处理匹配结果
        matched_cust_ids = set(assignment.keys())
        
        current_window_wait_times = []
        current_window_loaded_dist = 0
        
        # 3.4.1 处理已匹配顾客
        unmatched_customers = []
        
        for cust in waiting_customers:
            if cust.id in matched_cust_ids:
                car_id = assignment[cust.id]
                car = next(c for c in cars if c.id == car_id)
                
                pickup_dist, _ = details[(cust.id, car.id)]
                
                # 更新车辆位置到顾客终点 (为下一轮做准备)
                car.current_node = cust.end_node
                
                # 统计
                wait_cost_time = (cust.missed_windows * T_win) + pickup_dist # 广义等待成本
                current_window_wait_times.append(wait_cost_time)
                
                trip_dist, _ = shortest_path(cust.start_node.id, cust.end_node.id, net)
                current_window_loaded_dist += trip_dist
                
            else:
                # 3.4.2 处理未匹配顾客
                cust.missed_windows += 1 # 增加等待计数
                unmatched_customers.append(cust)
        
        # 更新等待池，只保留未匹配的
        waiting_customers = unmatched_customers
        
        # 3.5 记录本轮数据
        avg_wait = np.mean(current_window_wait_times) if current_window_wait_times else 0
        total_total_dist = total_empty_dist + current_window_loaded_dist
        utilization = current_window_loaded_dist / total_total_dist if total_total_dist > 0 else 0
        
        stats = {
            'window': t,
            'matched': len(matched_cust_ids),
            'left_over': len(waiting_customers),
            'avg_wait_score': avg_wait,
            'total_empty': total_empty_dist,
            'utilization': utilization
        }
        history_stats.append(stats)
        
        print(f"   >>> 匹配成功: {stats['matched']}, 滞留: {stats['left_over']}")
        print(f"   >>> 本轮空驶: {stats['total_empty']:.1f}, 平均等待分数: {stats['avg_wait_score']:.1f}")

    # 4. 最终总结
    print("\n" + "="*60)
    print("📊 仿真结束总结报告")
    print("="*60)
    print(f"{'时间窗':<5} | {'匹配数':<5} | {'滞留数':<7} | {'利用率':<8} | {'平均等待分数':<12}")
    print("-" * 60)
    for s in history_stats:
        print(f"{s['window']:<8} | {s['matched']:<8} | {s['left_over']:<10} | {s['utilization']*100:.1f}%{'':<6} | {s['avg_wait_score']:.2f}")
    
    print("-" * 60)
    leftover_max_wait = max([c.missed_windows for c in waiting_customers]) if waiting_customers else 0
    print(f"最终滞留顾客数: {len(waiting_customers)}")
    print(f"滞留最久的顾客已等待: {leftover_max_wait} 个时间窗")
    
    # 5. 可视化地图
    print("\n正在生成地图可视化...")
    visualize_net_weights(net)

if __name__ == "__main__":
    run_experiment()