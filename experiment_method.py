import random
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap, Normalize 
from abstract_class import Net, Customer, Car 
from main_method import match_orders_dynamic, shortest_path # 确保 match_orders_dynamic 接受 alpha
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

# ----------------- 辅助函数 (保持不变) -----------------

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

def create_detailed_colormap():
    """创建自定义颜色映射：从黄色到橙色到红色，表示流量因子增加"""
    colors = [(1, 1, 0), (1, 0.5, 0), (1, 0, 0)] 
    cmap_name = 'hotspot_traffic'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    return cm

def visualize_net_weights(net):
    """可视化网络地图，根据流量因子对节点进行颜色编码"""
    HOTSPOT_MAX_FACTOR = 2.5 
    
    x_coords = [node.x for node in net.nodes]
    y_coords = [node.y for node in net.nodes]
    
    node_factors = np.ones(len(net.nodes))
    for i, node in enumerate(net.nodes):
        if node.type == 'hotspot':
            # 假设 Net 类内部有 _calculate_hotspot_max_factor 方法
            factor = net._calculate_hotspot_max_factor(node.x, node.y)
            node_factors[i] = factor

    cmap = create_detailed_colormap()
    min_factor = 1.0 
    max_factor = HOTSPOT_MAX_FACTOR 
    norm = Normalize(min_factor, max_factor)
    
    plt.figure(figsize=(10, 10)) 
    
    scatter = plt.scatter(x_coords, y_coords, 
                          c=node_factors, 
                          cmap=cmap, 
                          norm=norm, 
                          s=150, 
                          edgecolors='black', 
                          linewidths=0.5)

    traffic_light_indices = [i for i, n in enumerate(net.nodes) if n.type == 'traffic_light']
    tl_x = [x_coords[i] for i in traffic_light_indices]
    tl_y = [y_coords[i] for i in traffic_light_indices]
    
    plt.scatter(tl_x, tl_y, s=200, marker='s', color='magenta', alpha=0.8, label='红绿灯节点')

    cbar = plt.colorbar(scatter, fraction=0.04, pad=0.04)
    cbar.set_label(f'节点交通因子 ({HOTSPOT_MAX_FACTOR} = 闹市区最大值, 1.0 = 正常)', fontsize=12)

    plt.title('网络地图可视化 (基于交通因子)', fontsize=16)
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.xticks(np.arange(net.m))
    plt.yticks(np.arange(net.n))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# ----------------- 核心仿真逻辑 (可复用) -----------------

def _run_simulation_core(net, initial_cars, new_orders_per_win, num_windows, T_win, alpha):
    """
    运行单个 alpha 值下的仿真实验，返回详细历史和最终统计。
    """
    # 每次运行需要重新初始化车辆和顾客数据
    cars = []
    for i in range(initial_cars):
        start_node = random.choice(net.nodes)
        cars.append(Car(id=i, current_node=start_node))
    
    waiting_customers = [] 
    customer_id_counter = 0
    
    history_stats = []

    # 累积指标
    total_loaded_dist_acc = 0
    total_empty_dist_acc = 0
    total_wait_score_acc = 0
    total_matched_customers = 0

    # 1. 时间窗循环
    for t in range(1, num_windows + 1):
        
        new_orders, customer_id_counter = generate_new_orders(net, new_orders_per_win, customer_id_counter, t)
        waiting_customers.extend(new_orders)
        
        available_cars = cars 
        
        # 核心匹配：传入 alpha
        assignment, total_empty_dist, details = match_orders_dynamic(waiting_customers, available_cars, net, T_win, alpha)
        
        matched_cust_ids = set(assignment.keys())
        
        current_window_wait_times = []
        current_window_loaded_dist = 0
        
        unmatched_customers = []
        
        for cust in waiting_customers:
            if cust.id in matched_cust_ids:
                car_id = assignment[cust.id]
                car = next(c for c in cars if c.id == car_id)
                
                pickup_dist, _ = details[(cust.id, car.id)]
                car.current_node = cust.end_node
                
                # 统计等待分数 (用于报告)
                wait_cost_time = (cust.missed_windows * T_win) + pickup_dist 
                current_window_wait_times.append(wait_cost_time)
                
                trip_dist, _ = shortest_path(cust.start_node.id, cust.end_node.id, net)
                current_window_loaded_dist += trip_dist
                
            else:
                cust.missed_windows += 1 
                unmatched_customers.append(cust)
        
        waiting_customers = unmatched_customers
        
        # 记录本轮数据
        matched_count = len(matched_cust_ids)
        total_total_dist = total_empty_dist + current_window_loaded_dist
        utilization = current_window_loaded_dist / total_total_dist if total_total_dist > 0 else 0
        avg_wait = np.mean(current_window_wait_times) if current_window_wait_times else 0

        stats = {
            'window': t,
            'matched': matched_count,
            'left_over': len(waiting_customers),
            'avg_wait_score': avg_wait,
            'total_empty': total_empty_dist,
            'utilization': utilization
        }
        history_stats.append(stats)
        
        # 累加总指标
        if current_window_wait_times:
            total_wait_score_acc += sum(current_window_wait_times)
            
        total_loaded_dist_acc += current_window_loaded_dist
        total_empty_dist_acc += total_empty_dist
        total_matched_customers += matched_count
    
    # 计算最终性能指标
    total_total_dist_final = total_empty_dist_acc + total_loaded_dist_acc
    final_utilization = total_loaded_dist_acc / total_total_dist_final if total_total_dist_final > 0 else 0
    final_avg_wait_score = total_wait_score_acc / total_matched_customers if total_matched_customers > 0 else 0
    leftover_cust = len(waiting_customers)
    
    final_summary = {
        'alpha': alpha,
        'final_utilization': final_utilization,
        'final_avg_wait_score': final_avg_wait_score,
        'total_matched': total_matched_customers,
        'leftover_cust': leftover_cust
    }

    return history_stats, final_summary

# ----------------- 主实验驱动函数 -----------------

def run_experiment(run_grid_search=False):
    """
    驱动仿真实验。
    如果 run_grid_search=False (默认)，则以 alpha=1.0 运行单次实验。
    如果 run_grid_search=True，则运行网格搜索。
    """
    
    # 1. 固定参数设置
    GRID_W, GRID_H = 20, 20
    NUM_WINDOWS = 2          
    INITIAL_CARS = 15       
    NEW_ORDERS_PER_WIN = 20 
    
    print(f">>> 初始化网络 ({GRID_W}x{GRID_H})...")
    net = Net(GRID_W, GRID_H)
    
    avg_edge = calculate_average_edge_weight(net)
    T_win = avg_edge * 3.0 
    print(f">>> T_win (时间窗权重) = {T_win:.2f}")

    if run_grid_search:
        # --- 网格搜索模式 ---
        alpha_values = np.linspace(0.0, 1.0, num=11)
        search_results = []
        
        print("\n" + "="*80)
        print("🔬 开始网格搜索 (alpha: 0.0 -> 1.0)")
        print("="*80)

        for alpha in alpha_values:
            print(f"\n--- 运行 alpha={alpha:.2f} 的实验 ---")
            # 运行核心模拟，但只保留最终统计信息
            _, result = _run_simulation_core(
                net, INITIAL_CARS, NEW_ORDERS_PER_WIN, NUM_WINDOWS, T_win, alpha
            )
            search_results.append(result)
            print(f"   --- 结果：匹配 {result['total_matched']} | 滞留 {result['leftover_cust']} | 利用率 {result['final_utilization']*100:.2f}% | 平均等待 {result['final_avg_wait_score']:.2f}")


        # 4. 输出最终网格搜索报告
        print("\n" + "="*80)
        print("⭐ 网格搜索最终报告")
        print("="*80)
        
        print(f"{'alpha':<8} | {'总匹配数':<8} | {'最终滞留':<10} | {'总利用率':<10} | {'平均等待分数':<15}")
        print("-" * 80)
        
        best_result = None
        
        for r in search_results:
            print(f"{r['alpha']:<8.2f} | {r['total_matched']:<8} | {r['leftover_cust']:<10} | {r['final_utilization']*100:.2f}%{'':<6} | {r['final_avg_wait_score']:.2f}")
            
            # 假设简单的选择标准：最大化利用率
            if best_result is None or r['final_utilization'] > best_result['final_utilization']:
                 best_result = r
                 
        print("-" * 80)
        if best_result:
            print(f"✅ 建议的最优 alpha: {best_result['alpha']:.2f} (以最大化利用率为目标)")


    else:
        # --- 单次实验模式 (默认 alpha=1.0) ---
        FIXED_ALPHA = 1.0 
        print("\n" + "="*80)
        print(f"🚀 开始单次仿真实验 (默认 alpha={FIXED_ALPHA:.2f}：仅考虑距离成本)")
        print("="*80)

        history_stats, final_summary = _run_simulation_core(
            net, INITIAL_CARS, NEW_ORDERS_PER_WIN, NUM_WINDOWS, T_win, FIXED_ALPHA
        )
        
        # 2. 最终总结报告 (包含时间窗细节)
        print("\n" + "="*60)
        print(f"📊 仿真结束总结报告 (alpha={FIXED_ALPHA:.2f})")
        print("="*60)
        
        print(f"{'时间窗':<5} | {'匹配数':<5} | {'滞留数':<7} | {'利用率':<8} | {'平均等待分数':<15}")
        print("-" * 60)
        
        for s in history_stats:
            print(f"{s['window']:<8} | {s['matched']:<8} | {s['left_over']:<10} | {s['utilization']*100:.1f}%{'':<6} | {s['avg_wait_score']:.2f}")
        
        print("-" * 60)
        print(f"总体平均等待分数: {final_summary['final_avg_wait_score']:.2f}")
        print(f"总体车辆里程利用率: {final_summary['final_utilization']*100:.2f}%")
        print(f"总匹配顾客数: {final_summary['total_matched']}")
        print(f"最终滞留顾客数: {final_summary['leftover_cust']}")
        print("-" * 60)
    
    # 3. 可视化地图 (两种模式都执行)
    print("\n正在生成地图可视化...")
    visualize_net_weights(net)

if __name__ == "__main__":
    # 默认运行单次实验 (alpha=1.0)
    # 如果要运行网格搜索，请修改为 run_experiment(run_grid_search=True)
    run_experiment(run_grid_search=False)