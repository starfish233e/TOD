import random
import numpy as np
import math
from abstract_class import Net, Customer, Car, Node 
from main_method import match_orders, shortest_path, visualize_net_weights


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
    
    # 调用可视化函数 
    visualize_net_weights(net, title=f"Map Weights (Grid: {net.n}x{net.m}) - Max Factor Overlay, R={net.hotspot_radius}")
    
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