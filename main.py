import random
import numpy as np
from abstract_class import Net, Customer, Car, Node
from main_method import match_orders, shortest_path

def generate_mock_data(net, num_orders=20):
    """
    生成测试数据：
    为了观察“闹市区”效应，我们可以让更多顾客出现在 hotspot 类型的节点。
    """
    customers = []
    cars = []
    
    # 1. 找出所有的 Hotspot 节点和 Normal 节点
    hotspots = [n for n in net.nodes if n.type == 'hotspot']
    normals = [n for n in net.nodes if n.type == 'normal']
    
    print(f"地图统计: 总节点 {len(net.nodes)}, 闹市区节点 {len(hotspots)}, 普通节点 {len(normals)}")

    for i in range(num_orders):
        # --- 生成顾客 ---
        # 假设 70% 的订单起点在闹市区 (如果闹市区存在)
        if hotspots and random.random() < 0.7:
            start_node = random.choice(hotspots)
        else:
            start_node = random.choice(net.nodes)
            
        # 终点随机
        end_node = random.choice(net.nodes)
        
        # 确保起点终点不重合
        while end_node.id == start_node.id:
            end_node = random.choice(net.nodes)
            
        cust = Customer(id=i, start_node=start_node, end_node=end_node, creation_time=0)
        customers.append(cust)

        # --- 生成车辆 ---
        # 车辆随机分布
        car_start_node = random.choice(net.nodes)
        car = Car(id=i, current_node=car_start_node)
        cars.append(car)
        
    return customers, cars

def run_experiment():
    # 1. 初始化网络 (10x10 网格)
    print(">>> 正在初始化网络...")
    net = Net(10, 10)
    
    # 2. 生成数据 (20个订单，20辆车)
    print(">>> 正在生成模拟数据...")
    customers, cars = generate_mock_data(net, num_orders=20)
    
    # 3. 运行核心调度算法
    print(">>> 正在进行订单匹配 (匈牙利算法)...")
    # assignment: {cust_id: car_id}
    # total_empty_cost: 所有车辆去接乘客的总空驶代价
    # details: {(cust_id, car_id): (distance, path)}
    assignment, total_empty_cost, details = match_orders(customers, cars, net) # type: ignore
    
    if assignment is None:
        print("匹配失败，请检查车辆和订单数量。")
        return

    # 4. 计算核心指标
    print(">>> 正在计算核心实验指标...")
    
    wait_times = []      # 顾客等待时间 (空驶距离)
    loaded_distances = [] # 载客行程距离 (订单起点 -> 终点)
    
    # 遍历所有匹配结果
    for cust_id, car_id in assignment.items():
        # --- A. 获取空驶数据 (Pickup Phase) ---
        pickup_dist, _ = details[(cust_id, car_id)]
        wait_times.append(pickup_dist)
        
        # --- B. 计算载客数据 (Delivery Phase) ---
        # 这一步在匹配函数里没算，我们需要单独算一下订单本身的长度
        # 注意：这里需要根据 ID 找到对应的对象
        cust = next(c for c in customers if c.id == cust_id)
        
        # 计算从顾客起点到终点的距离
        trip_dist, _ = shortest_path(cust.start_node.id, cust.end_node.id, net)
        loaded_distances.append(trip_dist)

    # 5. 统计输出
    avg_wait_time = np.mean(wait_times)
    max_wait_time = np.max(wait_times)
    total_loaded_dist = sum(loaded_distances)
    total_total_dist = total_empty_cost + total_loaded_dist
    
    # 车辆利用率 = 载客里程 / (空驶里程 + 载客里程)
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

    # 打印一个具体的匹配案例看看
    sample_cust_id = list(assignment.keys())[0]
    sample_car_id = assignment[sample_cust_id]
    dist, path = details[(sample_cust_id, sample_car_id)]
    print(f"\n[样例] 顾客 {sample_cust_id} 被指派给 车辆 {sample_car_id}")
    print(f"       接驾距离: {dist}")
    print(f"       接驾路径: {path}")

if __name__ == "__main__":
    run_experiment()