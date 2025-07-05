import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import time
import pulp


def plot_routes(x, y, best_routes, best_cost, gruz, clients, trucks):
    # Plot the depot (garage) as a red square
    plt.plot(x[0], y[0], 'ks', linewidth=2, markersize=14, markerfacecolor='r')
    plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.ylabel('Расстояния, км')
    plt.xlabel('Расстояния, км')
    plt.grid(True)
    
    # Plot routes for each truck
    for truck in range(trucks):
        ind = best_routes[truck]
        plt.plot(x[ind], y[ind], 'o-', linewidth=2, markersize=8)
    
    # Set title with cargo and fuel cost
    title_text = f'Груз клиентам: {np.mean(gruz):.1f} ± {np.std(gruz):.1f} кг   Затраты топлива: {best_cost:.1f} литров'
    plt.title(title_text)
    
    # Add legend
    plt.legend(['Гараж', 'Газель 1', 'Газель 2', 'Isuzu 1', 'Isuzu 2', 'КамАЗ'], loc='best')
    
    # Label clients
    for i in range(1, clients):
        plt.text(x[i]+1, y[i]+1, str(i), verticalalignment='bottom', horizontalalignment='right')
    
    plt.show()

def plot_bar_dlina_fuels(m_trucks, best_routes, D, fuel_per_km, txt):
    # Инициализация массивов
    truck_dist = np.zeros(m_trucks)
    truck_dist_no1 = np.zeros(m_trucks)
    
    # Расчет пробегов
    for truck in range(m_trucks):
        ind = best_routes[truck]
        truck_dist_no1[truck] = calc_route_cost(ind[:-1], D, 1)
        truck_dist[truck] = calc_route_cost(ind, D, 1)
    
    # Расчет расхода топлива
    truck_fuel = truck_dist * fuel_per_km
    
    # Создание фигуры
    plt.figure()
    
    # Первый подграфик: пробег
    plt.subplot(2, 1, 1)
    x = np.arange(m_trucks)  # Позиции для грузовиков
    width = 0.35  # Ширина каждого столбца (меньше, чтобы столбцы не перекрывались)
    
    # Построение сгруппированных столбцов
    plt.bar(x - width/2, truck_dist_no1, width, color='y', label='Развоз')
    plt.bar(x + width/2, truck_dist, width, color='b', label='В гараж')
    
    # Настройка заголовка, меток и сетки
    plt.title(f'Пробег грузовиков: {round(sum(truck_dist))} км')
    plt.ylabel('Пробег (км)')
    plt.grid(True)
    plt.xticks(x, txt)
    plt.xlim(-0.5, m_trucks - 0.5)
    plt.ylim(0, max(truck_dist) * 1.15)
    plt.legend(loc='best')
    
    # Добавление числовых значений над столбцами
    plot_value1(np.array([truck_dist_no1, truck_dist]).T, m_trucks, 'k')
    
    # Второй подграфик: расход топлива
    plt.subplot(2, 1, 2)
    plt.bar(x, truck_fuel, color='g')
    plt.title(f'Расход топлива грузовиками: {round(sum(truck_fuel))} литр')
    plt.ylabel('Расход (литр)')
    plt.grid(True)
    plt.xticks(x, txt)
    plt.ylim(0, max(truck_fuel) * 1.15)
    plot_value(truck_fuel, m_trucks, 'k')
    
    plt.tight_layout()
    plt.show()
    
    return truck_dist, truck_fuel
def plot_value1(param, n_trucks, color):
    dy = np.max(param) * 0.1
    for i in range(n_trucks):
        for j in range(param.shape[1]):
            x = i - 0.22 + (j * 0.44)
            y = param[i, j]
            if y > 0.1:
                plt.text(x, y + dy, str(round(y)), ha='center', fontsize=12, color=color)

def plot_value(param, n_trucks, color):
    dy = np.max(param) * 0.1
    for i in range(n_trucks):
        b = param[i]
        a = str(round(b)) if b > 0.1 else ''
        plt.text(i, b - dy, a, ha='center', fontsize=12, color=color)

def plot_bar_client_gruz(trucks, best_routes, gruz, txt):
    truck_loads = np.zeros(trucks)
    truck_clients = np.zeros(trucks)
    
    for truck in range(trucks):
        ind = np.array(best_routes[truck])  # Преобразуем в numpy массив
        ind = ind[ind != 0]  # Убираем гараж (индекс 0 в Python)
        if len(ind) > 0:  # Проверяем, что есть клиенты
            truck_loads[truck] = np.sum(gruz[ind])  # Суммируем грузы
        else:
            truck_loads[truck] = 0  # Если нет клиентов, груз равен 0
        truck_clients[truck] = len(ind)  # Количество клиентов
    
    plt.figure()
    
    # Первый подграфик: количество клиентов
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(trucks), truck_clients, color=[0.2, 0.6, 1])
    plt.ylabel('Количество клиентов')
    plt.title('Количество клиентов, обслуживаемых грузовиками')
    plt.xticks(np.arange(trucks), txt)
    plt.grid(True)
    plot_value(truck_clients, trucks, 'w')
    
    # Второй подграфик: загрузка грузовиков
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(trucks), truck_loads, color=[1, 0.6, 0.6])
    plt.ylabel('Загрузка (объем)')
    plt.title(f'Загрузка грузовиков: {round(sum(truck_loads))} кг')
    plt.xticks(np.arange(trucks), txt)
    plt.grid(True)
    plt.ylim(0, max(truck_loads) * 1.15)
    plot_value(truck_loads, trucks, 'k')
    
    plt.tight_layout()
    plt.show()
    
    return truck_loads, truck_clients

def local_optim(best_routes, dist, fuel_per_km):
    trucks = len(best_routes)
    new_routes = [None] * trucks
    new_dlina = 0
    
    for i in range(trucks):
        route, dlina = two_opt(best_routes[i], dist)
        new_routes[i] = route
        new_dlina += dlina * fuel_per_km[i]
    
    return new_routes, new_dlina

def two_opt(route, dist):
    best_route = route.copy()
    best_distance = calc_route_cost(route, dist, 1)
    improvement = True
    
    while improvement:
        improvement = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = two_opt_swap(best_route, i, j)
                new_distance = calc_route_cost(new_route, dist, 1)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
    
    return best_route, best_distance

def two_opt_swap(route, i, j):
    return route[:i] + route[i:j+1][::-1] + route[j+1:]

def ant_one(trucks, gruz, dist, pheromones, aco):
    visited = trucks['visited'].copy()  # Список клиентов, уже посещённых каким-либо грузовиком
    routes = [None] * trucks['m']       # Список маршрутов для каждого грузовика
    total_cost = 0                      # Общая стоимость маршрутов
    test_cities = 0                     # Количество обслуженных клиентов (без учёта гаража)

    for t_idx in range(trucks['m']):
        truck = trucks['order'][t_idx]          # Текущий грузовик по порядку
        curr_client = 0                         # Начинаем с гаража (индекс 0)
        curr_capacity = trucks['v'][truck]      # Оставшаяся грузоподъёмность
        route = [curr_client]                   # Начальный маршрут: из гаража
        visited[curr_client] = True
        current_dist = 0                        # Пройденное расстояние для этого грузовика

        # Пока есть место в грузовике и остались неотгруженные клиенты
        while curr_capacity > 0 and sum(gruz[route]) < sum(gruz):
            # Выбираем следующего клиента по вероятностной функции (с учётом феромонов и расстояний)
            next_client = select_next_client(
                curr_client, pheromones, dist,
                aco['alpha'], aco['beta'],
                gruz, curr_capacity, visited
            )

            if next_client == -1:
                break  # Нет подходящих клиентов

            dist_to_next = dist[curr_client, next_client]

            # Проверяем, сможет ли грузовик вернуться в гараж после посещения клиента
            if current_dist + dist_to_next + trucks['to_home'][truck] * dist[next_client, 0] > trucks['max_dist'][truck]:
                break  # Превышение лимита по пробегу

            # Обновляем маршрут и параметры
            current_dist += dist_to_next
            route.append(next_client)
            curr_capacity -= gruz[next_client]
            curr_client = next_client
            visited[curr_client] = True

        test_cities += len(route) - 1  # Учитываем количество клиентов (без гаража)
        route.append(0)                # Возврат в гараж
        routes[truck] = route

        # Считаем стоимость маршрута с учётом расхода топлива этого грузовика
        total_cost += calc_route_cost(route, dist, trucks['fuel'][truck])

    return routes, total_cost, test_cities  # Возвращаем маршруты, общую стоимость и число обслуженных клиентов


def select_next_client(curr_client, pheromones, dist, alpha, beta, gruz, curr_capacity, visited, use_random=False):
    clients = dist.shape[0]
    prob = np.zeros(clients)
    
    if use_random:
        # Случайный выбор клиента
        available_clients = [j for j in range(clients) if j != curr_client and gruz[j] <= curr_capacity and not visited[j]]
        return random.choice(available_clients) if available_clients else -1
    
    # Pheromone-based selection
    for j in range(clients):
        if j != curr_client and gruz[j] <= curr_capacity and not visited[j]:
            prob[j] = (pheromones[curr_client, j]**alpha) * (1 / dist[curr_client, j])**beta
    
    if sum(prob) == 0:
        return -1
    
    prob /= sum(prob)
    return np.random.choice(clients, p=prob)

def calc_route_cost(route, dist, fuel_per_km):
    cost = 0
    for i in range(len(route)-1):
        cost += dist[route[i], route[i+1]]
    return cost * fuel_per_km

def ant_colony(clients, trucks, demand, dist, aco, local_opt):
    pheromones = np.ones((clients, clients))
    best_cost = float('inf')
    best_routes = []
    end_iter = 0
    
    for iter in range(aco['iters']):
        for ant in range(aco['ants']):
            np.random.seed(random.randint(1, 10000))
            trucks['visited'] = np.zeros(clients, dtype=bool)
            trucks['order'] = np.random.permutation(trucks['m'])
            
            routes, total_cost, test_cities = ant_one(trucks, demand, dist, pheromones, aco)
            
            if total_cost < best_cost and test_cities == clients - 1:
                print(f'--- Iter={iter+1} --- Ant={ant+1} --- TotalCost={total_cost:.1f}')
                if local_opt:
                    best_routes, best_cost = local_optim(routes, dist, trucks['fuel'])
                    print(f'----- Iter={iter+1} --- Ant={ant+1} --- BestCost={best_cost:.1f}')
                else:
                    best_cost = total_cost
                    best_routes = routes
                end_iter = iter + 1
            
            for t in range(len(routes)):
                for j in range(len(routes[t])-1):
                    pheromones[routes[t][j], routes[t][j+1]] += 1 / total_cost
        
        pheromones *= (1 - aco['rho'])
    
    return best_routes, best_cost, end_iter

# Метод ближайшего соседа
def nearest_neighbor(clients, trucks, gruz, dist):
    visited = np.zeros(clients, dtype=bool)
    routes = [None] * trucks['m']
    total_cost = 0
    
    visited[0] = True  # Депо отмечено как посещенное
    remaining_clients = clients - 1  # Количество клиентов, которых нужно посетить (исключая депо)
    
    for t_idx in range(trucks['m']):
        truck = t_idx  # Используем грузовики по порядку
        curr_client = 0  # Начинаем из депо
        curr_capacity = trucks['v'][truck]
        route = [curr_client]
        current_dist = 0
        
        while curr_capacity > 0 and np.sum(visited[1:]) < remaining_clients:
            # Находим ближайшего непосещенного клиента, удовлетворяющего ограничениям
            min_dist = float('inf')
            next_client = -1
            
            for j in range(1, clients):
                if not visited[j] and gruz[j] <= curr_capacity:
                    dist_to_j = dist[curr_client, j]
                    # Проверяем, что добавление клиента и возврат в депо не превышает max_dist
                    if current_dist + dist_to_j + dist[j, 0] <= trucks['max_dist'][truck]:
                        if dist_to_j < min_dist:
                            min_dist = dist_to_j
                            next_client = j
            
            if next_client == -1:
                break  # Нет подходящих клиентов
            
            # Добавляем клиента в маршрут
            route.append(next_client)
            visited[next_client] = True
            curr_capacity -= gruz[next_client]
            current_dist += min_dist
            curr_client = next_client
        
        route.append(0)  # Возвращаемся в депо
        routes[truck] = route
        total_cost += calc_route_cost(route, dist, trucks['fuel'][truck])
    
    # Проверяем, все ли клиенты посещены
    if np.sum(visited[1:]) < remaining_clients:
        print("Внимание: Не все клиенты были посещены методом ближайшего соседа!")
    
    return routes, total_cost

def solve_vrp_milp(clients, trucks, gruz, dist):
    try:
        # Инициализация модели с минимизацией
        model = pulp.LpProblem("VRP", pulp.LpMinimize)
        
        # Переменные: x[i,j,k] = 1, если грузовик k едет от i к j
        x = {}
        for i in range(clients):
            for j in range(clients):
                for k in range(trucks['m']):
                    if i != j:
                        x[i, j, k] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
        
        # Переменные для предотвращения подциклов (MTZ-формулировка)
        u = {}
        for i in range(1, clients):
            u[i] = pulp.LpVariable(f"u_{i}", lowBound=0, upBound=clients-1, cat="Continuous")
        
        # Целевая функция
        fuel_cost = pulp.lpSum(dist[i, j] * trucks['fuel'][k] * x[i, j, k] 
                               for i in range(clients) for j in range(clients) 
                               for k in range(trucks['m']) if i != j)
        model += fuel_cost, "Total_Fuel_Cost"
        
        # Ограничения
        # 1. Каждый клиент (кроме депо) посещается ровно одним грузовиком
        for j in range(1, clients):
            model += pulp.lpSum(x[i, j, k] for i in range(clients) for k in range(trucks['m']) if i != j) == 1, f"Visit_{j}"
        
        # 2. Ограничения на грузоподъемность
        for k in range(trucks['m']):
            model += pulp.lpSum(gruz[j] * pulp.lpSum(x[i, j, k] for i in range(clients) if i != j) 
                                for j in range(1, clients)) <= trucks['v'][k], f"Capacity_{k}"
        
        # 3. Ограничения на максимальную дистанцию
        for k in range(trucks['m']):
            model += pulp.lpSum(dist[i, j] * x[i, j, k] 
                                for i in range(clients) for j in range(clients) if i != j) <= trucks['max_dist'][k], f"Distance_{k}"
        
        # 4. Поток: грузовик должен выйти и вернуться в депо
        for k in range(trucks['m']):
            model += pulp.lpSum(x[0, j, k] for j in range(1, clients)) >= 1, f"Leave_Depot_{k}"
            model += pulp.lpSum(x[j, 0, k] for j in range(1, clients)) >= 1, f"Return_Depot_{k}"
            for j in range(1, clients):
                model += (pulp.lpSum(x[i, j, k] for i in range(clients) if i != j) == 
                          pulp.lpSum(x[j, i, k] for i in range(clients) if i != j), f"Flow_{j}_{k}")
        
        # 5. Предотвращение подциклов (MTZ-формулировка)
        for i in range(1, clients):
            for j in range(1, clients):
                if i != j:
                    for k in range(trucks['m']):
                        model += u[i] - u[j] + clients * x[i, j, k] <= clients - 1, f"Subtour_{i}_{j}_{k}"
        
        # Решение задачи с ограничением времени (5 минут) и 4 потоками
        solver = pulp.PULP_CBC_CMD(timeLimit=300, threads=12, msg=1)  # msg=1 для вывода логов
        status = model.solve(solver)
        if status != 1:
            print(f"Статус решения: {pulp.LpStatus[status]}")
            return None, None
        
        # Извлечение маршрутов
        routes = [[] for _ in range(trucks['m'])]
        for k in range(trucks['m']):
            routes[k].append(0)
            current = 0
            visited = set([0])
            while True:
                next_node = None
                for j in range(clients):
                    if j not in visited and j != current and pulp.value(x[current, j, k]) == 1:
                        next_node = j
                        break
                if next_node is None or next_node == 0:
                    routes[k].append(0)
                    break
                routes[k].append(next_node)
                visited.add(next_node)
                current = next_node
        
        total_cost = pulp.value(model.objective)
        return routes, total_cost
    
    except Exception as e:
        print(f"Ошибка при решении MILP: {str(e)}")
        return None, None

# Обновленный основной скрипт
if __name__ == "__main__":

    # Truck parameters
    trucks = {
        'm': 5,
        'v': [1500, 1500, 5000, 5000, 13000],
        'fuel': [0.12, 0.12, 0.16, 0.16, 0.33],
        'max_dist': [1000, 1000, 1000, 1000, 1000],
        'to_home': [0, 0, 0, 0, 0],
        'visited': None,
        'order': None
    }
    
    # Load data
    x = np.loadtxt('x.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')
    dist = np.loadtxt('D.csv', delimiter=',')
    gruz = np.loadtxt('gruz.csv', delimiter=',')
    
    clients = len(x)
    gruz_cl = gruz[1:]
    
    # ACO parameters
    aco = {
        'ants': 30,
        'iters': 50,
        'alpha': 1,
        'beta': 10,
        'rho': 0.05
    }
    
    np.random.seed(80)
    local_opt = True
    
    # Run ACO
   
    start_time = time.time()
    best_routes_aco, best_cost_aco, end_iter = ant_colony(clients, trucks, gruz, dist, aco, local_opt)
    aco_time = time.time() - start_time
    
    # Run Nearest Neighbor
    start_time = time.time()
    best_routes_nn, best_cost_nn = nearest_neighbor(clients, trucks, gruz, dist)
    nn_time = time.time() - start_time
    
    # Локальная оптимизация для метода ближайшего соседа
    best_routes_nn_opt, best_cost_nn_opt = local_optim(best_routes_nn, dist, trucks['fuel'])
    
    # Визуализация маршрутов ACO
    print("\n=== Результаты муравьиного алгоритма ===")
    plt.figure()
    plot_routes(x, y, best_routes_aco, best_cost_aco, gruz, clients, trucks['m'])
    
    # Визуализация маршрутов Nearest Neighbor
    print("\n=== Результаты метода ближайшего соседа ===")
    plt.figure()
    plot_routes(x, y, best_routes_nn, best_cost_nn, gruz, clients, trucks['m'])
    
    # Вывод результатов ACO
    print("\n=== Результаты муравьиного алгоритма ===")
    print(f"Время выполнения: {aco_time:.2f} секунд")
    print(f"Суммарный перевозимый груз: {sum(gruz):.1f} кг")
    print(f"Общая потребность топлива: {best_cost_aco:.1f} литров")
    co2_aco = best_cost_aco * 2.7
    print(f"Общие выбросы CO₂: {co2_aco:.1f} кг")
    print("Лучшие маршруты для каждого грузовика:")
    
    truck_loads_aco = np.zeros(trucks['m'])
    truck_clients_aco = np.zeros(trucks['m'])
    for truck in range(trucks['m']):
        print(f"Грузовик {truck+1}: {best_routes_aco[truck]}")
        ind = np.array(best_routes_aco[truck])
        ind = ind[ind != 0]  # Убираем депо
        truck_loads_aco[truck] = sum(gruz[ind])
        truck_clients_aco[truck] = len(ind)
    
    print("Загрузка грузовиков:", truck_loads_aco)
    print("Клиентов на грузовик:", truck_clients_aco)
    
    # Расчет пробега и расхода топлива для ACO
    txt = ['Газель 1', 'Газель 2', 'Isuzu 1', 'Isuzu 2', 'КамАЗ']
    truck_dist_aco, truck_fuel_aco = plot_bar_dlina_fuels(trucks['m'], best_routes_aco, dist, trucks['fuel'], txt)
    print("Пробег грузовиков, км:", truck_dist_aco)
    print("Расход топлива, литры:", truck_fuel_aco)
    
    # Расчет загрузки и клиентов для ACO
    truck_loads_aco, truck_clients_aco = plot_bar_client_gruz(trucks['m'], best_routes_aco, gruz, txt)
    print("Загрузка грузовиков:", truck_loads_aco)
    print("Клиентов на грузовик:", truck_clients_aco)
    
    # Вывод результатов Nearest Neighbor
    print("\n=== Результаты метода ближайшего соседа ===")
    print(f"Время выполнения: {nn_time:.2f} секунд")
    print(f"Суммарный перевозимый груз: {sum(gruz):.1f} кг")
    print(f"Общая потребность топлива (до оптимизации): {best_cost_nn:.1f} литров")
    print(f"Общая потребность топлива (после оптимизации): {best_cost_nn_opt:.1f} литров")
    co2_nn = best_cost_nn * 2.7
    print(f"Общие выбросы CO₂ (до оптимизации): {co2_nn:.1f} кг")
    co2_nn_opt = best_cost_nn_opt * 2.7
    print(f"Общие выбросы CO₂ (после оптимизации): {co2_nn_opt:.1f} кг")
    print("Лучшие маршруты для каждого грузовика:")
    
    truck_loads_nn = np.zeros(trucks['m'])
    truck_clients_nn = np.zeros(trucks['m'])
    for truck in range(trucks['m']):
        print(f"Грузовик {truck+1}: {best_routes_nn[truck]}")
        ind = np.array(best_routes_nn[truck])
        ind = ind[ind != 0]  # Убираем депо
        truck_loads_nn[truck] = sum(gruz[ind])
        truck_clients_nn[truck] = len(ind)
    
    print("Загрузка грузовиков:", truck_loads_nn)
    print("Клиентов на грузовик:", truck_clients_nn)
    
    # Расчет пробега и расхода топлива для Nearest Neighbor
    truck_dist_nn, truck_fuel_nn = plot_bar_dlina_fuels(trucks['m'], best_routes_nn, dist, trucks['fuel'], txt)
    print("Пробег грузовиков, км:", truck_dist_nn)
    print("Расход топлива, литры:", truck_fuel_nn)
    
    # Расчет загрузки и клиентов для Nearest Neighbor
    truck_loads_nn, truck_clients_nn = plot_bar_client_gruz(trucks['m'], best_routes_nn, gruz, txt)
    print("Загрузка грузовиков:", truck_loads_nn)
    print("Клиентов на грузовик:", truck_clients_nn)
