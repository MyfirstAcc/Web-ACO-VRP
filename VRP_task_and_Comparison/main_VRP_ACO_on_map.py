import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import folium
import polyline  # Для декодирования полилиний OSRM
from folium.plugins import AntPath  # Для анимированных маршрутов (опционально)
import requests
import time
import json

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
    visited = trucks['visited'].copy()
    routes = [None] * trucks['m']
    total_cost = 0
    test_cities = 0
    
    for t_idx in range(trucks['m']):
        truck = trucks['order'][t_idx]
        curr_client = 0  # Start from garage (index 0 in Python)
        curr_capacity = trucks['v'][truck]
        route = [curr_client]
        visited[curr_client] = True
        current_dist = 0
        
        while curr_capacity > 0 and sum(gruz[route]) < sum(gruz):
            next_client = select_next_client(curr_client, pheromones, dist, aco['alpha'], aco['beta'], gruz, curr_capacity, visited)
            if next_client == -1:
                break
            
            dist_to_next = dist[curr_client, next_client]
            if current_dist + dist_to_next + trucks['to_home'][truck] * dist[next_client, 0] > trucks['max_dist'][truck]:
                break
            
            current_dist += dist_to_next
            route.append(next_client)
            curr_capacity -= gruz[next_client]
            curr_client = next_client
            visited[curr_client] = True
        
        test_cities += len(route) - 1
        route.append(0)  # Return to garage
        routes[truck] = route
        total_cost += calc_route_cost(route, dist, trucks['fuel'][truck])
    
    return routes, total_cost, test_cities

def select_next_client(curr_client, pheromones, dist, alpha, beta, gruz, curr_capacity, visited, use_random=False):
    clients = dist.shape[0]
    prob = np.zeros(clients)
    
    if use_random:
        # Random client selection
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

def get_osrm_distance_matrix(coordinates):
    """
    coordinates: список кортежей [(lat1, lon1), (lat2, lon2), ...]
    Возвращает матрицу расстояний в километрах
    """
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        distances = np.array(data["distances"]) / 1000  # Переводим метры в километры
        return distances
    except requests.RequestException as e:
        print(f"Ошибка запроса к OSRM: {e}")
        return None

def get_osrm_route(coordinates, route_indices):
    """
    coordinates: список кортежей [(lat1, lon1), (lat2, lon2), ...]
    route_indices: список индексов точек маршрута, например [0, 1, 2, 0]
    Возвращает список координат маршрута
    """
    coords_str = ";".join([f"{coordinates[i][1]},{coordinates[i][0]}" for i in route_indices])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=polyline"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        route_polyline = data["routes"][0]["geometry"]
        return polyline.decode(route_polyline)
    except requests.RequestException as e:
        print(f"Ошибка запроса к OSRM для маршрута: {e}")
        return None

def plot_routes_on_map(coordinates, best_routes, truck_names):
    """
    coordinates: список кортежей [(lat1, lon1), (lat2, lon2), ...]
    best_routes: список маршрутов, например [[0, 1, 2, 0], ...]
    truck_names: список имён грузовиков
    """
    map_center = [np.mean([lat for lat, lon in coordinates]), np.mean([lon for lat, lon in coordinates])]
    m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Добавляем маркеры для узлов
    for i, (lat, lon) in enumerate(coordinates):
        label = f"Depot" if i == 0 else f"Node {i}"
        folium.Marker(
            [lat, lon],
            popup=label,
            icon=folium.Icon(color="red" if i == 0 else "blue")
        ).add_to(m)
    
    # Добавляем маршруты
    for truck_idx, route in enumerate(best_routes):
        route_coords = get_osrm_route(coordinates, route)
        if route_coords:
            AntPath(
                route_coords,
                color=colors[truck_idx % len(colors)],
                weight=5,
                opacity=0.7,
                popup=truck_names[truck_idx]
            ).add_to(m)
    
    m.save("routes_map.html")
    print("Карта сохранена в routes_map.html")
    return m

# Основной код
if __name__ == "__main__":
    # Загрузка данных из JSON
    with open('data.json', 'r') as f:  # Замените 'data.json' на путь к вашему файлу
        data = json.load(f)
    
    # Извлечение данных узлов
    nodes = data['nodes']
    x = np.array([node['lng'] for node in nodes])  # Долгота
    y = np.array([node['lat'] for node in nodes])  # Широта
    gruz = np.array([node['cargo'] for node in nodes])  # Грузы
    coordinates = list(zip(y, x))  # [(lat, lon), ...]
    clients = len(nodes)
    
    # Извлечение данных грузовиков
    trucks_data = data['trucks']
    trucks = {
        'm': len(trucks_data),
        'v': [truck['capacity'] for truck in trucks_data],
        'fuel': [truck['fuelConsumption'] for truck in trucks_data],
        'max_dist': [truck['maxDistance'] for truck in trucks_data],
        'to_home': [0] * len(trucks_data),  # Предполагаем, что возврат в депо не влияет
        'visited': None,
        'order': None
    }
    
    # Получение матрицы расстояний через OSRM
    dist = get_osrm_distance_matrix(coordinates)
    if dist is None:
        raise ValueError("Не удалось загрузить матрицу расстояний")
    
    # Параметры ACO
    aco = {
        'ants': 20,
        'iters': 10,
        'alpha': 1,
        'beta': 10,
        'rho': 0.05
    }
    
    np.random.seed(80)
    local_opt = True
    
    # Запуск ACO
    start_time = time.time()
    best_routes, best_cost, end_iter = ant_colony(clients, trucks, gruz, dist, aco, local_opt)
    elapsed_time = time.time() - start_time
    
    # Отрисовка маршрутов на карте
    truck_names = ['Газель 1', 'Газель 2', 'Isuzu 1', 'Isuzu 2', 'КамАЗ']
    plot_routes_on_map(coordinates, best_routes, truck_names)
    
    # Отрисовка существующих графиков
    plot_routes(x, y, best_routes, best_cost, gruz, clients, trucks['m'])
    
    # Расчёт пробега и расхода топлива
    truck_dist, truck_fuel = plot_bar_dlina_fuels(trucks['m'], best_routes, dist, trucks['fuel'], truck_names)
    
    # Расчёт загрузки и количества клиентов
    truck_loads, truck_clients = plot_bar_client_gruz(trucks['m'], best_routes, gruz, truck_names)
    
    # Вывод результатов
    print("\n-----------------------")
    print(f"Суммарный перевозимый груз: {sum(gruz):.1f}")
    print(f"Общая потребность топлива: {best_cost:.1f}")
    print("Лучшие маршруты для каждого грузовика:")
    for truck in range(trucks['m']):
        print(best_routes[truck])
    print("Загрузка грузовиков:", truck_loads)
    print("Клиентов на грузовик:", truck_clients)
    print("Пробег грузовиков, км:", truck_dist)
    print("Расход топлива, литры:", truck_fuel)