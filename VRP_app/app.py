import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import folium
import polyline
from folium.plugins import AntPath
import requests
import time
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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
        curr_client = 0
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
        route.append(0)
        routes[truck] = route
        total_cost += calc_route_cost(route, dist, trucks['fuel'][truck])
    
    return routes, total_cost, test_cities

def select_next_client(curr_client, pheromones, dist, alpha, beta, gruz, curr_capacity, visited, use_random=False):
    clients = dist.shape[0]
    prob = np.zeros(clients)
    
    if use_random:
        available_clients = [j for j in range(clients) if j != curr_client and gruz[j] <= curr_capacity and not visited[j]]
        return random.choice(available_clients) if available_clients else -1
    
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
                if local_opt:
                    best_routes, best_cost = local_optim(routes, dist, trucks['fuel'])
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
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        distances = np.array(data["distances"]) / 1000
        return distances
    except requests.RequestException as e:
        print(f"Ошибка запроса к OSRM: {e}")
        return None

def get_osrm_route(coordinates, route_indices):
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_vrp', methods=['POST'])
def solve_vrp():
    data = request.get_json()
    
    nodes = data['nodes']
    # Находим индекс депо (груз = 0)
    depot_index = next(i for i, node in enumerate(nodes) if node['cargo'] == 0)
    # Переставляем депо на первое место
    nodes.insert(0, nodes.pop(depot_index))
    
    x = np.array([node['lng'] for node in nodes])
    y = np.array([node['lat'] for node in nodes])
    gruz = np.array([node['cargo'] for node in nodes])
    coordinates = list(zip(y, x))
    clients = len(nodes)
    
    trucks_data = data['trucks']
    trucks = {
        'm': len(trucks_data),
        'v': [truck['capacity'] for truck in trucks_data],
        'fuel': [truck['fuelConsumption'] for truck in trucks_data],
        'max_dist': [truck['maxDistance'] for truck in trucks_data],
        'to_home': [0] * len(trucks_data),
        'visited': None,
        'order': None
    }
    
    aco = data['aco']
    local_opt = True
    truck_names = [truck['name'] for truck in trucks_data]
    
    dist = get_osrm_distance_matrix(coordinates)
    if dist is None:
        return jsonify({'error': 'Не удалось загрузить матрицу расстояний'}), 500
    
    #np.random.seed(80)
    start_time = time.time()
    best_routes, best_cost, end_iter = ant_colony(clients, trucks, gruz, dist, aco, local_opt)
    elapsed_time = time.time() - start_time
    print(f"маршрут: {best_routes}, время: {elapsed_time}")
    
    # Расчёт данных для графиков
    truck_dist = np.zeros(trucks['m'])
    truck_dist_no1 = np.zeros(trucks['m'])
    truck_loads = np.zeros(trucks['m'])
    truck_clients = np.zeros(trucks['m'])

    for truck in range(trucks['m']):
        ind = best_routes[truck]
        # Проверяем, что маршрут не пустой
        if len(ind) <= 2:  # Если маршрут [0, 0], считаем его пустым
            truck_dist[truck] = 0
            truck_dist_no1[truck] = 0
            truck_loads[truck] = 0
            truck_clients[truck] = 0
            continue
        truck_dist_no1[truck] = calc_route_cost(ind[:-1], dist, 1)
        truck_dist[truck] = calc_route_cost(ind, dist, 1)
        # Преобразуем ind в массив numpy и фильтруем
        ind_array = np.array(ind)
        ind_no_depot = ind_array[ind_array != 0]
        truck_loads[truck] = np.sum(gruz[ind_no_depot]) if len(ind_no_depot) > 0 else 0
        truck_clients[truck] = len(ind_no_depot)

    truck_fuel = truck_dist * np.array(trucks['fuel'])
    
    # Генерация маршрутов для карты
    routes_geo = []
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    for truck_idx, route in enumerate(best_routes):
        # Пропускаем маршрут, если он содержит только депо (длина маршрута <= 2)
        if len(route) <= 2:
            continue
        route_coords = get_osrm_route(coordinates, route)
        if route_coords:
            # Формируем routeIndices для текущего маршрута
            route_indices = [{'nodeIndex': node_idx, 'routeIndex': route_idx + 1} for route_idx, node_idx in enumerate(route)]
            routes_geo.append({
                'truck': truck_names[truck_idx],
                'route': route_coords,
                'color': colors[truck_idx % len(colors)],
                'routeIndices': route_indices
            })
    
    response = {
        'routes': best_routes,
        'best_cost': best_cost,
        'total_cargo': float(sum(gruz)),
        'truck_dist': truck_dist.tolist(),
        'truck_dist_no1': truck_dist_no1.tolist(),
        'truck_fuel': truck_fuel.tolist(),
        'truck_loads': truck_loads.tolist(),
        'truck_clients': truck_clients.tolist(),
        'routes_geo': routes_geo,
        'truck_names': truck_names
    }
    
    return jsonify(response)

@app.route('/solve_vrp_test', methods=['POST'])
def solve_vrp_test():
    data = None
    with open('test.json', 'r') as f:  # Замените 'data.json' на путь к вашему файлу
        data = json.load(f)
    
    nodes = data['nodes']
    # Находим индекс депо (груз = 0)
    depot_index = next(i for i, node in enumerate(nodes) if node['cargo'] == 0)
    # Переставляем депо на первое место
    nodes.insert(0, nodes.pop(depot_index))
    
    x = np.array([node['lng'] for node in nodes])
    y = np.array([node['lat'] for node in nodes])
    gruz = np.array([node['cargo'] for node in nodes])
    coordinates = list(zip(y, x))
    clients = len(nodes)
    
    trucks_data = data['trucks']
    trucks = {
        'm': len(trucks_data),
        'v': [truck['capacity'] for truck in trucks_data],
        'fuel': [truck['fuelConsumption'] for truck in trucks_data],
        'max_dist': [truck['maxDistance'] for truck in trucks_data],
        'to_home': [0] * len(trucks_data),
        'visited': None,
        'order': None
    }
    
    aco = data['aco']
    local_opt = True
    truck_names = [truck['name'] for truck in trucks_data]
    
    dist = get_osrm_distance_matrix(coordinates)
    if dist is None:
        return jsonify({'error': 'Не удалось загрузить матрицу расстояний'}), 500
    
    #np.random.seed(80)
    start_time = time.time()
    best_routes, best_cost, end_iter = ant_colony(clients, trucks, gruz, dist, aco, local_opt)
    elapsed_time = time.time() - start_time
    print(f"маршрут: {best_routes}, время: {elapsed_time}")
    
    # Расчёт данных для графиков
    truck_dist = np.zeros(trucks['m'])
    truck_dist_no1 = np.zeros(trucks['m'])
    truck_loads = np.zeros(trucks['m'])
    truck_clients = np.zeros(trucks['m'])

    for truck in range(trucks['m']):
        ind = best_routes[truck]
        # Проверяем, что маршрут не пустой
        if len(ind) <= 2:  # Если маршрут [0, 0], считаем его пустым
            truck_dist[truck] = 0
            truck_dist_no1[truck] = 0
            truck_loads[truck] = 0
            truck_clients[truck] = 0
            continue
        truck_dist_no1[truck] = calc_route_cost(ind[:-1], dist, 1)
        truck_dist[truck] = calc_route_cost(ind, dist, 1)
        # Преобразуем ind в массив numpy и фильтруем
        ind_array = np.array(ind)
        ind_no_depot = ind_array[ind_array != 0]
        truck_loads[truck] = np.sum(gruz[ind_no_depot]) if len(ind_no_depot) > 0 else 0
        truck_clients[truck] = len(ind_no_depot)

    truck_fuel = truck_dist * np.array(trucks['fuel'])
    
    # Генерация маршрутов для карты
    routes_geo = []
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    for truck_idx, route in enumerate(best_routes):
        # Пропускаем маршрут, если он содержит только депо (длина маршрута <= 2)
        if len(route) <= 2:
            continue
        route_coords = get_osrm_route(coordinates, route)
        if route_coords:
            # Формируем routeIndices для текущего маршрута
            route_indices = [{'nodeIndex': node_idx, 'routeIndex': route_idx + 1} for route_idx, node_idx in enumerate(route)]
            routes_geo.append({
                'truck': truck_names[truck_idx],
                'route': route_coords,
                'color': colors[truck_idx % len(colors)],
                'routeIndices': route_indices
            })
    
    response = {
        'routes': best_routes,
        'best_cost': best_cost,
        'total_cargo': float(sum(gruz)),
        'truck_dist': truck_dist.tolist(),
        'truck_dist_no1': truck_dist_no1.tolist(),
        'truck_fuel': truck_fuel.tolist(),
        'truck_loads': truck_loads.tolist(),
        'truck_clients': truck_clients.tolist(),
        'routes_geo': routes_geo,
        'truck_names': truck_names
    }
    
    return jsonify(response)   

if __name__ == '__main__':
    app.run(debug=True)