import numpy as np
import json
import requests
import polyline
import folium
from folium.plugins import AntPath
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Truck:
    id: int
    capacity: float  # Грузоподъёмность (кг)
    fuel_per_km: float  # Расход топлива (литры/км)
    max_distance: float  # Максимальная дальность (км)
    to_home: float = 0.0  # Расстояние до депо
    visited: List[int] = None
    order: Optional[List[int]] = None
    name: str = "Unnamed"  # Имя грузовика

    def __post_init__(self):
        if self.visited is None:
            self.visited = []
        if self.order is None:
            self.order = []

@dataclass
class Client:
    id: int  # Идентификатор клиента (0 для депо, 1..n для клиентов)
    x: float  # Координата x (долгота, lng)
    y: float  # Координата y (широта, lat)
    weight: float = 0.0  # Вес груза (кг)
    priority: int = 1  # Приоритет (1 - низкий, 2 - средний, 3 - высокий)
    label: str = ""  # Метка клиента

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float
    loads: List[float]

class ACO:
    def __init__(self, n_ants: int, alpha: float, beta: float, gamma: float, rho: float, n_iterations: int):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.n_iterations = n_iterations

    def get_osrm_distance_matrix(self, coordinates: List[tuple[float, float]]) -> np.ndarray:
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
        url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            distances = np.array(data["distances"]) / 1000  # Конвертация метров в километры
            return distances
        except requests.RequestException as e:
            print(f"Ошибка запроса к OSRM: {e}")
            return None

    def calc_distance_matrix(self, clients: List[Client]) -> np.ndarray:
        coordinates = [(client.y, client.x) for client in clients]  # (lat, lng)
        dist = self.get_osrm_distance_matrix(coordinates)
        if dist is None:
            # Запасной вариант: евклидово расстояние
            print("Используется евклидово расстояние из-за ошибки OSRM")
            n_points = len(clients)
            dist = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(n_points):
                    dist[i][j] = np.sqrt((clients[i].x - clients[j].x)**2 + (clients[i].y - clients[j].y)**2)
        return dist

    def calc_route_cost(self, route: List[int], dist: np.ndarray, scale: float = 1) -> float:
        total_dist = 0
        for i in range(len(route)-1):
            total_dist += dist[route[i]][route[i+1]] * scale
        return total_dist

    def two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        return route[:i] + route[i:j+1][::-1] + route[j+1:]

    def two_opt(self, route: List[int], dist: np.ndarray, clients: List[Client]) -> tuple[List[int], float]:
        best_route = route.copy()
        best_distance = self.calc_route_cost(route, dist, 1)
        improvement = True

        # Защищаем точки с приоритетами 3 и 2
        protected_count = 0
        for i in range(1, len(route)-1):
            client = next(c for c in clients if c.id == route[i])
            if client.priority in [3, 2]:
                protected_count += 1
            else:
                break

        while improvement:
            improvement = False
            for i in range(1 + protected_count, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = self.two_opt_swap(best_route, i, j)
                    new_distance = self.calc_route_cost(new_route, dist, 1)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improvement = True
        return best_route, best_distance

    def local_optim(self, routes: List[List[int]], dist: np.ndarray, fuel_per_km: List[float], clients: List[Client]) -> tuple[List[List[int]], float]:
        new_routes = [None] * len(routes)
        new_dlina = 0
        for i in range(len(routes)):
            route, dlina = self.two_opt(routes[i], dist, clients)
            new_routes[i] = route
            new_dlina += dlina * fuel_per_km[i]
        return new_routes, new_dlina

    def choose_next(self, ant: List[int], visited: set, tau: np.ndarray, dist: np.ndarray, clients: List[Client], truck: Truck, current_load: float, current_distance: float) -> int:
        unvisited = [c.id for c in clients if c.id not in visited and c.id != 0]
        if not unvisited:
            return 0
        probs = []
        for j in unvisited:
            client = next(c for c in clients if c.id == j)
            if current_load + client.weight > truck.capacity or current_distance + dist[ant[-1]][j] > truck.max_distance:
                probs.append(0)
                continue
            tau_ij = tau[ant[-1]][j] ** self.alpha
            eta_ij = (1.0 / dist[ant[-1]][j]) ** self.beta if dist[ant[-1]][j] > 0 else 0
            prio_j = client.priority ** self.gamma
            probs.append(tau_ij * eta_ij * prio_j)
        if sum(probs) == 0:
            return 0
        probs = np.array(probs) / np.sum(probs)
        next_point = unvisited[np.random.choice(len(unvisited), p=probs)]
        return next_point

    def build_routes(self, trucks: List[Truck], clients: List[Client]) -> Solution:
        dist = self.calc_distance_matrix(clients)
        tau = np.ones((len(clients), len(clients))) * (1.0 / (len(clients) * np.mean(dist)))
        best_solution = Solution([[] for _ in trucks], float('inf'), [0] * len(trucks))

        priority_3_points = [client.id for client in clients if client.priority == 3 and client.id != 0]
        priority_2_points = [client.id for client in clients if client.priority == 2 and client.id != 0]
        priority_1_points = [client.id for client in clients if client.priority == 1 and client.id != 0]
        print(f"Точки с приоритетом 3 (высокий): {priority_3_points}")
        print(f"Точки с приоритетом 2 (средний): {priority_2_points}")
        print(f"Точки с приоритетом 1 (низкий): {priority_1_points}")

        for iteration in range(self.n_iterations):
            routes = [[0] for _ in trucks]
            loads = [0] * len(trucks)
            distances = [0] * len(trucks)
            global_visited = {0}

            # Этап 1: Точки с приоритетом 3
            for point in priority_3_points:
                if point in global_visited:
                    continue
                client = next(c for c in clients if c.id == point)
                added = False
                for truck in trucks:
                    route = routes[truck.id]
                    load = loads[truck.id]
                    distance = distances[truck.id]
                    if load + client.weight <= truck.capacity and distance + dist[route[-1]][point] <= truck.max_distance:
                        route.append(point)
                        loads[truck.id] += client.weight
                        distances[truck.id] += dist[route[-1]][point]
                        global_visited.add(point)
                        added = True
                        # print(f"Грузовик {truck.name}, точка {point} (приоритет 3), нагрузка: {loads[truck.id]} кг (макс: {truck.capacity} кг), расстояние: {distances[truck.id]:.2f} км (макс: {truck.max_distance} км)")
                        break
                # if not added:
                #     print(f"Ошибка: Точка {point} (приоритет 3) не добавлена: превышены ограничения")

            # Этап 2: Точки с приоритетом 2
            for point in priority_2_points:
                if point in global_visited:
                    continue
                client = next(c for c in clients if c.id == point)
                added = False
                for truck in trucks:
                    route = routes[truck.id]
                    load = loads[truck.id]
                    distance = distances[truck.id]
                    if any(client.id in priority_3_points for client in clients if client.id in route[1:]):
                        if load + client.weight <= truck.capacity and distance + dist[route[-1]][point] <= truck.max_distance:
                            route.append(point)
                            loads[truck.id] += client.weight
                            distances[truck.id] += dist[route[-1]][point]
                            global_visited.add(point)
                            added = True
                            # print(f"Грузовик {truck.name}, точка {point} (приоритет 2), нагрузка: {loads[truck.id]} кг (макс: {truck.capacity} кг), расстояние: {distances[truck.id]:.2f} км (макс: {truck.max_distance} км)")
                            break
                if added:
                    continue
                for truck in trucks:
                    route = routes[truck.id]
                    load = loads[truck.id]
                    distance = distances[truck.id]
                    if load + client.weight <= truck.capacity and distance + dist[route[-1]][point] <= truck.max_distance:
                        route.append(point)
                        loads[truck.id] += client.weight
                        distances[truck.id] += dist[route[-1]][point]
                        global_visited.add(point)
                        added = True
                        # print(f"Грузовик {truck.name}, точка {point} (приоритет 2), нагрузка: {loads[truck.id]} кг (макс: {truck.capacity} кг), расстояние: {distances[truck.id]:.2f} км (макс: {truck.max_distance} км)")
                        break
                # if not added:
                #     print(f"Ошибка: Точка {point} (приоритет 2) не добавлена: превышены ограничения")

            # Этап 3: Точки с приоритетом 1
            for truck in trucks:
                route = routes[truck.id]
                visited = set(route)
                load = loads[truck.id]
                distance = distances[truck.id]
                available_points = [c.id for c in clients if c.id not in global_visited and c.id != 0]
                while available_points and load < truck.capacity and distance < truck.max_distance:
                    next_point = self.choose_next(route, global_visited, tau, dist, clients, truck, load, distance)
                    if next_point == 0 or next_point not in available_points:
                        route.append(0)
                        break
                    client = next(c for c in clients if c.id == next_point)
                    load += client.weight
                    distance += dist[route[-1]][next_point]
                    route.append(next_point)
                    visited.add(next_point)
                    global_visited.add(next_point)
                    available_points = [c.id for c in clients if c.id not in global_visited and c.id != 0]
                    # print(f"Грузовик {truck.name}, точка {next_point} (приоритет {client.priority}), нагрузка: {load} кг (макс: {truck.capacity} кг), расстояние: {distance:.2f} км (макс: {truck.max_distance} км)")
                if 0 not in route[-1:]:
                    route.append(0)
                routes[truck.id] = route
                loads[truck.id] = load
                distances[truck.id] = distance

            fuel_per_km = [truck.fuel_per_km for truck in trucks]
            routes, total_cost = self.local_optim(routes, dist, fuel_per_km, clients)

            if total_cost < best_solution.cost:
                best_solution.cost = total_cost
                best_solution.routes = [route[:] for route in routes]
                best_solution.loads = loads[:]

            tau *= (1 - self.rho)
            for truck in trucks:
                for i in range(len(best_solution.routes[truck.id])-1):
                    tau[best_solution.routes[truck.id][i]][best_solution.routes[truck.id][i+1]] += 1.0 / (best_solution.cost + 1e-10)

            # print(f"Итерация {iteration}, лучшие маршруты: {best_solution.routes}, стоимость: {best_solution.cost:.2f}")

        return best_solution

def get_osrm_route(coordinates: List[tuple[float, float]], route_indices: List[int]) -> Optional[List[tuple[float, float]]]:
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

def plot_routes_on_map(coordinates: List[tuple[float, float]], best_routes: List[List[int]], truck_names: List[str], clients: List[Client]):
    map_center = [np.mean([lat for lat, lon in coordinates]), np.mean([lon for lat, lon in coordinates])]
    m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")
    colors = ['blue', 'green', 'red', 'purple', 'orange','cyan']

    # Добавляем маркеры для узлов
    for i, (lat, lon) in enumerate(coordinates):
        client = next(c for c in clients if c.id == i)
        label = client.label
        priority = client.priority
        color = "red" if i == 0 else "green" if priority == 3 else "orange" if priority == 2 else "blue"
        folium.Marker(
            [lat, lon],
            popup=f"{label} (Приоритет {priority})",
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Добавляем маршруты
    for truck_idx, route in enumerate(best_routes):
        if len(route) < 2:
            continue
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

if __name__ == "__main__":
    # Чтение данных из test.json
    with open("test.json", "r") as f:
        data = json.load(f)
    
    clients = []
    for i, node in enumerate(data["nodes"]):
        # Предполагаем, что приоритеты задаются вручную, если отсутствуют в JSON
        priority = 1
        if i == 5:  # Например, Node 5 имеет приоритет 3 (быстрая доставка)
            priority = 3
        elif i in [2, 3]:  # Node 3 и Node 4 имеют приоритет 2
            priority = 2
        clients.append(Client(
            id=i,
            x=node["lng"],  # Долгота
            y=node["lat"],  # Широта
            weight=node["cargo"],
            priority=priority,
            label=node["label"]
        ))

    # Создание грузовиков
    trucks = [
        Truck(0, 3500, 0.12, 1000, name="Грузовик 1"),
        Truck(1, 2500, 0.12, 2000, name="Грузовик 2"),
        Truck(2, 5000, 0.16, 2000, name="Грузовик 3"),
        Truck(3, 7000, 0.24, 2000, name="Грузовик 4"),
        Truck(4, 7000, 0.24, 3000, name="Грузовик 5"),
        Truck(5, 7000, 0.24, 3000, name="Грузовик 6")

    ]

    # Инициализация ACO
    aco = ACO(n_ants=20, alpha=0.5, beta=2.0, gamma=10.0, rho=0.01, n_iterations=100)

    # Проверка весов и расстояний
    dist = aco.calc_distance_matrix(clients)
    for client in clients:
        if client.priority in [3, 2] or client.id == 0:
            print(f"Клиент {client.label} (id={client.id}): вес {client.weight} кг, приоритет {client.priority}, расстояние от депо {dist[0][client.id]:.2f} км")

    # Запуск алгоритма
    solution = aco.build_routes(trucks, clients)
    print(f"Финальные маршруты: {solution.routes}")
    print(f"Общая стоимость: {solution.cost:.2f}")

    # Визуализация на карте
    coordinates = [(client.y, client.x) for client in clients]  # (lat, lng)
    truck_names = [truck.name for truck in trucks]
    plot_routes_on_map(coordinates, solution.routes, truck_names, clients)