import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from dataclasses import dataclass
from typing import List, Optional

# Класс для грузовика
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
    x: float  # Координата x
    y: float  # Координата y
    weight: float = 0.0  # Вес груза (кг)
    priority: int = 1  # Приоритет (1 - низкий, 2 - средний, 3 - высокий)

# Класс для груза клиента
@dataclass
class Cargo:
    id: int  # Идентификатор клиента (1..n)
    weight: float  # Вес груза (кг)
    priority: int  # Приоритет (1 - низкий, 2 - средний, 3 - высокий)

# Класс для решения
@dataclass
class Solution:
    routes: List[List[int]]
    cost: float
    loads: List[float]

# Класс визуализации
class Plotter:
    def __init__(self, clients: List[Client]):
        self.clients = clients

    def plot_routes(self, solution: Solution, trucks: List[Truck]):
        depot = next(c for c in self.clients if c.id == 0)
        plt.plot(depot.x, depot.y, 'ks', linewidth=2, markersize=14, markerfacecolor='r', label='Гараж')
        plt.axis([min(c.x for c in self.clients)-1, max(c.x for c in self.clients)+1,
                  min(c.y for c in self.clients)-1, max(c.y for c in self.clients)+1])
        plt.ylabel('Расстояния, км')
        plt.xlabel('Расстояния, км')
        plt.grid(True)

        truck_names = [truck.name for truck in trucks]
        for truck_id in range(len(trucks)):
            ind = solution.routes[truck_id]
            coords_x = [self.clients[i].x for i in ind]
            coords_y = [self.clients[i].y for i in ind]
            plt.plot(coords_x, coords_y, 'o-', linewidth=2, markersize=8, label=truck_names[truck_id])

        for client in self.clients:
            if client.id == 0:
                continue
            if client.priority == 3:
                plt.scatter(client.x, client.y, c='green', marker='^', s=100, label='Высокий приоритет (3)' if client.id == 1 else "")
            elif client.priority == 2:
                plt.scatter(client.x, client.y, c='red', marker='^', s=100, label='Средний приоритет (2)' if client.id == 2 else "")
            else:
                plt.scatter(client.x, client.y, c='blue', marker='o', s=50, label='Низкий приоритет (1)' if client.id == 1 else "")
            plt.text(client.x + 0.5, client.y + 0.5, str(client.id), verticalalignment='bottom', horizontalalignment='right')

        total_weight = sum(c.weight for c in self.clients if c.id != 0)
        title_text = f'Груз клиентам: {total_weight/len(self.clients):.1f} ± {np.std([c.weight for c in self.clients if c.id != 0]):.1f} кг   Затраты топлива: {solution.cost:.1f} литров'
        plt.title(title_text)
        plt.legend(loc='best')
        plt.show()

    def plot_bar_dlina_fuels(self, solution: Solution, dist_matrix: np.ndarray, trucks: List[Truck]):
        truck_dist = np.zeros(len(trucks))
        truck_dist_no1 = np.zeros(len(trucks))
        for truck_id in range(len(trucks)):
            ind = solution.routes[truck_id]
            truck_dist_no1[truck_id] = sum(dist_matrix[ind[i]][ind[i+1]] for i in range(len(ind)-2)) if len(ind) > 2 else 0
            truck_dist[truck_id] = sum(dist_matrix[ind[i]][ind[i+1]] for i in range(len(ind)-1)) if len(ind) > 1 else 0
        truck_fuel = truck_dist * np.array([truck.fuel_per_km for truck in trucks])

        plt.figure()
        plt.subplot(2, 1, 1)
        x = np.arange(len(trucks))
        width = 0.35
        plt.bar(x - width/2, truck_dist_no1, width, color='y', label='Развоз')
        plt.bar(x + width/2, truck_dist, width, color='b', label='В гараж')
        plt.title(f'Пробег грузовиков: {round(sum(truck_dist))} км')
        plt.ylabel('Пробег (км)')
        plt.grid(True)
        plt.xticks(x, [truck.name for truck in trucks])
        plt.legend(loc='best')
        for i in range(len(trucks)):
            plt.text(i, max(truck_dist_no1[i], truck_dist[i]) + 1, f'{round(truck_dist[i])}', ha='center')

        plt.subplot(2, 1, 2)
        plt.bar(x, truck_fuel, color='g')
        plt.title(f'Расход топлива грузовиками: {round(sum(truck_fuel))} литр')
        plt.ylabel('Расход (литр)')
        plt.grid(True)
        plt.xticks(x, [truck.name for truck in trucks])
        for i in range(len(trucks)):
            plt.text(i, truck_fuel[i] + 0.5, f'{round(truck_fuel[i])}', ha='center')
        plt.tight_layout()
        plt.show()

    def plot_loads(self, trucks: List[Truck], loads: List[float]):
        plt.figure()
        x = np.arange(len(trucks))
        plt.bar(x, loads, color='orange')
        plt.title(f'Загрузка грузовиков: {round(sum(loads))} кг')
        plt.ylabel('Загрузка (кг)')
        plt.grid(True)
        plt.xticks(x, [truck.name for truck in trucks])
        for i in range(len(trucks)):
            plt.text(i, loads[i] + 50, f'{round(loads[i])} кг', ha='center')
        plt.tight_layout()
        plt.show()

class ACO:
    def __init__(self, n_ants: int, alpha: float, beta: float, gamma: float, rho: float, n_iterations: int):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.n_iterations = n_iterations

    def calc_distance_matrix(self, clients: List[Client]) -> np.ndarray:
        n_points = len(clients)
        dist = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                dist[i][j] = sqrt((clients[i].x - clients[j].x)**2 + (clients[i].y - clients[j].y)**2)
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

        # Подсчитываем количество точек с приоритетами 3 и 2 в начале маршрута
        protected_count = 0
        for i in range(1, len(route)-1):
            client = next(c for c in clients if c.id == route[i])
            if client.priority in [3, 2]:
                protected_count += 1
            else:
                break

        while improvement:
            improvement = False
            # Оптимизируем только после защищённых точек
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
            return 0  # Возвращение в депо
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

            # Этап 1: Назначаем точки с приоритетом 3
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

            # Этап 2: Назначаем точки с приоритетом 2
            for point in priority_2_points:
                if point in global_visited:
                    continue
                client = next(c for c in clients if c.id == point)
                added = False
                # Сначала пробуем грузовики с точками приоритета 3
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
                # Если не добавлено в грузовик с приоритетом 3, пробуем другие грузовики
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

            # Этап 3: Назначаем точки с приоритетом 1
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

            # Применяем 2-opt, защищая точки с приоритетами 3 и 2
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


# Главная функция
if __name__ == "__main__":
    # Создание клиентов (включая депо)
    clients = [Client(0, 4.0, 4.0)]  # Депо
    for i in range(1, 51):  # Клиенты 1..50
        x = [18.0, 12.5, 11.0, 4.5, 17.0, 17.5, 7.5, 12.5, 2.5, 14.0, 16.0, 16.5, 11.0, 14.5, 8.5, 16.5, 18.0, 17.0, 12.5, 15.0, 16.0, 14.5, 12.0, 18.5, 12.5, 16.5, 2.0, 11.0, 15.5, 12.5, 17.5, 16.5, 16.0, 2.5, 7.5, 18.0, 4.5, 9.5, 15.5, 16.0, 12.5, 12.5, 13.0, 17.5, 7.5, 7.5, 12.5, 7.5, 18.0, 18.5][i-1]
        y = [1.0, 5.5, 10.0, 3.5, 17.5, 12.5, 2.5, 9.5, 2.5, 14.5, 15.5, 15.0, 10.0, 14.0, 7.5, 15.0, 12.5, 17.5, 5.5, 14.0, 8.5, 10.0, 5.5, 0.5, 7.5, 8.5, 2.0, 11.5, 7.5, 11.0, 17.5, 16.5, 7.5, 2.5, 2.5, 7.0, 4.5, 10.0, 7.0, 8.0, 5.5, 9.5, 11.0, 12.5, 2.5, 2.0, 5.5, 2.5, 17.5, 17.0][i-1]
        weight = np.random.randint(100, 500)  # Случайный вес от 100 до 500 кг
        priority = 1
        if i in [17]:
            priority = 3
        elif i in [2, 3, 4, 24]:
            priority = 2
        clients.append(Client(i, x, y, weight, priority))

    # Создание грузовиков
    trucks = [
        Truck(0, 1500, 0.12, 1000, name="Грузовик 1"),
        Truck(1, 1500, 0.12, 1000, name="Грузовик 2"),
        Truck(2, 5000, 0.16, 1000, name="Грузовик 3"),
        Truck(3, 5000, 0.16, 1000, name="Грузовик 4"),
        Truck(4, 13000, 0.33, 1000, name="Грузовик 5")
    ]

    # Инициализация ACO с усиленным влиянием приоритета
    aco = ACO(n_ants=20, alpha=0.5, beta=2.0, gamma=10.0, rho=0.01, n_iterations=200)
    plotter = Plotter(clients)

    # Проверка весов и расстояний для приоритетных точек
    dist = aco.calc_distance_matrix(clients)
    for client in clients:
        if client.priority in [3, 2]:
            print(f"Клиент {client.id}: вес {client.weight} кг, приоритет {client.priority}, расстояние от депо {dist[0][client.id]:.2f} км")

    # Запуск алгоритма и получение решения
    solution = aco.build_routes(trucks, clients)
    print(f"Финальные маршруты: {solution.routes}")
    print(f"Общая стоимость: {solution.cost:.2f}")

    # Визуализация
    plotter.plot_routes(solution, trucks)
    plotter.plot_bar_dlina_fuels(solution, aco.calc_distance_matrix(clients), trucks)
    plotter.plot_loads(trucks, solution.loads)
