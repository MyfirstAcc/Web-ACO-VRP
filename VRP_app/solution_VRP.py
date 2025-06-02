from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import polyline
import time
import numpy as np
import requests

@dataclass
class Client:
    """Represents a client with location and cargo demand."""
    lat: float
    lng: float
    cargo: float
    index: int

    def to_dict(self) -> dict:
        return {'lat': self.lat, 'lng': self.lng, 'cargo': self.cargo}

@dataclass
class Truck:
    """Represents a truck with its properties."""
    name: str
    capacity: float
    fuel_consumption: float
    max_distance: float
    index: int

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'capacity': self.capacity,
            'fuelConsumption': self.fuel_consumption,
            'maxDistance': self.max_distance
        }

@dataclass
class ACOConfig:
    """Configuration for the Ant Colony Optimization algorithm."""
    iters: int
    ants: int
    alpha: float
    beta: float
    rho: float

class VRPResult:
    """Stores the results of the VRP solution."""
    def __init__(self, routes: List[List[int]], best_cost: float, end_iter: int, 
                 truck_dist: List[float], truck_dist_no1: List[float], 
                 truck_fuel: List[float], truck_loads: List[float], 
                 truck_clients: List[float], routes_geo: List[dict], 
                 truck_names: List[str], total_cargo: float):
        self.routes = routes
        self.best_cost = best_cost
        self.end_iter = end_iter
        self.truck_dist = truck_dist
        self.truck_dist_no1 = truck_dist_no1
        self.truck_fuel = truck_fuel
        self.truck_loads = truck_loads
        self.truck_clients = truck_clients
        self.routes_geo = routes_geo
        self.truck_names = truck_names
        self.total_cargo = total_cargo

    def to_dict(self) -> dict:
        return {
            'routes': self.routes,
            'best_cost': self.best_cost,
            'total_cargo': self.total_cargo,
            'truck_dist': self.truck_dist,
            'truck_dist_no1': self.truck_dist_no1,
            'truck_fuel': self.truck_fuel,
            'truck_loads': self.truck_loads,
            'truck_clients': self.truck_clients,
            'routes_geo': self.routes_geo,
            'truck_names': self.truck_names
        }

class VRP:
    """Main class for solving the Vehicle Routing Problem using ACO."""
    def __init__(self, clients: List[Client], trucks: List[Truck], aco_config: ACOConfig):
        self.clients = clients
        self.trucks = trucks
        self.aco_config = aco_config
        self.dist_matrix = self._get_osrm_distance_matrix()
        self.pheromones = np.ones((len(clients), len(clients)))

    def _get_osrm_distance_matrix(self) -> Optional[np.ndarray]:
        """Fetches distance matrix from OSRM."""
        coordinates = [(client.lat, client.lng) for client in self.clients]
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
        url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return np.array(data["distances"]) / 1000
        except requests.RequestException as e:
            print(f"Ошибка запроса к OSRM: {e}")
            return None

    def _get_osrm_route(self, route_indices: List[int]) -> Optional[List[Tuple[float, float]]]:
        """Fetches route geometry from OSRM."""
        coordinates = [(self.clients[i].lat, self.clients[i].lng) for i in route_indices]
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=polyline"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return polyline.decode(data["routes"][0]["geometry"])
        except requests.RequestException as e:
            print(f"Ошибка запроса к OSRM для маршрута: {e}")
            return None

    def _calc_route_cost(self, route: List[int], fuel_per_km: float) -> float:
        """Calculates the cost of a route."""
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist_matrix[route[i], route[i + 1]]
        return cost * fuel_per_km

    def _two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        """Performs a 2-opt swap on the route."""
        return route[:i] + route[i:j + 1][::-1] + route[j + 1:]

    def _two_opt(self, route: List[int]) -> Tuple[List[int], float]:
        """Optimizes a route using 2-opt algorithm."""
        best_route = route.copy()
        best_distance = self._calc_route_cost(route, 1)
        improvement = True
        
        while improvement:
            improvement = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = self._two_opt_swap(best_route, i, j)
                    new_distance = self._calc_route_cost(new_route, 1)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improvement = True
        return best_route, best_distance

    def _local_optim(self, routes: List[List[int]]) -> Tuple[List[List[int]], float]:
        """Applies local optimization (2-opt) to all routes."""
        new_routes = [None] * len(self.trucks)
        new_cost = 0
        
        for i in range(len(self.trucks)):
            route, dlina = self._two_opt(routes[i])
            new_routes[i] = route
            new_cost += dlina * self.trucks[i].fuel_consumption
        
        return new_routes, new_cost

    def _select_next_client(self, curr_client: int, curr_capacity: float, 
                           visited: np.ndarray, use_random: bool = False) -> int:
        """Selects the next client based on pheromone trails or randomly."""
        clients_count = len(self.clients)
        prob = np.zeros(clients_count)
        
        if use_random:
            available = [j for j in range(clients_count) 
                        if j != curr_client and self.clients[j].cargo <= curr_capacity and not visited[j]]
            return np.random.choice(available) if available else -1
        
        for j in range(clients_count):
            if j != curr_client and self.clients[j].cargo <= curr_capacity and not visited[j]:
                prob[j] = (self.pheromones[curr_client, j] ** self.aco_config.alpha) * \
                         (1 / self.dist_matrix[curr_client, j]) ** self.aco_config.beta
        
        if sum(prob) == 0:
            return -1
        
        prob /= sum(prob)
        return np.random.choice(clients_count, p=prob)

    def _ant_one(self) -> Tuple[List[List[int]], float, int]:
        """Simulates one ant's solution construction."""
        visited = np.zeros(len(self.clients), dtype=bool)
        routes = [None] * len(self.trucks)
        total_cost = 0
        test_cities = 0
        
        truck_order = np.random.permutation(len(self.trucks))
        
        for t_idx in truck_order:
            truck = self.trucks[t_idx]
            curr_client = 0
            curr_capacity = truck.capacity
            route = [curr_client]
            visited[curr_client] = True
            current_dist = 0
            
            while curr_capacity > 0 and sum(self.clients[i].cargo for i in route) < sum(c.cargo for c in self.clients):
                next_client = self._select_next_client(curr_client, curr_capacity, visited)
                if next_client == -1:
                    break
                
                dist_to_next = self.dist_matrix[curr_client, next_client]
                if current_dist + dist_to_next + self.dist_matrix[next_client, 0] > truck.max_distance:
                    break
                
                current_dist += dist_to_next
                route.append(next_client)
                curr_capacity -= self.clients[next_client].cargo
                curr_client = next_client
                visited[curr_client] = True
            
            test_cities += len(route) - 1
            route.append(0)
            routes[t_idx] = route
            total_cost += self._calc_route_cost(route, truck.fuel_consumption)
        
        return routes, total_cost, test_cities

    def solve(self, local_opt: bool = True) -> VRPResult:
        """Solves the VRP using ACO."""
        if self.dist_matrix is None:
            raise ValueError("Distance matrix not available")
        
        best_cost = float('inf')
        best_routes = []
        end_iter = 0
        start_time = time.time()
        
        for iter in range(self.aco_config.iters):
            for _ in range(self.aco_config.ants):
                np.random.seed(np.random.randint(1, 10000))
                routes, total_cost, test_cities = self._ant_one()
                
                if total_cost < best_cost and test_cities == len(self.clients) - 1:
                    if local_opt:
                        best_routes, best_cost = self._local_optim(routes)
                    else:
                        best_cost = total_cost
                        best_routes = routes
                    end_iter = iter + 1
                
                for t in range(len(routes)):
                    for j in range(len(routes[t]) - 1):
                        self.pheromones[routes[t][j], routes[t][j + 1]] += 1 / total_cost
            self.pheromones *= (1 - self.aco_config.rho)
        
        # Calculate metrics
        truck_dist = np.zeros(len(self.trucks))
        truck_dist_no1 = np.zeros(len(self.trucks))
        truck_loads = np.zeros(len(self.trucks))
        truck_clients = np.zeros(len(self.trucks))
        
        for truck_idx, route in enumerate(best_routes):
            if len(route) <= 2:
                continue
            truck_dist_no1[truck_idx] = self._calc_route_cost(route[:-1], 1)
            truck_dist[truck_idx] = self._calc_route_cost(route, 1)
            ind_array = np.array(route)
            ind_no_depot = ind_array[ind_array != 0]
            truck_loads[truck_idx] = np.sum([self.clients[i].cargo for i in ind_no_depot]) if len(ind_no_depot) > 0 else 0
            truck_clients[truck_idx] = len(ind_no_depot)
        
        truck_fuel = truck_dist * np.array([truck.fuel_consumption for truck in self.trucks])
        
        # Generate map routes
        routes_geo = []
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        for truck_idx, route in enumerate(best_routes):
            if len(route) <= 2:
                continue
            route_coords = self._get_osrm_route(route)
            if route_coords:
                route_indices = [{'nodeIndex': node_idx, 'routeIndex': route_idx + 1} 
                               for route_idx, node_idx in enumerate(route)]
                routes_geo.append({
                    'truck': self.trucks[truck_idx].name,
                    'route': route_coords,
                    'color': colors[truck_idx % len(colors)],
                    'routeIndices': route_indices
                })
        
        elapsed_time = time.time() - start_time
        print(f"маршрут: {best_routes}, время: {elapsed_time}")
        
        return VRPResult(
            routes=best_routes,
            best_cost=best_cost,
            end_iter=end_iter,
            truck_dist=truck_dist.tolist(),
            truck_dist_no1=truck_dist_no1.tolist(),
            truck_fuel=truck_fuel.tolist(),
            truck_loads=truck_loads.tolist(),
            truck_clients=truck_clients.tolist(),
            routes_geo=routes_geo,
            truck_names=[truck.name for truck in self.trucks],
            total_cargo=float(sum(client.cargo for client in self.clients))
        )
