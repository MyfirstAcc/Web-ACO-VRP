import math
import random
import time
import numpy as np
from typing import List, Tuple
import json
import uuid

class Node:
    def __init__(self, lat: float, lng: float, label: str, cargo: int):
        self.lat = lat
        self.lng = lng
        self.label = label
        self.cargo = cargo

class Truck:
    def __init__(self, capacity: int, fuel_consumption: float, max_distance: float):
        self.capacity = capacity
        self.fuel_consumption = fuel_consumption
        self.max_distance = max_distance

class InputData:
    def __init__(self, x_coords: list, y_coords: list, distances: np.ndarray, cargo_list: list, max_iterations: int):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.distances = distances
        self.cargo_list = cargo_list
        self.max_iterations = max_iterations
        self.max_ants = 50
        self.alpha = 1.0
        self.beta = 2.0
        self.evaporation_rate = 0.5
        self.flag_two_opt = True

class AlgorithmResult:
    def __init__(self, best_routes: List[List[int]], best_cost: float, truck_loads: List[int],
                 truck_clients: List[int], truck_distances: List[float], truck_fuel: List[float],
                 algorithm_time: float, cargo: List[int], labels: List[str]):
        self.best_routes = best_routes
        self.best_cost = best_cost
        self.truck_loads = truck_loads
        self.truck_clients = truck_clients
        self.truck_distances = truck_distances
        self.truck_fuel = truck_fuel
        self.algorithm_time = algorithm_time
        self.cargo = cargo
        self.labels = labels

    def to_dict(self):
        return {
            'best_routes': self.best_routes,
            'best_cost': self.best_cost,
            'truck_loads': self.truck_loads,
            'truck_clients': self.truck_clients,
            'truck_distances': [float(d) for d in self.truck_distances],
            'truck_fuel': [float(f) for f in self.truck_fuel],
            'algorithm_time': self.algorithm_time,
            'cargo': self.cargo,
            'labels': self.labels
        }

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Haversine formula for distance between two points
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def generate_leaflet_html(nodes: List[Node], aco_result: AlgorithmResult, nn_result: List[List[int]]):
    # Generate colors for different routes
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'black']
    
    # Prepare route coordinates for both algorithms
    aco_routes = []
    nn_routes = []
    
    for route_idx, route in enumerate(aco_result.best_routes):
        route_coords = [[nodes[i].lat, nodes[i].lng] for i in route]
        aco_routes.append({'coords': route_coords, 'color': colors[route_idx % len(colors)]})
    
    for route_idx, route in enumerate(nn_result):
        route_coords = [[nodes[i].lat, nodes[i].lng] for i in route]
        nn_routes.append({'coords': route_coords, 'color': colors[route_idx % len(colors)]})

    # HTML template with Leaflet
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VRP Routes</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 600px; }}
            .info {{ padding: 6px 8px; font: 14px/16px Arial, sans-serif; background: white; 
                    background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); 
                    border-radius: 5px; }}
            .legend {{ line-height: 18px; color: #555; }}
            .legend i {{ width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([{nodes[0].lat}, {nodes[0].lng}], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            }}).addTo(map);

            // Add nodes
            var nodes = {json.dumps([[n.lat, n.lng, n.label, n.cargo] for n in nodes])};
            nodes.forEach(function(node) {{
                L.circleMarker([node[0], node[1]], {{radius: 5, color: 'black', fillColor: 'black', fillOpacity: 1}})
                    .addTo(map)
                    .bindPopup('Label: ' + node[2] + '<br>Cargo: ' + node[3]);
            }});

            // ACO Routes
            var acoRoutes = {json.dumps(aco_routes)};
            acoRoutes.forEach(function(route) {{
                L.polyline(route.coords, {{color: route.color, weight: 4, opacity: 0.6}})
                    .addTo(map)
                    .bindPopup('ACO Route');
            }});

            // Nearest Neighbor Routes
            var nnRoutes = {json.dumps(nn_routes)};
            nnRoutes.forEach(function(route) {{
                L.polyline(route.coords, {{color: route.color, weight: 4, opacity: 0.6, dashArray: '5, 10'}})
                    .addTo(map)
                    .bindPopup('NN Route');
            }});

            // Legend
            var legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function(map) {{
                var div = L.DomUtil.create('div', 'info legend');
                div.innerHTML += '<h4>Legend</h4>';
                div.innerHTML += '<i style="background: blue"></i> Route 1<br>';
                div.innerHTML += '<i style="background: red"></i> Route 2<br>';
                div.innerHTML += '<i style="background: green"></i> Route 3<br>';
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    
    with open('vrp_routes.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Leaflet map generated as 'vrp_routes.html'")

class ACO:
    def __init__(self, trucks: List[Truck], input_data: InputData):
        self.trucks = trucks
        self.input_data = input_data
        self.clients = len(input_data.x_coords)
        self.pheromones = np.ones((self.clients, self.clients))

    def run(self, threads: int = 4) -> AlgorithmResult:
        start_time = time.time()
        best_routes, best_cost, _ = self._fn_ant_colony(
            self.clients, self.trucks, self.input_data.cargo_list,
            self.input_data.distances, self.input_data.max_ants,
            self.input_data.max_iterations, self.input_data.alpha,
            self.input_data.beta, self.input_data.evaporation_rate,
            self.input_data.flag_two_opt, threads
        )

        if not best_routes:
            raise Exception("ACO did not find a solution covering all clients.")

        algorithm_time = time.time() - start_time

        # Calculate truck loads and clients
        truck_loads = []
        truck_clients = []
        for route in best_routes:
            route_clients = [i for i in route if i != 0]
            truck_loads.append(sum(self.input_data.cargo_list[i] for i in route_clients))
            truck_clients.append(len(route_clients))

        # Calculate distances and fuel
        truck_distances = []
        truck_fuel = []
        for idx, route in enumerate(best_routes):
            distance = self._calc_route_cost(route, self.input_data.distances, 1)
            truck_distances.append(distance)
            truck_fuel.append(distance * self.trucks[idx].fuel_consumption)

        return AlgorithmResult(
            best_routes, best_cost, truck_loads, truck_clients,
            truck_distances, truck_fuel, algorithm_time,
            self.input_data.cargo_list, self.input_data.labels
        )

    def _fn_ant_colony(self, clients: int, trucks: List[Truck], cargo: List[int],
                      distances: np.ndarray, ants: int, iters: int, alpha: float,
                      beta: float, rho: float, local_opt: bool, threads: int) -> Tuple[List[List[int]], float, int]:
        best_cost = float('inf')
        best_routes = None
        end_iter = 0

        for iter in range(iters):
            local_pheromones = np.zeros((clients, clients))
            local_best_routes = None
            local_best_cost = float('inf')
            local_best_test_cities = 0

            for ant in range(ants):
                visited = [False] * clients
                visited[0] = True
                order = list(range(len(trucks)))
                random.shuffle(order)

                routes, total_cost, test_cities = self._fn_ant_one(
                    trucks, cargo, distances, alpha, beta, random.Random(9056)
                )

                if total_cost < local_best_cost and test_cities == clients - 1:
                    local_best_routes = [r[:] for r in routes]
                    local_best_cost = total_cost
                    local_best_test_cities = test_cities

                for route in routes:
                    for j in range(len(route) - 1):
                        local_pheromones[route[j], route[j + 1]] += 1 / total_cost

            if local_best_cost < best_cost and local_best_test_cities == clients - 1:
                if local_opt:
                    opt_routes, opt_cost = self._fn_local_optim(local_best_routes, distances, [t.fuel_consumption for t in trucks])
                    best_cost = opt_cost
                    best_routes = opt_routes
                else:
                    best_cost = local_best_cost
                    best_routes = local_best_routes
                end_iter = iter

            self.pheromones = self.pheromones * (1 - rho) + local_pheromones

        return best_routes, best_cost, end_iter

    def _fn_ant_one(self, trucks: List[Truck], cargo: List[int], distances: np.ndarray,
                   alpha: float, beta: float, rand: random.Random) -> Tuple[List[List[int]], float, int]:
        routes = []
        total_cost = 0
        test_cities = 0

        for t in range(len(trucks)):
            truck = trucks[t]
            curr_client = 0
            curr_capacity = truck.capacity
            route = [curr_client]
            visited = [False] * len(cargo)
            visited[0] = True
            current_dist = 0

            while curr_capacity > 0 and sum(cargo[i] for i in range(1, len(cargo)) if not visited[i]) > 0:
                next_client = self._select_next_client(
                    curr_client, self.pheromones, distances, alpha, beta, cargo,
                    curr_capacity, visited, rand
                )
                if next_client == -1:
                    break

                dist_to_next = distances[curr_client, next_client]
                dist_to_home = distances[next_client, 0]
                if current_dist + dist_to_next + dist_to_home > truck.max_distance:
                    break

                current_dist += dist_to_next
                route.append(next_client)
                curr_capacity -= cargo[next_client]
                curr_client = next_client
                visited[curr_client] = True

            test_cities += len(route) - 1
            route.append(0)
            routes.append(route)
            total_cost += self._calc_route_cost(route, distances, truck.fuel_consumption)

        return routes, total_cost, test_cities

    def _select_next_client(self, curr_client: int, pheromones: np.ndarray, distances: np.ndarray,
                           alpha: float, beta: float, cargo: List[int], curr_capacity: int,
                           visited: List[bool], rand: random.Random) -> int:
        prob = [0.0] * len(cargo)
        total_prob = 0

        for j in range(1, len(cargo)):
            if not visited[j] and cargo[j] <= curr_capacity and j != curr_client:
                prob[j] = (pheromones[curr_client, j] ** alpha) * ((1 / distances[curr_client, j]) ** beta)
                total_prob += prob[j]

        if total_prob == 0:
            return -1

        r = rand.random() * total_prob
        cumulative = 0
        for j in range(1, len(cargo)):
            cumulative += prob[j]
            if cumulative >= r:
                return j
        return -1

    def _calc_route_cost(self, route: List[int], distances: np.ndarray, fuel: float) -> float:
        cost = 0
        for i in range(len(route) - 1):
            cost += distances[route[i], route[i + 1]]
        return cost * fuel

    def _fn_local_optim(self, routes: List[List[int]], distances: np.ndarray, fuel: List[float]) -> Tuple[List[List[int]], float]:
        new_routes = []
        new_cost = 0

        for i, route in enumerate(routes):
            opt_route, distance = self._two_opt(route, distances)
            new_routes.append(opt_route)
            new_cost += distance * fuel[i]

        return new_routes, new_cost

    def _two_opt(self, route: List[int], distances: np.ndarray) -> Tuple[List[int], float]:
        best_route = route[:]
        best_distance = self._calc_route_cost(route, distances, 1)
        improvement = True

        while improvement:
            improvement = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = self._two_opt_swap(best_route, i, j)
                    new_distance = self._calc_route_cost(new_route, distances, 1)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improvement = True
        return best_route, best_distance

    def _two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        return route[:i] + route[i:j+1][::-1] + route[j+1:]

    def nearest_neighbor(self, data: InputData, trucks: List[Truck]) -> List[List[int]]:
        routes = []
        visited = [False] * len(data.x_coords)
        visited[0] = True

        for t in range(len(trucks)):
            route = [0]
            curr = 0
            capacity = trucks[t].capacity
            dist = 0

            while capacity > 0 and any(not visited[i] and i != 0 for i in range(len(data.cargo_list))):
                next_client = -1
                min_dist = float('inf')
                for i in range(1, len(data.x_coords)):
                    if not visited[i] and data.cargo_list[i] <= capacity and data.distances[curr, i] < min_dist:
                        min_dist = data.distances[curr, i]
                        next_client = i

                if next_client == -1 or dist + min_dist + data.distances[next_client, 0] > trucks[t].max_distance:
                    break

                route.append(next_client)
                visited[next_client] = True
                capacity -= data.cargo_list[next_client]
                dist += min_dist
                curr = next_client

            route.append(0)
            routes.append(route)
        return routes

def main():
    # Load JSON data from file
    with open('test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Parse nodes
    nodes = [
        Node(
            lat=node['lat'],
            lng=node['lng'],
            label=node['label'],
            cargo=node['cargo']
        ) for node in data['nodes']
    ]

    # Parse trucks
    trucks = [
        Truck(
            capacity=truck['capacity'],
            fuel_consumption=truck['fuelConsumption'],
            max_distance=truck['maxDistance']
        ) for truck in data['trucks']
    ]

    latitudes = [node.lat for node in nodes]
    longitudes = [node.lng for node in nodes]
    cargo_list = [node.cargo for node in nodes]
    labels = [node.label for node in nodes]

    # Calculate distance matrix
    distances = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                distances[i, j] = 0
            else:
                distances[i, j] = calculate_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

    input_data = InputData(
        x_coords=latitudes,
        y_coords=longitudes,
        distances=distances,
        cargo_list=cargo_list,
        max_iterations=100
    )

    aco = ACO(trucks, input_data)
    aco_result = aco.run(threads=4)
    nn_result = aco.nearest_neighbor(input_data, trucks)

    # Print results
    print("ACO Results:")
    print(f"Time: {aco_result.algorithm_time:.2f} seconds")
    print(f"Total Cargo: {sum(aco_result.cargo):.1f}")
    print(f"Total Fuel: {aco_result.best_cost:.1f}")
    print("Routes:")
    for route in aco_result.best_routes:
        print(f"  {[labels[i] for i in route]}")
    print(f"Truck Loads: {aco_result.truck_loads}")
    print(f"Truck Clients: {aco_result.truck_clients}")
    print(f"Truck Distances: {[f'{d:.1f}' for d in aco_result.truck_distances]}")
    print(f"Truck Fuel: {[f'{f:.1f}' for f in aco_result.truck_fuel]}")

    print("\nNearest Neighbor Results:")
    print("Routes:")
    for route in nn_result:
        print(f"  {[labels[i] for i in route]}")

    # Generate Leaflet visualization
    generate_leaflet_html(nodes, aco_result, nn_result)

if __name__ == "__main__":
    main()