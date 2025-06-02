import numpy as np
import json
import urllib.parse
from flask import Flask, request, jsonify, render_template
from solution_VRP import Truck, Client, ACOConfig, VRP


app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_vrp', methods=['POST'])
def solve_vrp():
    data = request.get_json()
    
    nodes = data['nodes']
    depot_index = next(i for i, node in enumerate(nodes) if node['cargo'] == 0)
    nodes.insert(0, nodes.pop(depot_index))
    
    clients = [Client(lat=node['lat'], lng=node['lng'], cargo=node['cargo'], index=i) 
              for i, node in enumerate(nodes)]
    
    trucks_data = data['trucks']
    trucks = [Truck(name=truck['name'], capacity=truck['capacity'], 
                   fuel_consumption=truck['fuelConsumption'], 
                   max_distance=truck['maxDistance'], index=i) 
              for i, truck in enumerate(trucks_data)]
    
    aco = ACOConfig(**data['aco'])
    
    vrp = VRP(clients, trucks, aco)
    result = vrp.solve(local_opt=True)
    
    return jsonify(result.to_dict())

@app.route('/api/solve_vrp_test', methods=['POST'])
def solve_vrp_test():
    with open('test.json', 'r') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    depot_index = next(i for i, node in enumerate(nodes) if node['cargo'] == 0)
    nodes.insert(0, nodes.pop(depot_index))
    
    clients = [Client(lat=node['lat'], lng=node['lng'], cargo=node['cargo'], index=i) 
              for i, node in enumerate(nodes)]
    
    trucks_data = data['trucks']
    trucks = [Truck(name=truck['name'], capacity=truck['capacity'], 
                   fuel_consumption=truck['fuelConsumption'], 
                   max_distance=truck['maxDistance'], index=i) 
              for i, truck in enumerate(trucks_data)]
    
    aco = ACOConfig(**data['aco'])
    
    vrp = VRP(clients, trucks, aco)
    result = vrp.solve(local_opt=True)
    
    return jsonify(result.to_dict())

@app.route('/api/autocomplete', methods=['GET'])
def autocomplete_address():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    query = urllib.parse.unquote(query)
    
    try:
        response = requests.get('https://nominatim.openstreetmap.org/search', params={
            'q': query,
            'format': 'json',
            'limit': 5,
            'countrycodes': 'ru'
        }, headers={
            'User-Agent': 'VRP-Web-App/1.0'
        }, timeout=5)
        
        if response.status_code != 200:
            return jsonify({'error': 'Не удалось получить данные c Nominatim', 'status': response.status_code}), response.status_code
        
        data = response.json()
        print('Ответ Nominatim:', data)
        return jsonify(data)
    
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)