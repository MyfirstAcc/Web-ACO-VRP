<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VRP Web App</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@geoman-io/leaflet-geoman-free@2.17.0/dist/leaflet-geoman.min.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@geoman-io/leaflet-geoman-free@2.17.0/dist/leaflet-geoman.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios@1.6.7/dist/axios.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="m-0 font-sans">
    <div class="flex h-screen">
        <div id="map" class="flex-1"></div>
        <div class="w-1/3 p-6 overflow-y-auto bg-gray-100">
            <div class="mb-6">
                <h2 class="text-xl font-bold mb-4">Инструкция</h2>
                <p id="welcome-message" class="mb-4"></p>
                <div id="step-1" class="hidden">
                    <p class="mb-2">Добавьте начальную точку (депо) на карте. Это будет точка с грузом 0 кг.</p>
                    <button onclick="nextStep()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Продолжить</button>
                </div>
                <div id="step-2" class="hidden">
                    <p class="mb-2">Добавьте клиентские точки на карте. Укажите имя клиента и груз (больше 0 кг).</p>
                    <table class="w-full border-collapse border border-gray-300 mb-4">
                        <thead>
                            <tr class="bg-gray-200">
                                <th class="border border-gray-300 p-2">Имя клиента</th>
                                <th class="border border-gray-300 p-2">ID</th>
                                <th class="border border-gray-300 p-2">Широта</th>
                                <th class="border border-gray-300 p-2">Долгота</th>
                                <th class="border border-gray-300 p-2">Груз (кг)</th>
                                <th class="border border-gray-300 p-2">Действия</th>
                            </tr>
                        </thead>
                        <tbody id="clients-table"></tbody>
                    </table>
                    <button onclick="nextStep()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Продолжить</button>
                </div>
                <div id="step-3" class="hidden">
                    <h3 class="text-lg font-bold mb-2">Грузовики</h3>
                    <table class="w-full border-collapse border border-gray-300 mb-4">
                        <thead>
                            <tr class="bg-gray-200">
                                <th class="border border-gray-300 p-2">Название</th>
                                <th class="border border-gray-300 p-2">Грузоподъёмность (кг)</th>
                                <th class="border border-gray-300 p-2">Расход топлива (л/км)</th>
                                <th class="border border-gray-300 p-2">Макс. расстояние (км)</th>
                                <th class="border border-gray-300 p-2">Действия</th>
                            </tr>
                        </thead>
                        <tbody id="trucks"></tbody>
                    </table>
                    <button onclick="addTruck()" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 mb-2">Добавить грузовик</button>
                    <button onclick="nextStep()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Продолжить</button>
                </div>
                <div id="step-4" class="hidden">
                    <h3 class="text-lg font-bold mb-2">Параметры ACO</h3>
                    <label class="block mb-2">Количество муравьёв:</label>
                    <input type="number" id="ants" value="20" class="w-full p-2 border rounded">
                    <label class="block mb-2">Количество итераций:</label>
                    <input type="number" id="iters" value="10" class="w-full p-2 border rounded">
                    <label class="block mb-2">Alpha (феромоны):</label>
                    <input type="number" id="alpha" value="1" step="0.1" class="w-full p-2 border rounded">
                    <label class="block mb-2">Beta (расстояние):</label>
                    <input type="number" id="beta" value="10" step="0.1" class="w-full p-2 border rounded">
                    <label class="block mb-2">Rho (испарение):</label>
                    <input type="number" id="rho" value="0.05" step="0.01" class="w-full p-2 border rounded">
                    <div class="mt-4">
                        <button onclick="solveVRP()" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 mr-2">Запустить</button>
                        <button onclick="clearAll()" class="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600">Очистить</button>
                    </div>
                </div>
            </div>
            <div>
                <h2 class="text-xl font-bold mb-4">Результаты</h2>
                <p id="total-cargo" class="mb-2"></p>
                <p id="best-cost" class="mb-2"></p>
                <div id="legend" class="mt-4"></div>
                <div class="mt-6"><canvas id="routes-plot" class="w-full"></canvas></div>
                <div class="mt-6"><canvas id="dist-plot" class="w-full"></canvas></div>
                <div class="mt-6"><canvas id="load-plot" class="w-full"></canvas></div>
            </div>
        </div>
    </div>

    <script>
        let map = L.map('map').setView([59.914, 30.43], 10);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

        let nodes = [];
        let markers = [];
        let routesLayer = L.layerGroup().addTo(map);
        let step = 1;
        let clientIdCounter = 1;

        let trucks = [
            { name: "Газель 1", capacity: 1500, fuelConsumption: 0.1, maxDistance: 500 },
            { name: "Газель 2", capacity: 1500, fuelConsumption: 0.1, maxDistance: 500 },
            { name: "Isuzu 1", capacity: 3000, fuelConsumption: 0.15, maxDistance: 600 },
            { name: "Isuzu 2", capacity: 3000, fuelConsumption: 0.15, maxDistance: 600 },
            { name: "КамАЗ", capacity: 10000, fuelConsumption: 0.3, maxDistance: 800 }
        ];

        let charts = {};

        // Кастомная иконка для депо (красный маркер)
        const depotIcon = L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });

        map.pm.addControls({
            position: 'topleft',
            drawMarker: true,
            drawPolygon: false,
            drawPolyline: false,
            drawCircle: false,
            drawCircleMarker: false,
            drawRectangle: false,
            editMode: true,
            removeLayer: true
        });

        function updateWelcomeMessage() {
            const messages = {
                1: "Шаг 1: Добавьте начальную точку (депо) на карте. Это будет точка с грузом 0 кг.",
                2: "Шаг 2: Добавьте клиентские точки на карте. Укажите имя клиента и груз (больше 0 кг).",
                3: "Шаг 3: Настройте грузовики в таблице ниже.",
                4: "Шаг 4: Укажите параметры алгоритма и нажмите 'Запустить' для расчёта маршрутов."
            };
            document.getElementById('welcome-message').textContent = messages[step];
            
            document.getElementById('step-1').classList.toggle('hidden', step !== 1);
            document.getElementById('step-2').classList.toggle('hidden', step !== 2);
            document.getElementById('step-3').classList.toggle('hidden', step !== 3);
            document.getElementById('step-4').classList.toggle('hidden', step !== 4);
        }

        map.on('pm:create', async function(e) {
            let marker = e.layer;
            let latlng = marker.getLatLng();

            if (step === 1) {
                if (nodes.length > 0) {
                    await Swal.fire({
                        icon: 'warning',
                        title: 'Ошибка',
                        text: 'На этом шаге можно добавить только одну точку (депо).',
                    });
                    map.removeLayer(marker);
                    return;
                }
                nodes.push({ lat: latlng.lat, lng: latlng.lng, cargo: 0, isDepot: true });
                marker.setIcon(depotIcon); // Устанавливаем красную иконку для депо
                marker.bindPopup("Depot");
                markers.push(marker);
                marker.on('pm:remove', function() {
                    let index = markers.indexOf(marker);
                    nodes.splice(index, 1);
                    markers.splice(index, 1);
                });
            } else if (step === 2) {
                const { value: clientName } = await Swal.fire({
                    title: 'Введите имя клиента',
                    input: 'text',
                    inputValue: `Клиент ${clientIdCounter}`,
                    showCancelButton: true,
                    inputValidator: (value) => {
                        if (!value) {
                            return 'Имя клиента не может быть пустым!';
                        }
                    }
                });

                if (!clientName) {
                    map.removeLayer(marker);
                    return;
                }

                const { value: cargo } = await Swal.fire({
                    title: 'Введите груз (кг)',
                    input: 'number',
                    inputValue: '100',
                    showCancelButton: true,
                    inputValidator: (value) => {
                        const num = parseFloat(value);
                        if (!value || num <= 0) {
                            return 'Груз должен быть больше 0!';
                        }
                    }
                });

                if (!cargo) {
                    map.removeLayer(marker);
                    return;
                }

                let cargoValue = parseFloat(cargo);
                let clientId = clientIdCounter++;
                nodes.push({ lat: latlng.lat, lng: latlng.lng, cargo: cargoValue, isDepot: false, clientName: clientName, clientId: clientId });
                marker.bindPopup(`Клиент ${clientName} (Cargo: ${cargoValue} кг)`);
                markers.push(marker);
                marker.on('pm:remove', function() {
                    let index = markers.indexOf(marker);
                    nodes.splice(index, 1);
                    markers.splice(index, 1);
                    renderClientsTable();
                });
                renderClientsTable();
            } else {
                await Swal.fire({
                    icon: 'warning',
                    title: 'Ошибка',
                    text: 'Добавление точек возможно только на шагах 1 и 2.',
                });
                map.removeLayer(marker);
            }
        });

        function renderClientsTable() {
            let clientsTbody = document.getElementById('clients-table');
            clientsTbody.innerHTML = '';
            nodes.filter(node => !node.isDepot).forEach((node, index) => {
                let globalIndex = nodes.indexOf(node);
                let row = document.createElement('tr');
                row.innerHTML = `
                    <td class="border border-gray-300 p-2">
                        <input type="text" value="${node.clientName}" class="w-full p-1 border rounded" onchange="nodes[${globalIndex}].clientName = this.value">
                    </td>
                    <td class="border border-gray-300 p-2">${node.clientId}</td>
                    <td class="border border-gray-300 p-2">${node.lat.toFixed(4)}</td>
                    <td class="border border-gray-300 p-2">${node.lng.toFixed(4)}</td>
                    <td class="border border-gray-300 p-2">${node.cargo}</td>
                    <td class="border border-gray-300 p-2">
                        <button onclick="removeClient(${globalIndex})" class="bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600">Удалить</button>
                    </td>
                `;
                clientsTbody.appendChild(row);
            });
        }

        function removeClient(index) {
            map.removeLayer(markers[index]);
            nodes.splice(index, 1);
            markers.splice(index, 1);
            renderClientsTable();
        }

        function renderTrucks() {
            let trucksTbody = document.getElementById('trucks');
            trucksTbody.innerHTML = '';
            trucks.forEach((truck, index) => {
                let row = document.createElement('tr');
                row.innerHTML = `
                    <td class="border border-gray-300 p-2">
                        <input type="text" value="${truck.name}" class="w-full p-1 border rounded" onchange="trucks[${index}].name = this.value">
                    </td>
                    <td class="border border-gray-300 p-2">
                        <input type="number" value="${truck.capacity}" class="w-full p-1 border rounded" onchange="trucks[${index}].capacity = parseFloat(this.value)">
                    </td>
                    <td class="border border-gray-300 p-2">
                        <input type="number" value="${truck.fuelConsumption}" step="0.01" class="w-full p-1 border rounded" onchange="trucks[${index}].fuelConsumption = parseFloat(this.value)">
                    </td>
                    <td class="border border-gray-300 p-2">
                        <input type="number" value="${truck.maxDistance}" class="w-full p-1 border rounded" onchange="trucks[${index}].maxDistance = parseFloat(this.value)">
                    </td>
                    <td class="border border-gray-300 p-2">
                        <button onclick="removeTruck(${index})" class="bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600">Удалить</button>
                    </td>
                `;
                trucksTbody.appendChild(row);
            });
        }

        function addTruck() {
            trucks.push({ name: `Грузовик ${trucks.length + 1}`, capacity: 1500, fuelConsumption: 0.1, maxDistance: 500 });
            renderTrucks();
        }

        function removeTruck(index) {
            trucks.splice(index, 1);
            renderTrucks();
        }

        async function nextStep() {
            if (step === 1 && nodes.length !== 1) {
                await Swal.fire({
                    icon: 'warning',
                    title: 'Ошибка',
                    text: 'Добавьте ровно одну точку (депо) на первом шаге.',
                });
                return;
            }
            if (step === 2 && nodes.filter(n => !n.isDepot).length < 1) {
                await Swal.fire({
                    icon: 'warning',
                    title: 'Ошибка',
                    text: 'Добавьте хотя бы одну клиентскую точку на втором шаге.',
                });
                return;
            }
            if (step === 3 && trucks.length < 1) {
                await Swal.fire({
                    icon: 'warning',
                    title: 'Ошибка',
                    text: 'Добавьте хотя бы один грузовик на третьем шаге.',
                });
                return;
            }
            step++;
            updateWelcomeMessage();
        }

        async function solveVRP() {
            let aco = {
                ants: parseInt(document.getElementById('ants').value),
                iters: parseInt(document.getElementById('iters').value),
                alpha: parseFloat(document.getElementById('alpha').value),
                beta: parseFloat(document.getElementById('beta').value),
                rho: parseFloat(document.getElementById('rho').value)
            };

            let data = { nodes, trucks, aco };

            try {
                const response = await axios.post('/solve_vrp', data);
                let dataResponse = response.data;
                if (dataResponse.error) {
                    await Swal.fire({
                        icon: 'error',
                        title: 'Ошибка',
                        text: dataResponse.error,
                    });
                    return;
                }

                document.getElementById('total-cargo').textContent = `Суммарный перевозимый груз: ${dataResponse.total_cargo.toFixed(1)} кг`;
                document.getElementById('best-cost').textContent = `Общая потребность топлива: ${dataResponse.best_cost.toFixed(1)} литров`;

                Object.values(charts).forEach(chart => chart.destroy());

                charts['routes'] = new Chart(document.getElementById('routes-plot'), {
                    type: 'bar',
                    data: {
                        labels: dataResponse.truck_names,
                        datasets: [{
                            label: 'Пробег (км)',
                            data: dataResponse.truck_dist,
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: { y: { beginAtZero: true } },
                        plugins: { legend: { position: 'top' } }
                    }
                });

                charts['dist'] = new Chart(document.getElementById('dist-plot'), {
                    type: 'bar',
                    data: {
                        labels: dataResponse.truck_names,
                        datasets: [
                            {
                                label: 'Развоз (км)',
                                data: dataResponse.truck_dist_no1,
                                backgroundColor: 'rgba(255, 206, 86, 0.6)',
                                borderColor: 'rgba(255, 206, 86, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'В гараж (км)',
                                data: dataResponse.truck_dist,
                                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        scales: { y: { beginAtZero: true } },
                        plugins: { legend: { position: 'top' } }
                    }
                });

                charts['load'] = new Chart(document.getElementById('load-plot'), {
                    type: 'bar',
                    data: {
                        labels: dataResponse.truck_names,
                        datasets: [
                            {
                                label: 'Количество клиентов',
                                data: dataResponse.truck_clients,
                                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'Загрузка (кг)',
                                data: dataResponse.truck_loads,
                                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        scales: { y: { beginAtZero: true } },
                        plugins: { legend: { position: 'top' } }
                    }
                });

                routesLayer.clearLayers();

                dataResponse.routes_geo.forEach(route => {
                    L.polyline(route.route, { color: route.color, weight: 5, opacity: 0.7 })
                        .bindPopup(route.truck)
                        .addTo(routesLayer);
                });

                let legendDiv = document.getElementById('legend');
                legendDiv.innerHTML = '<h3 class="text-lg font-bold mb-2">Легенда</h3>';
                dataResponse.routes_geo.forEach(route => {
                    let item = document.createElement('div');
                    item.className = 'flex items-center mb-2';
                    item.innerHTML = `<div class="w-5 h-5 mr-2" style="background: ${route.color}"></div>${route.truck}`;
                    legendDiv.appendChild(item);
                });
            } catch (error) {
                console.error('Ошибка:', error);
                await Swal.fire({
                    icon: 'error',
                    title: 'Ошибка',
                    text: 'Ошибка при расчёте маршрутов',
                });
            }
        }

        function clearAll() {
            nodes = [];
            markers.forEach(marker => map.removeLayer(marker));
            markers = [];
            routesLayer.clearLayers();
            trucks = [
                { name: "Газель 1", capacity: 1500, fuelConsumption: 0.1, maxDistance: 500 },
                { name: "Газель 2", capacity: 1500, fuelConsumption: 0.1, maxDistance: 500 },
                { name: "Isuzu 1", capacity: 3000, fuelConsumption: 0.15, maxDistance: 600 },
                { name: "Isuzu 2", capacity: 3000, fuelConsumption: 0.15, maxDistance: 600 },
                { name: "КамАЗ", capacity: 10000, fuelConsumption: 0.3, maxDistance: 800 }
            ];
            clientIdCounter = 1;
            step = 1;
            document.getElementById('total-cargo').textContent = '';
            document.getElementById('best-cost').textContent = '';
            document.getElementById('legend').innerHTML = '';
            Object.values(charts).forEach(chart => chart.destroy());
            charts = {};
            renderClientsTable();
            renderTrucks();
            updateWelcomeMessage();
        }

        renderTrucks();
        updateWelcomeMessage();
    </script>
</body>
</html>