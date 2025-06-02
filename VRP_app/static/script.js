document.addEventListener('DOMContentLoaded', function () {
    renderTrucks();
    updateWelcomeMessage();
    setupAddressAutocomplete();
});

let map = L.map('map').setView([59.914, 30.43], 10);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

setTimeout(() => map.invalidateSize(), 100);
let nodes = [];
let markers = [];
let labels = [];
let routeLabels = [];
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
let routePolylines = {};
let currentDataResponse = null;

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
    removeLayers: false,
    cutPolygon: false,
    drawText: false,
    dragMode: false,
    removalMode: false,
    rotateMode: false
});

const sidepanel = document.getElementById('sidepanel');
const toggleButton = document.getElementById('sidepanel-toggle');
const toggleIcon = document.getElementById('toggle-icon');
let isSidepanelOpen = true;

toggleButton.addEventListener('click', () => {
    if (isSidepanelOpen) {
        sidepanel.classList.add('collapsed');
        toggleButton.classList.add('rotated');
        isSidepanelOpen = false;
    } else {
        sidepanel.classList.remove('collapsed');
        toggleButton.classList.remove('rotated');
        isSidepanelOpen = true;
    }
    setTimeout(() => map.invalidateSize(), 300);
});

function updateWelcomeMessage() {
    const messages = {
        1: "Шаг 1: Добавьте начальную точку (депо) на карте. Это будет точка с грузом 0 кг.",
        2: "Шаг 2: Добавьте клиентские точки на карте или используйте поиск по адресу. Укажите имя клиента и груз (больше 0 кг).",
        3: "Шаг 3: Настройте грузовики в таблице ниже.",
        4: "Шаг 4: Укажите параметры алгоритма и нажмите 'Запустить' для расчёта маршрутов."
    };
    document.getElementById('welcome-message').textContent = messages[step];

    document.getElementById('step-1').classList.toggle('hidden', step !== 1);
    document.getElementById('step-2').classList.toggle('hidden', step !== 2);
    document.getElementById('step-3').classList.toggle('hidden', step !== 3);
    document.getElementById('step-4').classList.toggle('hidden', step !== 4);

    const resultDiv = document.getElementById('result');
    if (step === 4) {
        resultDiv.classList.remove('hidden');
    } else {
        resultDiv.classList.add('hidden');
    }
}

function setupAddressAutocomplete() {
    const addressInput = document.getElementById('address-input');
    const suggestionsList = document.getElementById('address-suggestions');

    let debounceTimeout;
    addressInput.addEventListener('input', () => {
        clearTimeout(debounceTimeout);
        debounceTimeout = setTimeout(async () => {
            const query = addressInput.value.trim();
            if (query.length < 3) {
                suggestionsList.innerHTML = '';
                suggestionsList.classList.add('hidden');
                return;
            }

            try {
                const response = await axios.get('/api/autocomplete', {
                    params: {
                        q: query,
                        limit: 5,
                        lang: 'ru'
                    }
                });

                suggestionsList.innerHTML = '';
                suggestionsList.classList.remove('hidden');

                // Nominatim возвращает массив объектов напрямую в response.data
                response.data.forEach(item => {
                    const li = document.createElement('li');
                    const name = item.name || '';
                    const displayName = item.display_name || '';
                    li.textContent = displayName || name;
                    li.classList.add('border-b', 'border-gray-200');
                    li.addEventListener('click', () => {
                        addressInput.value = li.textContent;
                        suggestionsList.innerHTML = '';
                        suggestionsList.classList.add('hidden');
                        searchAddress(parseFloat(item.lat), parseFloat(item.lon));
                    });
                    suggestionsList.appendChild(li);
                });

                if (response.data.length === 0) {
                    suggestionsList.innerHTML = '<li class="p-2 text-gray-500">Ничего не найдено</li>';
                }
            } catch (error) {
                console.error('Ошибка при автодополнении:', error);
                console.error('Код ответа:', error.response?.status);
                console.error('Данные ответа:', error.response?.data);
                suggestionsList.innerHTML = `<li class="p-2 text-red-500">Ошибка при поиске: ${error.response?.data?.error || error.message}</li>`;
                suggestionsList.classList.remove('hidden');
            }
        }, 500);
    });

    document.addEventListener('click', (e) => {
        if (!addressInput.contains(e.target) && !suggestionsList.contains(e.target)) {
            suggestionsList.innerHTML = '';
            suggestionsList.classList.add('hidden');
        }
    });
}

async function searchAddress(lat = null, lng = null) {
    if (!lat || !lng) {
        const addressInput = document.getElementById('address-input').value;
        if (!addressInput) {
            await Swal.fire({
                icon: 'warning',
                title: 'Ошибка',
                text: 'Введите адрес для поиска!',
            });
            return;
        }

        try {
            const response = await axios.get('/api/autocomplete', {
                params: {
                    q: addressInput,
                    limit: 1
                }
            });

            console.log('Ответ от /api/autocomplete:', response.data); // Для отладки

            if (!response.data || response.data.length === 0) {
                await Swal.fire({
                    icon: 'error',
                    title: 'Ошибка',
                    text: 'Адрес не найден!',
                });
                return;
            }

            lat = parseFloat(response.data[0].lat);
            lng = parseFloat(response.data[0].lon);

            if (isNaN(lat) || isNaN(lng)) {
                console.error('Некорректные координаты:', response.data[0]);
                await Swal.fire({
                    icon: 'error',
                    title: 'Ошибка',
                    text: 'Некорректные координаты адреса!',
                });
                return;
            }
        } catch (error) {
            console.error('Ошибка при поиске адреса:', error);
            await Swal.fire({
                icon: 'error',
                title: 'Ошибка',
                text: 'Не удалось найти адрес: ' + (error.response?.data?.error || error.message),
            });
            return;
        }
    }

    console.log('Добавление маркера на координаты:', { lat, lng }); // Для отладки

    // Создаём маркер
    const marker = L.marker([lat, lng]).addTo(map);
    map.setView([lat, lng], 15);
    setTimeout(() => map.invalidateSize(), 100); // Обновляем карту

    // Запрашиваем имя клиента
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
    }).then((result) => {
        setTimeout(() => map.invalidateSize(), 100);
        return result;
    });

    if (!clientName) {
        map.removeLayer(marker);
        console.log('Добавление отменено: имя клиента не введено');
        return;
    }

    // Запрашиваем груз
    const { value: cargo } = await Swal.fire({
        title: 'Введите груз (кг)',
        input: 'number',
        inputValue: getRandomInt(50, 399),
        showCancelButton: true,
        inputValidator: (value) => {
            const num = parseFloat(value);
            if (!value || num <= 0) {
                return 'Груз должен быть больше 0!';
            }
        }
    }).then((result) => {
        setTimeout(() => map.invalidateSize(), 100);
        return result;
    });

    if (!cargo) {
        map.removeLayer(marker);
        console.log('Добавление отменено: груз не введён');
        return;
    }

    let cargoValue = parseFloat(cargo);
    let clientId = clientIdCounter++;
    
    // Добавляем точку в nodes
    nodes.push({ lat, lng, cargo: cargoValue, isDepot: false, clientName: clientName, clientId: clientId });
    marker.bindPopup(`${clientName} (Груз: ${cargoValue} кг)`);
    markers.push(marker);

    // Добавляем метку
    let label = L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'label-icon',
            html: ``,
            iconSize: [20, 20]
        })
    }).addTo(map);
    labels.push(label);

    // Обработчик удаления
    marker.on('pm:remove', function () {
        let index = markers.indexOf(marker);
        nodes.splice(index, 1);
        markers.splice(index, 1);
        map.removeLayer(labels[index]);
        labels.splice(index, 1);
        renderClientsTable();
    });

    console.log('Точка добавлена:', { clientName, clientId, lat, lng, cargoValue }); // Для отладки
    renderClientsTable(); // Обновляем таблицу
}

map.on('pm:create', async function (e) {
    let marker = e.layer;
    let latlng = marker.getLatLng();

    if (step === 1) {
        if (nodes.length > 0) {
            await Swal.fire({
                icon: 'warning',
                title: 'Ошибка',
                text: 'На этом шаге можно добавить только одну точку (депо).',
            }).then(() => {
                setTimeout(() => map.invalidateSize(), 100);
            });
            map.removeLayer(marker);
            return;
        }
        nodes.push({ lat: latlng.lat, lng: latlng.lng, cargo: 0, isDepot: true });
        marker.setIcon(depotIcon);
        marker.bindPopup("Depot");
        markers.push(marker);

        let label = L.marker([latlng.lat, latlng.lng], {
            icon: L.divIcon({
                className: 'label-icon',
                html: ``,
                iconSize: [20, 20]
            })
        }).addTo(map);
        labels.push(label);

        marker.on('pm:remove', function () {
            let index = markers.indexOf(marker);
            nodes.splice(index, 1);
            markers.splice(index, 1);
            map.removeLayer(labels[index]);
            labels.splice(index, 1);
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
        }).then((result) => {
            setTimeout(() => map.invalidateSize(), 100);
            return result;
        });

        if (!clientName) {
            map.removeLayer(marker);
            return;
        }

        const { value: cargo } = await Swal.fire({
            title: 'Введите груз (кг)',
            input: 'number',
            inputValue: getRandomInt(50, 399),
            showCancelButton: true,
            inputValidator: (value) => {
                const num = parseFloat(value);
                if (!value || num <= 0) {
                    return 'Груз должен быть больше 0!';
                }
            }
        }).then((result) => {
            setTimeout(() => map.invalidateSize(), 100);
            return result;
        });

        if (!cargo) {
            map.removeLayer(marker);
            return;
        }

        let cargoValue = parseFloat(cargo);
        let clientId = clientIdCounter++;
        // Добавление киента в массив клиентов (геоточки, груз, имя клиента, ID)
        nodes.push({ lat: latlng.lat, lng: latlng.lng, 
            cargo: cargoValue, isDepot: false, 
            clientName: clientName, clientId: clientId });

        marker.bindPopup(`${clientName} (Груз: ${cargoValue} кг)`);
        markers.push(marker);
        // добавление маркера на карту 
        let label = L.marker([latlng.lat, latlng.lng], {
            icon: L.divIcon({
                className: 'label-icon',
                html: ``,
                iconSize: [20, 20]
            })
        }).addTo(map);
        labels.push(label);
        // обработчик для удаления маркера из таблицы клиентов
        marker.on('pm:remove', function () {
            let index = markers.indexOf(marker);
            nodes.splice(index, 1);
            markers.splice(index, 1);
            map.removeLayer(labels[index]);
            labels.splice(index, 1);
            renderClientsTable();
        });
        // добавление в таблицу клиентов
        renderClientsTable();
    } else {
        await Swal.fire({
            icon: 'warning',
            title: 'Ошибка',
            text: 'Добавление точек возможно только на шагах 1 и 2.',
        }).then(() => {
            setTimeout(() => map.invalidateSize(), 100);
        });
        map.removeLayer(marker);
    }
});

function renderClientsTable() {
    let clientsTbody = document.getElementById('clients-table');
    clientsTbody.innerHTML = '';
    nodes.filter(node => !node.isDepot).forEach((node, index) => {
        let globalIndex = nodes.indexOf(node);
        let clientNumber = index + 2;
        let row = document.createElement('tr');
        row.innerHTML = `
            <td class="border border-gray-300 p-2">
                <input type="text" value="${node.clientName}" class="w-full p-1 border rounded" onchange="nodes[${globalIndex}].clientName = this.value">
            </td>
            <td class="border border-gray-300 p-2">${clientNumber}</td>
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
    map.removeLayer(labels[index]);
    nodes.splice(index, 1);
    markers.splice(index, 1);
    labels.splice(index, 1);
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

function createCharts(activeTruckNames, activeDist, activeDistNo1, activeClients, activeLoads) {
    const routesCanvas = document.getElementById('routes-plot');
    const distCanvas = document.getElementById('dist-plot');
    const loadCanvas = document.getElementById('load-plot');

    Object.values(charts).forEach(chart => chart.destroy());

    charts['routes'] = new Chart(routesCanvas, {
        type: 'bar',
        data: {
            labels: activeTruckNames,
            datasets: [{
                label: 'Пробег (км)',
                data: activeDist,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: { y: { beginAtZero: true } },
            plugins: { legend: { position: 'top' } },
            responsive: false,
            maintainAspectRatio: false
        }
    });

    charts['dist'] = new Chart(distCanvas, {
        type: 'bar',
        data: {
            labels: activeTruckNames,
            datasets: [
                {
                    label: 'Развоз (км)',
                    data: activeDistNo1,
                    backgroundColor: 'rgba(255, 206, 86, 0.6)',
                    borderColor: 'rgba(255, 206, 86, 1)',
                    borderWidth: 1
                },
                {
                    label: 'В гараж (км)',
                    data: activeDist,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            scales: { y: { beginAtZero: true } },
            plugins: { legend: { position: 'top' } },
            responsive: false,
            maintainAspectRatio: false
        }
    });

    charts['load'] = new Chart(loadCanvas, {
        type: 'bar',
        data: {
            labels: activeTruckNames,
            datasets: [
                {
                    label: 'Количество клиентов',
                    data: activeClients,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Загрузка (кг)',
                    data: activeLoads,
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            scales: { y: { beginAtZero: true } },
            plugins: { legend: { position: 'top' } },
            responsive: false,
            maintainAspectRatio: false
        }
    });
}

async function solveVRP() {
    // формирование JSON для алгоритма ACO
    let aco = {
        ants: parseInt(document.getElementById('ants').value),
        iters: parseInt(document.getElementById('iters').value),
        alpha: parseFloat(document.getElementById('alpha').value),
        beta: parseFloat(document.getElementById('beta').value),
        rho: parseFloat(document.getElementById('rho').value)
    };
    // Другие параметры: клиенты, грущовики, параметры ACO
    let data = { nodes, trucks, aco };

    try {
        // Отправка запроса на сервер и получение ответа
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

        currentDataResponse = dataResponse;
        // Добавление результата в div'ы
        document.getElementById('total-cargo')
        .textContent = `Суммарный перевозимый груз: ${dataResponse.total_cargo.toFixed(1)} кг`;
        document.getElementById('best-cost')
        .textContent = `Общая потребность топлива: ${dataResponse.best_cost.toFixed(1)} литров`;

        let activeTruckNames = dataResponse.routes_geo.map(route => route.truck).filter(name => name);
        let activeTruckIndices = dataResponse.routes_geo.map((route, index) => index);

        let activeDist = activeTruckIndices.map(i => dataResponse.truck_dist[i] !== undefined ? dataResponse.truck_dist[i] : 0);
        let activeDistNo1 = activeTruckIndices.map(i => dataResponse.truck_dist_no1[i] !== undefined ? dataResponse.truck_dist_no1[i] : 0);
        let activeClients = activeTruckIndices.map(i => dataResponse.truck_clients[i] !== undefined ? dataResponse.truck_clients[i] : 0);
        let activeLoads = activeTruckIndices.map(i => dataResponse.truck_loads[i] !== undefined ? dataResponse.truck_loads[i] : 0);

        if (activeTruckNames.length === 0) {
            console.warn('Нет активных грузовиков в routes_geo');
            return;
        }

        createCharts(activeTruckNames, activeDist, activeDistNo1, activeClients, activeLoads);

        routesLayer.clearLayers();
        routePolylines = {};

        dataResponse.routes_geo.forEach((route, index) => {
            if (route.route && route.route.length > 0) {
                let polyline = L.polyline(route.route, { color: route.color, weight: 5, opacity: 0.7 })
                    .bindPopup(route.truck);
                routePolylines[index] = polyline;
                polyline.addTo(routesLayer);
            }
        });

        let select = document.getElementById('route-select');
        select.innerHTML = '<option value="-1">Все маршруты</option>';
        dataResponse.routes_geo.forEach((route, index) => {
            if (route.route && route.route.length > 0) {
                let option = document.createElement('option');
                option.value = index;
                option.text = `${route.truck} (Цвет: ${route.color})`;
                select.appendChild(option);
            }
        });

        showRoute("-1", dataResponse);

        let legendDiv = document.getElementById('legend');
        legendDiv.innerHTML = '<h3 class="text-lg font-bold mb-2">Легенда</h3>';
        dataResponse.routes_geo.forEach(route => {
            if (route.route && route.route.length > 0) {
                let item = document.createElement('div');
                item.className = 'flex items-center mb-2';
                item.innerHTML = `<div class="w-5 h-5 mr-2" style="background: ${route.color}"></div>${route.truck}`;
                legendDiv.appendChild(item);
            }
        });

        document.getElementById('result').classList.remove('hidden');
    } catch (error) {
        console.error('Ошибка:', error);
        await Swal.fire({
            icon: 'error',
            title: 'Ошибка',
            text: 'Ошибка при расчёте маршрутов: ' + error.message,
        });
    }
}

function showRoute(routeIndex, dataResponse) {
    routesLayer.clearLayers();
    routeLabels.forEach(label => map.removeLayer(label));
    routeLabels = [];

    if (routeIndex === "-1") {
        for (let index in routePolylines) {
            routePolylines[index].addTo(routesLayer);
        }
        labels.forEach(label => label.addTo(map));
    } else if (routePolylines[routeIndex]) {
        routePolylines[routeIndex].addTo(routesLayer);

        let routeIndices = currentDataResponse.routes_geo[routeIndex].routeIndices || [];
        if (routeIndices.length === 0) {
            console.warn(`routeIndices отсутствует для маршрута ${routeIndex}, пропускаем метки`);
            return;
        }

        routeIndices.forEach(indexInfo => {
            let nodeIdx = indexInfo.nodeIndex;
            let routeIdx = indexInfo.routeIndex;

            if (nodeIdx >= 0 && nodeIdx < nodes.length) {
                let node = nodes[nodeIdx];
                let point = [node.lat, node.lng];
                let label = L.marker(point, {
                    icon: L.divIcon({
                        className: 'label-icon',
                        html: `<div style="font-size: 14px; color: black; font-weight: bold;">${routeIdx}</div>`,
                        iconSize: [20, 20]
                    })
                }).addTo(routesLayer);
                routeLabels.push(label);
            }
        });
    }
}

function clearAll() {
    nodes = [];
    markers.forEach((marker, index) => {
        map.removeLayer(marker);
        map.removeLayer(labels[index]);
    });
    markers = [];
    labels = [];
    routeLabels.forEach(label => map.removeLayer(label));
    routeLabels = [];
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
    document.getElementById('route-select').innerHTML = '<option value="-1">Все маршруты</option>';
    currentDataResponse = null;
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};
    renderClientsTable();
    renderTrucks();
    updateWelcomeMessage();
    document.getElementById('result').classList.add('hidden');
}