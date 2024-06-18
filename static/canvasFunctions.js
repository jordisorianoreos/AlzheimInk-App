export function showCanvas_default() {
    // Hide the image and the "Realizar Test" and "Iniciar Sesión" buttons
    document.getElementById('image').style.display = 'none';
    document.getElementById('showCanvasButton').style.display = 'none';
    document.getElementById('loginButton').style.display = 'none';

    // Show the Canvas and the "Next Task" button
    document.getElementById('canvas').style.display = 'block';
    let nextTaskButton = document.getElementById('nextTaskButton');
    nextTaskButton.style.display = 'block';
    nextTaskButton.disabled = true;

    // Assign dimensions and other styles to the "Next Task" button
    nextTaskButton.style.height = '450px';  // Button height
    nextTaskButton.style.width = '80px';   // Button width

    alert('ATENCIÓN: La prueba que está a punto de realizar es puramente informativa. ' +
      'Por favor, tenga en cuenta que la prueba puede dar lugar a predicciones erróneas. ' +
      'Considere esta información antes de proceder.');

    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let drawing = false;
    let canStartDrawing = true;

    // Reset upon page refresh
    fetch('/reset_test', { method: 'POST' });

    function getPointerPosition(event) {
        let rect = canvas.getBoundingClientRect();
        if (event.touches) { // Touch event
            return {
                x: event.touches[0].clientX - rect.left,
                y: event.touches[0].clientY - rect.top
            };
        } else { // Mouse event
            return {
                x: event.offsetX,
                y: event.offsetY
            };
        }
    }

    function startDrawing(event) {
        if (!canStartDrawing) return;
        drawing = true;
        let pos = getPointerPosition(event);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        fetch('/start_drawing', { method: 'POST' });
    }

    function draw(event) {
        if (!drawing) return;
        let pos = getPointerPosition(event);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        fetch('/draw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: pos.x, y: pos.y })
        });
    }

    function stopDrawing() {
        if (!drawing) return;
        drawing = false;
        ctx.closePath();
        fetch('/stop_drawing', { method: 'POST' });
        nextTaskButton.disabled = false;
    }

    function loadImage(taskNumber) {
        let img = new Image();
        img.onload = function() {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = `/image/${taskNumber}`;
    }

    loadImage(1);

    nextTaskButton.addEventListener('click', () => {
        fetch('/next_task', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'finished') {
                    canStartDrawing = false;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.font = '30px Arial';
                    ctx.textAlign = "center";
                    ctx.fillText('Test finalizado', canvas.width / 2, canvas.height / 2 - 20);
                    ctx.fillText('(espera los resultados...)', canvas.width / 2, canvas.height / 2 + 20);
                    nextTaskButton.disabled = true;

                    fetch('/make_prediction', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'Prediction made') {
                                ctx.clearRect(0, 0, canvas.width, canvas.height);
                                ctx.font = '30px Arial';
                                ctx.textAlign = "center";
                                ctx.fillText('Diagnóstico: ' + data.prediction, canvas.width / 2, canvas.height / 2 - 10);
                                ctx.font = '18px Arial';
                                ctx.fillStyle = 'red'
                                ctx.fillText('ADVERTENCIA', canvas.width / 2, canvas.height / 2 + 40);
                                ctx.fillStyle = 'black'
                                ctx.fillText('La predicción realizada puede ser errónea. Si tienes dudas', canvas.width / 2, canvas.height / 2 + 60);
                                ctx.fillText('o crees que es necesario, consulta a un profesional de la salud.', canvas.width / 2, canvas.height / 2 + 80);
                            }
                        });
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Memorize Three Words
                    if (data.task_number === 13) {
                        canStartDrawing = false;
                        ctx.font = '30px Arial';
                        ctx.textAlign = "center";
                        ctx.fillText('Memoriza las siguientes', canvas.width / 2, canvas.height / 2 - 80);
                        ctx.fillText('palabras antes de continuar:', canvas.width / 2, canvas.height / 2 - 50);
                        ctx.fillText('teléfono', canvas.width / 2, canvas.height / 2 + 10);
                        ctx.fillText('cena', canvas.width / 2, canvas.height / 2 + 40);
                        ctx.fillText('negocio', canvas.width / 2, canvas.height / 2 + 70);

                        // Create and show the accept button
                        let acceptButton = document.createElement('button');
                        acceptButton.innerText = 'Ya las he memorizado';
                        acceptButton.style.position = 'absolute';
                        acceptButton.style.left = `${canvas.offsetLeft + canvas.width / 2 - 100}px`;
                        acceptButton.style.top = `${canvas.offsetTop + canvas.height / 2 + 100}px`;
                        acceptButton.style.width = '200px';  // Button width
                        acceptButton.style.height = '60px';  // Button height
                        acceptButton.style.fontSize = '20px';
                        acceptButton.addEventListener('click', () => {
                            // Load task 13 image after clicking accept button
                            canStartDrawing = true;
                            loadImage(data.task_number);
                            // Remove accept button after clicking it
                            document.body.removeChild(acceptButton);
                        });
                        document.body.appendChild(acceptButton);
                    } else {
                        loadImage(data.task_number);
                    }
                    nextTaskButton.disabled = true;
                }
            });
    });

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', function(event) {
        event.preventDefault();
        startDrawing(event);
    });
    canvas.addEventListener('touchmove', function(event) {
        event.preventDefault();
        draw(event);
    });
    canvas.addEventListener('touchend', function(event) {
        event.preventDefault();
        stopDrawing();
    });
}

export function showCanvas_patient() {
    // Hide the image and the "Realizar Test" and "Iniciar Sesión" buttons
    document.getElementById('image').style.display = 'none';
    document.getElementById('showCanvasButton').style.display = 'none';
    document.getElementById('loginButton').style.display = 'none';

    // Show the Canvas and the "Next Task" button
    document.getElementById('canvas').style.display = 'block';
    let nextTaskButton = document.getElementById('nextTaskButton');
    nextTaskButton.style.display = 'block';
    nextTaskButton.disabled = true;

    // Assign dimensions and other styles to the "Next Task" button
    nextTaskButton.style.height = '450px';  // Button height
    nextTaskButton.style.width = '80px';   // Button width

    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let drawing = false;
    let canStartDrawing = true;

    // Reset upon page refresh
    fetch('/reset_test', { method: 'POST' });

    function getPointerPosition(event) {
        let rect = canvas.getBoundingClientRect();
        if (event.touches) { // Touch event
            return {
                x: event.touches[0].clientX - rect.left,
                y: event.touches[0].clientY - rect.top
            };
        } else { // Mouse event
            return {
                x: event.offsetX,
                y: event.offsetY
            };
        }
    }

    function startDrawing(event) {
        if (!canStartDrawing) return;
        drawing = true;
        let pos = getPointerPosition(event);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        fetch('/start_drawing', { method: 'POST' });
    }

    function draw(event) {
        if (!drawing) return;
        let pos = getPointerPosition(event);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        fetch('/draw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: pos.x, y: pos.y })
        });
    }

    function stopDrawing() {
        if (!drawing) return;
        drawing = false;
        ctx.closePath();
        fetch('/stop_drawing', { method: 'POST' });
        nextTaskButton.disabled = false;
    }

    function loadImage(taskNumber) {
        let img = new Image();
        img.onload = function() {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = `/image/${taskNumber}`;
    }

    loadImage(1);

    nextTaskButton.addEventListener('click', () => {
        fetch('/next_task', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'finished') {
                    canStartDrawing = false;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.font = '30px Arial';
                    ctx.textAlign = "center";
                    ctx.fillText('Test finalizado', canvas.width / 2, canvas.height / 2 - 20);
                    ctx.fillText('(espera los resultados...)', canvas.width / 2, canvas.height / 2 + 20);
                    nextTaskButton.disabled = true;

                    fetch('/make_prediction_patient', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'Prediction made') {
                                ctx.clearRect(0, 0, canvas.width, canvas.height);
                                ctx.font = '30px Arial';
                                ctx.textAlign = "center";
                                ctx.fillText('Diagnóstico: ' + data.prediction, canvas.width / 2, canvas.height / 2);
                            }
                        });
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Memorize Three Words
                    if (data.task_number === 13) {
                        canStartDrawing = false;
                        ctx.font = '30px Arial';
                        ctx.textAlign = "center";
                        ctx.fillText('Memoriza las siguientes', canvas.width / 2, canvas.height / 2 - 80);
                        ctx.fillText('palabras antes de continuar:', canvas.width / 2, canvas.height / 2 - 50);
                        ctx.fillText('teléfono', canvas.width / 2, canvas.height / 2 + 10);
                        ctx.fillText('cena', canvas.width / 2, canvas.height / 2 + 40);
                        ctx.fillText('negocio', canvas.width / 2, canvas.height / 2 + 70);

                        // Create and show the accept button
                        let acceptButton = document.createElement('button');
                        acceptButton.innerText = 'Ya las he memorizado';
                        acceptButton.style.position = 'absolute';
                        acceptButton.style.left = `${canvas.offsetLeft + canvas.width / 2 - 100}px`;
                        acceptButton.style.top = `${canvas.offsetTop + canvas.height / 2 + 100}px`;
                        acceptButton.style.width = '200px';  // Button width
                        acceptButton.style.height = '60px';  // Button height
                        acceptButton.style.fontSize = '20px';
                        acceptButton.addEventListener('click', () => {
                            // Load task 13 image after clicking accept button
                            canStartDrawing = true;
                            loadImage(data.task_number);
                            // Remove accept button after clicking it
                            document.body.removeChild(acceptButton);
                        });
                        document.body.appendChild(acceptButton);
                    } else {
                        loadImage(data.task_number);
                    }
                    nextTaskButton.disabled = true;
                }
            });
    });

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', function(event) {
        event.preventDefault();
        startDrawing(event);
    });
    canvas.addEventListener('touchmove', function(event) {
        event.preventDefault();
        draw(event);
    });
    canvas.addEventListener('touchend', function(event) {
        event.preventDefault();
        stopDrawing();
    });
}

export function showCanvas_add_data() {
    // Hide the image and the "Realizar Test" and "Iniciar Sesión" buttons
    document.getElementById('image').style.display = 'none';
    document.getElementById('showCanvasButton').style.display = 'none';

    // Show the Canvas and the "Next Task" button
    document.getElementById('canvas').style.display = 'block';
    let nextTaskButton = document.getElementById('nextTaskButton');
    nextTaskButton.style.display = 'block';
    nextTaskButton.disabled = true;

    // Assign dimensions and other styles to the "Next Task" button
    nextTaskButton.style.height = '450px';  // Button height
    nextTaskButton.style.width = '80px';   // Button width

    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let drawing = false;
    let canStartDrawing = true;

    // Reset upon page refresh
    fetch('/reset_test_add_data', { method: 'POST' });

    function getPointerPosition(event) {
        let rect = canvas.getBoundingClientRect();
        if (event.touches) { // Touch event
            return {
                x: event.touches[0].clientX - rect.left,
                y: event.touches[0].clientY - rect.top
            };
        } else { // Mouse event
            return {
                x: event.offsetX,
                y: event.offsetY
            };
        }
    }

    function startDrawing(event) {
        if (!canStartDrawing) return;
        drawing = true;
        let pos = getPointerPosition(event);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        fetch('/start_drawing_add_data', { method: 'POST' });
    }

    function draw(event) {
        if (!drawing) return;
        let pos = getPointerPosition(event);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        fetch('/draw_add_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: pos.x, y: pos.y })
        });
    }

    function stopDrawing() {
        if (!drawing) return;
        drawing = false;
        ctx.closePath();
        fetch('/stop_drawing_add_data', { method: 'POST' });
        nextTaskButton.disabled = false;
    }

    function loadImage(taskNumber) {
        let img = new Image();
        img.onload = function() {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = `/image/${taskNumber}`;
    }

    loadImage(1);

    nextTaskButton.addEventListener('click', () => {
        fetch('/next_task_add_data', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'finished') {
                    canStartDrawing = false;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    nextTaskButton.disabled = true;

                    fetch('/save_data_new_database', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'Data saved') {
                                ctx.font = '30px Arial';
                                ctx.textAlign = "center";
                                ctx.fillText('Datos guardados', canvas.width / 2, canvas.height / 2 - 20);
                                ctx.fillText('(refresque la página para salir)', canvas.width / 2, canvas.height / 2 + 20);
                            }
                        });
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Memorize Three Words
                    if (data.task_number === 13) {
                        canStartDrawing = false;
                        ctx.font = '30px Arial';
                        ctx.textAlign = "center";
                        ctx.fillText('Memoriza las siguientes', canvas.width / 2, canvas.height / 2 - 80);
                        ctx.fillText('palabras antes de continuar:', canvas.width / 2, canvas.height / 2 - 50);
                        ctx.fillText('teléfono', canvas.width / 2, canvas.height / 2 + 10);
                        ctx.fillText('cena', canvas.width / 2, canvas.height / 2 + 40);
                        ctx.fillText('negocio', canvas.width / 2, canvas.height / 2 + 70);

                        // Create and show the accept button
                        let acceptButton = document.createElement('button');
                        acceptButton.innerText = 'Ya las he memorizado';
                        acceptButton.style.position = 'absolute';
                        acceptButton.style.left = `${canvas.offsetLeft + canvas.width / 2 - 100}px`;
                        acceptButton.style.top = `${canvas.offsetTop + canvas.height / 2 + 100}px`;
                        acceptButton.style.width = '200px';  // Button width
                        acceptButton.style.height = '60px';  // Button height
                        acceptButton.style.fontSize = '20px';
                        acceptButton.addEventListener('click', () => {
                            // Load task 13 image after clicking accept button
                            canStartDrawing = true;
                            loadImage(data.task_number);
                            // Remove accept button after clicking it
                            document.body.removeChild(acceptButton);
                        });
                        document.body.appendChild(acceptButton);
                    } else {
                        loadImage(data.task_number);
                    }
                    nextTaskButton.disabled = true;
                }
            });
    });

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', function(event) {
        event.preventDefault();
        startDrawing(event);
    });
    canvas.addEventListener('touchmove', function(event) {
        event.preventDefault();
        draw(event);
    });
    canvas.addEventListener('touchend', function(event) {
        event.preventDefault();
        stopDrawing();
    });
}
