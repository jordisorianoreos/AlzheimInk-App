// Importing canvas functions
import {showCanvas_add_data, showCanvas_default, showCanvas_patient} from './canvasFunctions.js';

// Event listeners for showing canvas
document.getElementById('showCanvasButton').addEventListener('click', showCanvas_default);

// Event listeners for login options
document.getElementById('loginButton').addEventListener('click', function() {
    document.getElementById('loginOptions').style.display = 'block';
    document.getElementById('loginButton').style.display = 'none';
    document.getElementById('showCanvasButton').style.display = 'none';
    document.getElementById('image').style.width = '20%';
});

document.getElementById('adminLogin').addEventListener('click', function() {
    document.getElementById('loginForm').style.display = 'block';
    document.getElementById('loginOptions').style.display = 'none';
    document.getElementById('image').style.width = '20%';
});

document.getElementById('patientLogin').addEventListener('click', function() {
    document.getElementById('patientLoginForm').style.display = 'block';
    document.getElementById('loginOptions').style.display = 'none';
    document.getElementById('image').style.width = '20%';
});

document.getElementById('addDataLogin').addEventListener('click', function() {
    document.getElementById('addDataLoginForm').style.display = 'block';
    document.getElementById('adminOptions').style.display = 'none';
});

// Event listeners for form submissions
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var dni = document.getElementById('dni_admin').value;
    var password = document.getElementById('password_admin').value;
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dni: dni, password: password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'Login successful') {
            document.getElementById('adminOptions').style.display = 'block';
            document.getElementById('loginForm').style.display = 'none';
        } else {
            alert('ERROR: DNI o contraseña incorrectos');
        }
    })
    .catch(error => console.error('Error:', error));
});

document.getElementById('patientLoginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var dni = document.getElementById('dni_patient').value;
    var password = document.getElementById('password_patient').value;
    fetch('/patient_login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dni: dni, password: password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'Login successful') {
            document.getElementById('patientLoginForm').style.display = 'none';
            showCanvas_patient()
        } else {
            alert('ERROR: DNI o contraseña incorrectos');
        }
    })
    .catch(error => console.error('Error:', error));
});

document.getElementById('addDataLoginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var dni = document.getElementById('dni_add_data').value;
    var diagnosis = parseInt(document.getElementById('diagnosis_add_data').value);
    fetch('/add_data_login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dni: dni, diagnosis: diagnosis })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'Login successful') {
            document.getElementById('addDataLoginForm').style.display = 'none';
            showCanvas_add_data()
        } else {
            alert('ERROR: ' + data.message);
        }
    })
    .catch(error => console.error('Error:', error));
});

// Event listeners for admin options
document.getElementById('registerPatientButton').addEventListener('click', function() {
    document.getElementById('registerForm').style.display = 'block';
    document.getElementById('adminOptions').style.display = 'none';
});

document.getElementById('registerForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const formDataObj = Object.fromEntries(formData.entries());

    const response = await fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formDataObj)
    });

    const result = await response.json();
    alert(result.message);
});

document.getElementById('consultPatientButton').addEventListener('click', function() {
    document.getElementById('consultForm').style.display = 'block';
    document.getElementById('adminOptions').style.display = 'none';
});

document.getElementById('consultForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var dni = document.getElementById('dni_consult').value;
    fetch('/consult', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dni: dni })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            alert(data.message);
        } else {
            document.getElementById('patientName').textContent = 'Nombre: ' + data.name;
            document.getElementById('patientSurname').textContent = 'Apellidos: ' + data.surname;
            document.getElementById('patientDiagnosis').textContent = 'Diagnóstico confirmado: ' + data.diagnosis;

            // Makes a GET request to the new path to get the HTML table
            fetch('/get_table/' + dni)
            .then(response => response.text())
            .then(html_table => {
                // HTML table is added
                document.getElementById('patientTable').innerHTML = html_table;
                document.getElementById('patientInfo').style.display = 'block';
            });
        }
    })
    .catch(error => console.error('Error:', error));
});
