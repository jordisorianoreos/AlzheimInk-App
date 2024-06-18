from flask import Flask, request, jsonify, send_from_directory, render_template, session
import time
import math
import numpy as np
import pandas as pd
import json
from app_predictor_model import app_predictor_model
from datetime import datetime
import csv
import os

# Initialize the App
app = Flask(__name__)

app.secret_key = os.urandom(24)

# Initialize Variables
TABLET_X_SIZE = 21000
TABLET_Y_SIZE = 29700
CANVAS_X_SIZE = 12089
CANVAS_Y_SIZE = 6800
X_MULTIPLIER = CANVAS_X_SIZE / 800
Y_MULTIPLIER = CANVAS_Y_SIZE / 450
canvas_origin_x = (TABLET_X_SIZE - CANVAS_X_SIZE) // 2
CANVAS_ORIGIN_Y = TABLET_Y_SIZE // 2
box_size = 4
d_gmrt = 4
positions = []
distances = []
radii = []
times = []
time_vectors = []
times_drawing = []
canvas_squares = np.zeros((CANVAS_Y_SIZE // box_size, CANVAS_X_SIZE // box_size))
tablet_squares = np.zeros((TABLET_Y_SIZE // box_size, TABLET_X_SIZE // box_size))
coord_right_corner = (21000, 0)
start_time = None
start_time_vector = None
start_time_segment = None
task_number = 1
NUM_TASKS = 19
all_metrics = []
num_of_pendown = 0


def reset_test():
    """
    Responsible for resetting the variables. It is then used when the page is reset.
    """
    global positions, distances, radii, times, time_vectors, times_drawing, \
           canvas_squares, tablet_squares, start_time, start_time_vector, \
           start_time_segment, task_number, all_metrics, num_of_pendown

    positions = []
    distances = []
    radii = []
    times = []
    time_vectors = []
    times_drawing = []
    canvas_squares = np.zeros((CANVAS_Y_SIZE // box_size, CANVAS_X_SIZE // box_size))
    tablet_squares = np.zeros((TABLET_Y_SIZE // box_size, TABLET_X_SIZE // box_size))
    start_time = None
    start_time_vector = None
    start_time_segment = None
    task_number = 1
    all_metrics = []
    num_of_pendown = 0


def validate_dni(dni):
    """It checks that the DNI entered is in the correct format."""
    if len(dni) != 9:
        return False
    if not dni[:8].isdigit():
        return False
    if not dni[-1].isalpha():
        return False
    return True


@app.route('/')
def index():
    """It is in charge of loading the html when the web page is started."""
    return render_template('index.html')


@app.route('/reset_test', methods=['POST'])
def reset_test_route():
    """It resets the variables with the reset test function"""
    reset_test()
    return jsonify({'status': 'Test reset'})


@app.route('/start_drawing', methods=['POST'])
def start_drawing():
    """
    It starts the initial time of the task at the first touch of the screen
    if this action has not already been performed. It also adds 1 to the pendowns counter.
    """
    global start_time, num_of_pendown
    if start_time is None:
        start_time = time.time()
    num_of_pendown += 1
    return jsonify({'status': 'started'})


@app.route('/draw', methods=['POST'])
def draw():
    """This manages the events for the act of drawing on the canvas."""
    global start_time, start_time_vector, start_time_segment

    # The times of Task Start, Vector Start between two points and of a drawing segment
    # without lifting the stroke are initialized.
    if start_time is None:
        start_time = time.time()

    if start_time_vector is None:
        start_time_vector = time.time()

    if start_time_segment is None:
        start_time_segment = time.time()

    # The coordinates where the stylus is drawing are retrieved.
    data = request.json
    x, y = data['x'], data['y']

    # An adjustment is made to the collected size of the task 17
    if task_number == 17:
        big_x = (x * X_MULTIPLIER + canvas_origin_x) * (700/403)
        big_y = (y * Y_MULTIPLIER + CANVAS_ORIGIN_Y) * (700/403)
    else:
        big_x = (x * X_MULTIPLIER + canvas_origin_x)
        big_y = (y * Y_MULTIPLIER + CANVAS_ORIGIN_Y)

    # Stylus positions are stored in a tuple
    positions.append((big_x, big_y))

    # Manages events of the current position with respect to the previous one
    if start_time_vector is not None:
        old_time = start_time_vector
        start_time_vector = time.time()
        time_vector = round((start_time_vector - old_time) * 1000, 8)

        time_vectors.append(time_vector)

        # Distances and writing time are calculated for further calculation of speed_on_paper
        if len(positions) > 1:
            prev_pos = positions[-2]
            distance = math.floor(((big_x - prev_pos[0]) ** 2 + (big_y - prev_pos[1]) ** 2) ** 0.5)
            distances.append(distance*0.55)
            times.append(time_vector)

            # Distances to the top-right corner are calculated for further calculation of GMRT
            right_corner_distance = ((big_x - coord_right_corner[0]) ** 2
                                     + (big_y - coord_right_corner[1]) ** 2) ** 0.5
            radii.append(right_corner_distance)

            # Stores a 1 in the array of the position where it is located for further calculation of Dispersion Index
            x_index = int((x * X_MULTIPLIER * 1) // box_size)
            y_index = int((y * Y_MULTIPLIER * 1) // box_size)
            if 0 <= y_index < canvas_squares.shape[0] and 0 <= x_index < canvas_squares.shape[1]:
                canvas_squares[y_index, x_index] = 1

    return jsonify({'status': 'drawing'})


@app.route('/stop_drawing', methods=['POST'])
def stop_drawing():
    """
    When the pencil is lifted, drawing stops and the time counter of the time_on_paper stops
    and stores the time_on_paper
    """
    global start_time_segment, start_time_vector
    start_time_vector = None

    if start_time_segment is not None:
        elapsed_time = time.time() - start_time_segment
        times_drawing.append(elapsed_time)
        start_time_segment = None

    return jsonify({'status': 'stopped'})


@app.route('/next_task', methods=['POST'])
def next_task():
    """
    Handles events when moving to the next task
    """
    global task_number

    # Calculate and store metrics for the current task
    metrics = calculate_metrics_internal()
    all_metrics.extend(metrics)

    reset_variables()

    # Update task number
    task_number += 1

    # If the tasks have been completed, it returns the status finished
    if task_number > NUM_TASKS:
        return jsonify({'status': 'finished'})

    return jsonify({'status': 'next task', 'task_number': task_number})


def calculate_metrics_internal():
    """
    The different predictor variables are calculated for each task.
    """
    if start_time is not None:
        end_time = time.time()
        total_time = round((end_time - start_time) * 1000)
        paper_time = round(sum(time_vectors))
        air_time = total_time - paper_time
        disp_index = np.sum(canvas_squares) / tablet_squares.size
        gmrt_on_paper = calculate_gmrt()
        max_x_extension, max_y_extension = calculate_extensions()
        total_distance = sum(distances)
        mean_speed_on_paper = total_distance / paper_time

        # The metrics of the task variables are stored in a list to be added
        # to the final list at a later date.
        metrics = [
            air_time, disp_index, gmrt_on_paper, max_y_extension,
            max_x_extension, mean_speed_on_paper, num_of_pendown, paper_time,
        ]

        return metrics


def calculate_gmrt():
    """
    It is responsible for calculating the GMRT from the distances of the write positions to
    the upper right corner
    """
    radii_variation = []
    if len(radii) > d_gmrt:
        for i in range(len(radii)):
            if i >= d_gmrt:
                result = abs(radii[i] - radii[i - d_gmrt + 1])
                radii_variation.append(result)
        return round((1 / len(radii_variation)) * sum(radii_variation), 10)
    else:
        return 0


def calculate_extensions():
    """It calculates the maximum extensions in X and Y from the minima and maxima of each axis"""
    max_x = max([pos[0] for pos in positions]) if positions else 0
    min_x = min([pos[0] for pos in positions]) if positions else 0
    max_x_extension = max_x - min_x
    max_y = max([pos[1] for pos in positions]) if positions else 0
    min_y = min([pos[1] for pos in positions]) if positions else 0
    max_y_extension = max_y - min_y

    return max_x_extension, max_y_extension


def reset_variables():
    """
    Resets the variables for when you want to move on to the next task
    """
    global positions, distances, radii, times, time_vectors, start_time, start_time_vector, \
        start_time_segment, num_of_pendown, canvas_squares
    positions = []
    distances = []
    radii = []
    times = []
    time_vectors = []
    start_time = None
    start_time_vector = None
    start_time_segment = None
    num_of_pendown = 0
    canvas_squares = np.zeros((CANVAS_Y_SIZE // box_size, CANVAS_X_SIZE // box_size))


@app.route('/make_prediction', methods=['POST'])
def make_prediction_route():
    """
    The prediction result is obtained for an unregistered user
    """
    prediction_result, perc_predictors = app_predictor_model(all_metrics)
    return jsonify({'status': 'Prediction made', 'prediction': prediction_result})


@app.route('/make_prediction_patient', methods=['POST'])
def make_prediction_patient_route():
    """
    The prediction result is obtained for a registered user.
    In addition, the results obtained are saved in a csv file.
    """
    prediction_result, perc_predictors = app_predictor_model(all_metrics)

    # Data is created for the table
    dni = session.get('dni')
    time_login = session.get('time_login')

    if prediction_result == 0:
        prediction = "Alzheimer"
    else:
        prediction = "Sano"

    time_end = time.time()
    duration_test = round((time_end - time_login), 2)

    now = datetime.now()
    formatted_time = now.strftime("%d/%b/%Y - %H:%M:%S")

    user_results_test = [dni, formatted_time, duration_test, prediction, perc_predictors, all_metrics]

    # The test result is saved in the file
    with open('databases/database_user_results.csv', 'a', newline='') as database_results:
        writer = csv.writer(database_results)
        writer.writerow(user_results_test)

    return jsonify({'status': 'Prediction made', 'prediction': prediction_result})


@app.route('/image/<task_number>')
def task_image(task_number):
    """Task background images are loaded according to the current task"""
    return send_from_directory('images', f'task{task_number}.png')


@app.route('/register', methods=['POST'])
def register_patient():
    """It is responsible for registering a new user from a form that is filled in.
    If successful it is stored in a json file as a database."""

    patient_data = request.get_json()

    if not patient_data:
        return jsonify({'message': 'No data provided'}), 400

    dni = patient_data.get('dni')

    if not dni:
        return jsonify({'message': 'DNI is required'}), 400

    dni = dni.upper()

    if not validate_dni(dni):
        return jsonify({'message': 'Invalid DNI format'}), 400

    try:
        with open('databases/users.json', 'r+', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if not isinstance(data, dict):
                    return jsonify({'message': 'Invalid users format'}), 500
            except json.JSONDecodeError:
                data = {}

            if dni in data:
                return jsonify({'message': 'DNI already exists'}), 400

            data[dni] = {
                'password': patient_data.get('password'),
                'name': patient_data.get('name'),
                'surname': patient_data.get('surname'),
                'diagnosis': patient_data.get('diagnosis')
            }

            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open('databases/users.json', 'w', encoding='utf-8') as file:
            json.dump({dni: {
                'name': patient_data.get('name'),
                'surname': patient_data.get('surname'),
                'diagnosis': patient_data.get('diagnosis')
            }}, file, ensure_ascii=False, indent=4)

    return jsonify({'message': 'Paciente registrado exitosamente'})


@app.route('/login', methods=['POST'])
def login():
    """Login as administrator is managed"""
    credentials = request.get_json()
    dni = credentials.get('dni').upper()
    password = credentials.get('password')
    try:
        with open('databases/admins.json', 'r') as file:
            admins = json.load(file)
            if not isinstance(admins, list):
                return jsonify({'message': 'Invalid admins format'}), 500
            admin = next((admin for admin in admins if admin['dni'] == dni and admin['password'] == password), None)
            if admin:
                return jsonify({'message': 'Login successful'})
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
    except FileNotFoundError:
        return jsonify({'message': 'Admins file not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'message': 'Error decoding admins file'}), 500


@app.route('/patient_login', methods=['POST'])
def patient_login():
    """Login as user/patient is managed"""
    credentials = request.get_json()
    dni = credentials.get('dni').upper()
    password = credentials.get('password')
    try:
        with open('databases/users.json', 'r', encoding='utf-8') as file:
            users = json.load(file)
            user = users.get(dni)
            if user and user.get('password') == password:
                session['dni'] = dni
                session['time_login'] = time.time()
                return jsonify({'message': 'Login successful', 'name': user.get('name')})
            else:
                return jsonify({'message': 'Invalid DNI or password'}), 401
    except FileNotFoundError:
        return jsonify({'message': 'Users file not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'message': 'Error decoding users file'}), 500


@app.route('/consult', methods=['POST'])
def consult_patient():
    """This searches if a user exists in the database and if so, a json is returned confirming it"""
    credentials = request.get_json()
    dni = credentials.get('dni').upper()

    if not dni:
        return jsonify({'message': 'DNI is required'}), 400

    if not validate_dni(dni):
        return jsonify({'message': 'Invalid DNI format'}), 400

    try:
        with open('databases/users.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            patient = data.get(dni)
            if patient:
                return jsonify(patient)
            else:
                return jsonify({'message': 'Patient not found'}), 404
    except FileNotFoundError:
        return jsonify({'message': 'Users file not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'message': 'Error decoding users file'}), 500


@app.route('/get_table/<dni>', methods=['GET'])
def get_table(dni):
    """It creates a table with the test records based on the user’s DNI number that is
    to be queried in the database. This is then sent to be displayed in the app."""

    # CSV is read in a pandas DataFrame
    df = pd.read_csv('databases/database_user_results.csv', encoding='ISO-8859-1')

    # Modifying the data frame and the most recent first
    df = df.iloc[::-1]
    df = df.drop(df.columns[-1], axis=1)
    columns = ["DNI", "Fecha", "Duración Test (s)", "Predicción", "% Predictores EA"]
    df.columns = columns

    # The DataFrame is filtered to include only the rows that correspond to the DNI provided.
    df = df[df['DNI'] == dni]

    html_table = df.to_html(index=False)

    return html_table


# OTHER FUNCTIONS TO ADD DATA TO NEW DATABASE FOR TRAINING
"""
The following functions are practically identical to the others, only small modifications have 
been made so that the data is collected independently of the graphic tablet from the original study.
"""

def reset_test_add_data():
    global positions, distances, radii, times, time_vectors, times_drawing, \
           canvas_squares, start_time, start_time_vector, \
           start_time_segment, task_number, all_metrics, num_of_pendown, coord_right_corner

    positions = []
    distances = []
    radii = []
    times = []
    time_vectors = []
    times_drawing = []
    canvas_squares = np.zeros((CANVAS_Y_SIZE // box_size, CANVAS_X_SIZE // box_size))
    start_time = None
    start_time_vector = None
    start_time_segment = None
    task_number = 1
    all_metrics = []
    num_of_pendown = 0
    coord_right_corner = (CANVAS_X_SIZE, 0)


@app.route('/reset_test_add_data', methods=['POST'])
def reset_test_route_add_data():
    reset_test_add_data()
    return jsonify({'status': 'Test reset'})


@app.route('/start_drawing_add_data', methods=['POST'])
def start_drawing_add_data():
    global start_time, start_time_vector, start_time_segment, num_of_pendown
    if start_time is None:
        start_time = time.time()
    num_of_pendown += 1
    return jsonify({'status': 'started'})


@app.route('/draw_add_data', methods=['POST'])
def draw_add_data():
    global start_time, start_time_vector, start_time_segment

    if start_time is None:
        start_time = time.time()

    if start_time_vector is None:
        start_time_vector = time.time()

    if start_time_segment is None:
        start_time_segment = time.time()

    data = request.json
    x, y = data['x'], data['y']

    big_x = (x * X_MULTIPLIER)
    big_y = (y * Y_MULTIPLIER)

    positions.append((big_x, big_y))

    if start_time_vector is not None:
        old_time = start_time_vector
        start_time_vector = time.time()
        time_vector = round((start_time_vector - old_time) * 1000, 8)

        time_vectors.append(time_vector)

        if len(positions) > 1:
            prev_pos = positions[-2]
            distance = ((big_x - prev_pos[0]) ** 2 + (big_y - prev_pos[1]) ** 2) ** 0.5
            distances.append(distance)
            times.append(time_vector)

            right_corner_distance = ((big_x - coord_right_corner[0]) ** 2
                                     + (big_y - coord_right_corner[1]) ** 2) ** 0.5
            radii.append(right_corner_distance)

            x_index = int((x * X_MULTIPLIER * 1) // box_size)
            y_index = int((y * Y_MULTIPLIER * 1) // box_size)
            if 0 <= y_index < canvas_squares.shape[0] and 0 <= x_index < canvas_squares.shape[1]:
                canvas_squares[y_index, x_index] = 1

    return jsonify({'status': 'drawing'})


@app.route('/stop_drawing_add_data', methods=['POST'])
def stop_drawing_add_data():
    global start_time_segment, start_time_vector
    start_time_vector = None

    if start_time_segment is not None:
        elapsed_time = time.time() - start_time_segment
        times_drawing.append(elapsed_time)
        start_time_segment = None

    return jsonify({'status': 'stopped'})


def reset_variables_add_data():
    global positions, distances, radii, times, time_vectors, start_time, start_time_vector, \
        start_time_segment, num_of_pendown, canvas_squares
    positions = []
    distances = []
    radii = []
    times = []
    time_vectors = []
    start_time = None
    start_time_vector = None
    start_time_segment = None
    num_of_pendown = 0
    canvas_squares = np.zeros((CANVAS_Y_SIZE // box_size, CANVAS_X_SIZE // box_size))


@app.route('/next_task_add_data', methods=['POST'])
def next_task_add_data():
    global task_number

    # Calculate and store metrics for the current task
    metrics = calculate_metrics_internal_add_data()
    all_metrics.extend(metrics)

    reset_variables_add_data()

    # Update task number
    task_number += 1
    if task_number > NUM_TASKS:
        return jsonify({'status': 'finished'})

    return jsonify({'status': 'next task', 'task_number': task_number})


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    global all_metrics
    return jsonify(all_metrics)


def calculate_gmrt_add_data():
    radii_variation = []
    if len(radii) > d_gmrt:
        for i in range(len(radii)):
            if i >= d_gmrt:
                result = abs(radii[i] - radii[i - d_gmrt + 1])
                radii_variation.append(result)
        return round((1 / len(radii_variation)) * sum(radii_variation), 10)
    else:
        return 0


def calculate_metrics_internal_add_data():
    if start_time is not None:
        end_time = time.time()
        total_time = round((end_time - start_time) * 1000)
        paper_time = round(sum(time_vectors))
        air_time = total_time - paper_time
        disp_index = np.sum(canvas_squares) / canvas_squares.size
        gmrt_on_paper = calculate_gmrt_add_data()
        max_x_extension, max_y_extension = calculate_extensions()
        total_distance = sum(distances)
        mean_speed_on_paper = total_distance / paper_time

        metrics = [
            air_time, disp_index, gmrt_on_paper, max_y_extension,
            max_x_extension, mean_speed_on_paper, num_of_pendown, paper_time,
        ]

        return metrics


@app.route('/add_data_login', methods=['POST'])
def add_data_login():
    credentials = request.get_json()
    dni = credentials.get('dni').upper()
    diagnosis = credentials.get('diagnosis')

    if not validate_dni(dni):
        return jsonify({'message': 'Invalid DNI format'}), 400
    else:
        session['dni'] = dni

    if diagnosis not in [0, 1]:
        return jsonify({'message': 'Please, select a diagnosis'}), 400
    else:
        session['diagnosis'] = diagnosis

    return jsonify({'message': 'Login successful'})


@app.route('/save_data_new_database', methods=['POST'])
def save_data_new_database():
    dni = session.get('dni')
    diagnosis = session.get('diagnosis')
    id_hour = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    id_test = f"{dni}_{id_hour}"
    if diagnosis == 1:
        diagnosis_letter = "P"
    else:
        diagnosis_letter = "H"
    new_data_user = [id_test] + all_metrics + [diagnosis_letter]
    filename = 'databases/new_database_training.csv'

    if not os.path.isfile(filename):
        variables = ['air_time', 'disp_index', 'gmrt_on_paper', 'max_x_extension', 'max_y_extension',
                     'mean_speed_on_paper', 'num_of_pendown', 'paper_time']
        all_variables = [f"{item}{i}" for i in range(1, 20) for item in variables]
        with open(filename, 'w', newline='') as database_results:
            writer = csv.writer(database_results)
            writer.writerow(all_variables)

    with open(filename, 'a', newline='') as database_results:
        writer = csv.writer(database_results)
        writer.writerow(new_data_user)

    return jsonify({'status': 'Data saved'})


# Starts Flask server on port 5000
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
