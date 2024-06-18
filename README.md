# AlzheimInk

AlzheimInk is a mobile web application designed to predict the diagnosis of Alzheimer's disease through handwriting analysis. The application conducts a test that includes various writing and drawing tasks that the user performs with a capacitive stylus. Once the test is completed, the application uses a machine learning model to process the collected data and predict whether the user has Alzheimer's disease.

## Features

- Conducts a handwriting and drawing test with a capacitive stylus
- Uses machine learning to analyze test data and predict Alzheimer's disease
- Allows user registration and viewing of registered user data
- Stores and reviews previously conducted tests
- Collects data to create a new database for more accurate predictive models

## Repository Contents

This repository contains the application and related code for the project. The most relevant files are:

- **app.py**: Contains the application developed using Flask. To run the application locally, this file must be executed.
- **models.py**: Includes everything related to the creation, training, validation, testing, and other functions of the various developed machine learning models.
- **app_predictor_model.py**: Contains the specific function chosen for making predictions in the application and everything it needs to operate.
- **results_analysis.py**: Covers everything related to the analysis of results of the project, including graphs and tables.
- **index.html, script.js, canvasFunctions.js, and styles.css**: Located within the templates and static folders, these files manage the front-end of the web application.

## Important Note

It's important to remember that, while the application offers a diagnostic prediction, these predictions can be inaccurate and should never replace the diagnosis of a medical professional.
