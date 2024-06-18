import pandas as pd
import numpy as np
import joblib
import random
import os
import matplotlib.pyplot as plt
import warnings
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm as SuportVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# CONTROL PANEL
NUM_TASKS = 19
k_fold = 5
num_random_splits = 20
num_variables_bytask = 8
rerun_data_split = False
rerun_models_training = False
warnings.filterwarnings("ignore")

# Variables are initialized
random.seed(0)
split_seeds = random.sample(range(1, 10000), num_random_splits)
models_performance_metrics = []
allmodels_performance_metrics = []
allmodels_performance_metrics_bytask = []
best_models_bytask = []
my_scaler = None
prediction_singleclassifier = None
prediction_multiclassifier = None

# Grid of Hyperparameters
param_grids = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 150],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 3],
        "bootstrap": [True, False]
    },
    "LogisticRegression": {
        "C": [round(x, 3) for x in np.arange(0.001, 5.005, 0.01)],
        "max_iter": [800],
        "solver": ["liblinear", "lbfgs"]
    },
    "KNeighborsClassifier": {
        "n_neighbors": list(range(3, 50, 5)),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"]
    },
    "LinearDiscriminantAnalysis": {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": [None, "auto"]
    },
    "GaussianNB": {
        "priors": [None, [[round(x, 1), round(1-x, 1)] for x in np.arange(0.0, 1.1, 0.1)]]
    },
    "SVC": {
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "C": [round(x, 1) for x in np.arange(0.1, 1.6, 0.05)],
        "gamma": ["scale", "auto", 0.5]
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 2, 5, 10],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_leaf_nodes": [None, 2, 5, 10]
    },
    "MLPClassifier": {
        "activation": ["relu", "logistic"],
        "hidden_layer_sizes": [5, 15, 30, 100],
        "max_iter": [1000],
        "solver": ["sgd", "adam"],
    },
    "XGBClassifier": {
        "max_depth": [4, 6, 10],
        "eta": [0.1, 0.3],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "colsample_bynode": [0.7, 1.0],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 150, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 150],
        "subsample": [0.7, 1.0],
        "max_depth": [None, 3],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [50, 100],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 3],
        "bootstrap": [True, False],
    }
}


# Main Function
def make_models():
    """
    This is the main function with which all functions are executed according to the settings
    in the control panel
    """
    if rerun_data_split:
        preprocess_data()
    if rerun_models_training:
        rerun_the_models()
    else:
        top_n_single_classifiers(num_of_top_models=3)
        top_n_single_classifiers(num_of_top_models=5)
        top_n_mixed_classifiers(num_of_top_models=3)
        top_n_mixed_classifiers(num_of_top_models=5)
        single_classifier_task_specific()
        mixed_classifier_task_specific()


def preprocess_data():
    dataframe = pd.read_csv('data.csv')

    # Next, both variables that could not be collected and tasks that have not
    # been decided not to be performed are removed from the dataset
    columns_to_delete = []
    excluded_tasks = [7, 17, 19, 20, 23, 25]
    for i in range(1, 26):
        if i not in excluded_tasks:
            to_delete = [
                f"gmrt_in_air{i}", f"mean_acc_in_air{i}", f"mean_gmrt{i}",
                f"mean_acc_on_paper{i}", f"mean_jerk_on_paper{i}", f"total_time{i}",
                f"mean_jerk_in_air{i}", f"mean_speed_in_air{i}", f"pressure_mean{i}",
                f"pressure_var{i}"
            ]
            columns_to_delete.extend(to_delete)
        else:
            to_delete = [
                f"air_time{i}", f"disp_index{i}", f"gmrt_in_air{i}", f"gmrt_on_paper{i}",
                f"max_x_extension{i}", f"max_y_extension{i}", f"mean_acc_in_air{i}",
                f"mean_acc_on_paper{i}", f"mean_gmrt{i}", f"mean_jerk_in_air{i}",
                f"mean_jerk_on_paper{i}", f"mean_speed_in_air{i}", f"mean_speed_on_paper{i}",
                f"num_of_pendown{i}", f"paper_time{i}", f"pressure_mean{i}",
                f"pressure_var{i}", f"total_time{i}"
            ]
            columns_to_delete.extend(to_delete)
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    dataframe.to_csv("dataframe_clean.csv", index=False)

    # The labels are managed and set as 0 and 1
    dataframe = dataframe.to_numpy()
    data = dataframe[:, 1:-1]
    labels = dataframe[:, -1]
    choose_scaler(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Data partitioning in train and test is performed for each random split
    for i in range(1, len(split_seeds) + 1):
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                            test_size=0.25,
                                                                            random_state=split_seeds[i-1],
                                                                            stratify=labels)
        os.makedirs(f'runs/{i}/splits', exist_ok=True)

        # Saves training and test data sets in files inside folders
        joblib.dump(data_train, f'runs/{i}/splits/data_train_{i}.pkl')
        joblib.dump(data_test, f'runs/{i}/splits/data_test_{i}.pkl')
        joblib.dump(labels_train, f'runs/{i}/splits/labels_train_{i}.pkl')
        joblib.dump(labels_test, f'runs/{i}/splits/labels_test_{i}.pkl')

        # Saves training and test data sets by task in files inside folders
        for task in range(NUM_TASKS):
            start_col = task * num_variables_bytask
            end_col = start_col + num_variables_bytask
            data_train_subset = data_train[:, start_col:end_col]
            data_test_subset = data_test[:, start_col:end_col]

            os.makedirs(f'runs/{i}/by_tasks/task_{task + 1}/splits', exist_ok=True)
            joblib.dump(data_train_subset, f'runs/{i}/by_tasks/task_{task + 1}/splits/data_train_{i}_{task + 1}.pkl')
            joblib.dump(data_test_subset, f'runs/{i}/by_tasks/task_{task + 1}/splits/data_test_{i}_{task + 1}.pkl')
            joblib.dump(labels_train, f'runs/{i}/by_tasks/task_{task + 1}/splits/labels_train_{i}_{task + 1}.pkl')
            joblib.dump(labels_test, f'runs/{i}/by_tasks/task_{task + 1}/splits/labels_test_{i}_{task + 1}.pkl')


def choose_scaler(data):

    global my_scaler
    count_gaussian = 0
    count_not_gaussian = 0
    for i in range(data.shape[1]):
        stat, p = shapiro(data[:, i])
        if p > 0.05:
            count_gaussian += 1
        else:
            count_not_gaussian += 1
    if count_gaussian > count_not_gaussian:
        my_scaler = StandardScaler()
    else:
        my_scaler = MinMaxScaler()


def rerun_the_models():
    """It is in charge of retraining the models, as well as testing and
     saving the performance metrics in a csv file
     It also performs other functions to create additional models"""
    global models_performance_metrics

    print("Starting Training...")
    models = [RandomForestClassifier(), LogisticRegression(), KNeighborsClassifier(),
              LinearDiscriminantAnalysis(), GaussianNB(), SuportVM.SVC(),
              DecisionTreeClassifier(), MLPClassifier(), XGBClassifier(),
              AdaBoostClassifier(), GradientBoostingClassifier(), ExtraTreesClassifier()]

    os.makedirs(f"performance_metrics", exist_ok=True)

    # Performance Metrics Single-Classifier Models
    for model in models:
        for run in range(1, len(split_seeds) + 1):
            print(f"Model: {model}, Run: {run}")
            evaluate_model(model, run)

    dataframe_metrics = pd.DataFrame(allmodels_performance_metrics)
    dataframe_metrics.columns = ["name_model", "classifier", "run", "accuracy", "sensitivity",
                                 "specificity", "precision", "recall", "f1", "kappa"]
    dataframe_metrics.to_csv(f"performance_metrics/single_classifiers.csv", index=False)

    # Performance Metrics Single-Classifier Task Specific Models
    for task in range(1, NUM_TASKS + 1):
        for model in models:
            for run in range(1, len(split_seeds) + 1):
                print(f"Task: {task}, Model: {model}, Run: {run}")
                evaluate_model_bytasks(task, model, run)

    dataframe_bytask_metrics = pd.DataFrame(allmodels_performance_metrics_bytask)
    dataframe_bytask_metrics.columns = ["name_model", "classifier", "run", "task", "accuracy",
                                        "sensitivity", "specificity", "precision", "recall",
                                        "f1", "kappa"]
    dataframe_bytask_metrics.to_csv(f"performance_metrics/allmodels_bytask_metrics.csv", index=False)

    # The functions for creating the additional models are executed
    top_n_single_classifiers(num_of_top_models=3)
    top_n_single_classifiers(num_of_top_models=5)
    top_n_mixed_classifiers(num_of_top_models=3)
    top_n_mixed_classifiers(num_of_top_models=5)
    single_classifier_task_specific()
    mixed_classifier_task_specific()


def calculate_metrics(labels_test, predictions_model):
    """
    It is responsible for calculating all performance metrics
    based on predictions and labels, and then creating a dictionary with these
    """
    tn, fp, fn, tp = confusion_matrix(labels_test, predictions_model).ravel()
    total_predictions = tp + tn + fp + fn
    accuracy = (tp + tn) / total_predictions
    sensitivity = tp / (tp + fn)  # or recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # or sensitivity
    f1 = 2 * (precision * recall) / (precision + recall)
    observed_accuracy = (tp + tn) / total_predictions
    expected_accuracy = (((tp + fp) / total_predictions) * ((tp + fn) / total_predictions) +
                         ((tn + fp) / total_predictions) * ((tn + fn) / total_predictions))
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "recall": recall, "f1": f1, "kappa": kappa}


def evaluate_model(model, run):
    """
    It handles hyperparameter search, cross-validation, scaling, testing,
    and storage of metrics for single-classifier models
    """
    data_train = joblib.load(f'runs/{run}/splits/data_train_{run}.pkl')
    data_test = joblib.load(f'runs/{run}/splits/data_test_{run}.pkl')
    labels_train = joblib.load(f'runs/{run}/splits/labels_train_{run}.pkl')
    labels_test = joblib.load(f'runs/{run}/splits/labels_test_{run}.pkl')

    name_model = str(model).split("(")[0]

    pipe = Pipeline([("scaler", my_scaler), ("model", model)])

    param_grid = param_grids[name_model]
    param_grid = {'model__' + key: value for key, value in param_grid.items()}

    GS = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      scoring=["accuracy", "precision"],
                      refit="accuracy",
                      cv=k_fold,
                      verbose=False)
    GS.fit(data_train, labels_train)
    best_model = GS.best_estimator_

    best_model.fit(data_train, labels_train)
    predictions_model = best_model.predict(data_test)

    confusion_matrix_model = confusion_matrix(labels_test, predictions_model)
    disp = ConfusionMatrixDisplay(confusion_matrix_model, display_labels=["Healthy", "Alzheimer"])
    disp.plot()
    plt.title(f"{name_model}_{run} model")

    os.makedirs(f"performance_metrics/confusion_matrices/{name_model}", exist_ok=True)
    plt.savefig(f"performance_metrics/confusion_matrices/{name_model}/{name_model}_{run}.png") # Guarda CrossTable

    os.makedirs(f"runs/{run}/models", exist_ok=True)
    joblib.dump(best_model, filename=f"runs/{run}/models/{name_model}_{run}.pkl") # Guarda el Modelo
    # plt.show()

    metrics_dict = calculate_metrics(labels_test, predictions_model)
    metrics = [f"{name_model}_{run}", f"{name_model}", f"{run}", metrics_dict["accuracy"],
               metrics_dict["sensitivity"], metrics_dict["specificity"], metrics_dict["precision"],
               metrics_dict["recall"], metrics_dict["f1"], metrics_dict["kappa"]]
    allmodels_performance_metrics.append(metrics)


def evaluate_model_bytasks(task, model, run):
    """
    It handles hyperparameter search, cross-validation, scaling, testing,
    and storage of metrics for task-specific classifier models
    """
    data_train = joblib.load(f'runs/{run}/by_tasks/task_{task}/splits/data_train_{run}_{task}.pkl')
    data_test = joblib.load(f'runs/{run}/by_tasks/task_{task}/splits/data_test_{run}_{task}.pkl')
    labels_train = joblib.load(f'runs/{run}/by_tasks/task_{task}/splits/labels_train_{run}_{task}.pkl')
    labels_test = joblib.load(f'runs/{run}/by_tasks/task_{task}/splits/labels_test_{run}_{task}.pkl')

    name_model = str(model).split("(")[0]

    pipe = Pipeline([("scaler", my_scaler), ("model", model)])

    param_grid = param_grids[name_model]
    param_grid = {'model__' + key: value for key, value in param_grid.items()}

    GS = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      scoring=["accuracy", "precision"],
                      refit="accuracy",
                      cv=k_fold,
                      verbose=False)
    GS.fit(data_train, labels_train)
    best_model = GS.best_estimator_

    best_model.fit(data_train, labels_train)
    predictions_model = best_model.predict(data_test)

    confusion_matrix_model = confusion_matrix(labels_test, predictions_model)
    disp = ConfusionMatrixDisplay(confusion_matrix_model, display_labels=["Healthy", "Alzheimer"])
    disp.plot()
    plt.title(f"{name_model}_{run}_task{task} model")

    os.makedirs(f"performance_metrics/confusion_matrices/by_tasks/{name_model}", exist_ok=True)
    plt.savefig(f"performance_metrics/confusion_matrices/by_tasks/{name_model}/{name_model}_{run}_task{task}.png")

    os.makedirs(f"runs/{run}/by_tasks/task_{task}/models", exist_ok=True)
    joblib.dump(best_model, filename=f"runs/{run}/by_tasks/task_{task}/models/{name_model}_{run}_{task}.pkl")

    metrics_dict = calculate_metrics(labels_test, predictions_model)
    metrics = [f"{name_model}_{run}_{task}", f"{name_model}",f"{run}", f"{task}", metrics_dict["accuracy"],
               metrics_dict["sensitivity"], metrics_dict["specificity"], metrics_dict["precision"],
               metrics_dict["recall"], metrics_dict["f1"], metrics_dict["kappa"]]
    allmodels_performance_metrics_bytask.append(metrics)


def top_n_single_classifiers(num_of_top_models=3):
    """Top N single classifiers are made from single classifiers where the N models are
    of the same classifier type"""
    global prediction_singleclassifier
    num_of_top_models = num_of_top_models
    all_metrics = []
    classifiers = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier',
                   'LinearDiscriminantAnalysis', 'GaussianNB', 'SVC',
                   'DecisionTreeClassifier', 'MLPClassifier', 'ExtraTreesClassifier',
                   'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']

    df_metrics = pd.read_csv("performance_metrics/single_classifiers.csv")
    df_metrics.sort_values("f1", ascending=False, inplace=True)

    for classifier in classifiers:
        print(f"Making {classifier} predictions...")
        for run in range(1, num_random_splits + 1):
            predictions_by_bestmodels = []
            for i in range(num_of_top_models):
                df_metrics_classifier = df_metrics[df_metrics["classifier"] == classifier]
                best_model_name = df_metrics_classifier["name_model"].iloc[i]
                best_model_run = df_metrics_classifier["run"].iloc[i]
                best_model = joblib.load(f"runs/{best_model_run}/models/{best_model_name}.pkl")

                data_train = joblib.load(f'runs/{run}/splits/data_train_{run}.pkl')
                data_test = joblib.load(f'runs/{run}/splits/data_test_{run}.pkl')
                labels_train = joblib.load(f'runs/{run}/splits/labels_train_{run}.pkl')
                labels_test = joblib.load(f'runs/{run}/splits/labels_test_{run}.pkl')

                best_model.fit(data_train, labels_train)
                predictions_model = best_model.predict(data_test)
                predictions_by_bestmodels.append(predictions_model)

            df = pd.DataFrame(predictions_by_bestmodels).transpose()
            columns = [f"model_{i}" for i in range(1, num_of_top_models + 1)]
            indexes = [f"{i}" for i in range(1, len(labels_test) + 1)]
            df.columns = columns
            df.index = indexes

            sum_byrows = df.sum(axis=1).to_list()

            predictions_top_classifiers = []
            for sum_byrow in sum_byrows:
                if sum_byrow > (num_of_top_models // 2):
                    predictions_top_classifiers.append(1)
                else:
                    predictions_top_classifiers.append(0)

            metrics_dict = calculate_metrics(labels_test, predictions_top_classifiers)

            metrics = [f"top{num_of_top_models}_{classifier}_voting_models_{run}", f"{classifier}", f"{run}", metrics_dict["accuracy"],
                       metrics_dict["sensitivity"], metrics_dict["specificity"], metrics_dict["precision"],
                       metrics_dict["recall"], metrics_dict["f1"], metrics_dict["kappa"]]
            all_metrics.append(metrics)

    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_metrics = df_all_metrics.reset_index(drop=True)
    df_all_metrics.columns = ["name_model", "classifier", "run", "accuracy", "sensitivity",
                              "specificity", "precision", "recall", "f1", "kappa"]

    df_all_metrics.to_csv(f"performance_metrics/top{num_of_top_models}_single_classifiers_voting_models.csv", index=False)


def top_n_mixed_classifiers(num_of_top_models=3):
    """Top N mixed classifiers are made from single classifiers where the N models are
    the best regardless of the type of classifiers that comprise it"""
    all_metrics = []
    num_of_top_models = num_of_top_models

    df_metrics = pd.read_csv("performance_metrics/single_classifiers.csv")
    df_metrics.sort_values("f1", ascending=False, inplace=True)

    print(f"Making {num_random_splits} mixed classifiers predictions...\n")
    for run in range(1, num_random_splits + 1):
        predictions_by_bestmodels = []
        for i in range(num_of_top_models):
            best_model_name = df_metrics["name_model"].iloc[i]
            best_model_run = df_metrics["run"].iloc[i]
            best_model = joblib.load(f"runs/{best_model_run}/models/{best_model_name}.pkl")
            data_train = joblib.load(f'runs/{run}/splits/data_train_{run}.pkl')
            data_test = joblib.load(f'runs/{run}/splits/data_test_{run}.pkl')
            labels_train = joblib.load(f'runs/{run}/splits/labels_train_{run}.pkl')
            labels_test = joblib.load(f'runs/{run}/splits/labels_test_{run}.pkl')

            best_model.fit(data_train, labels_train)
            predictions_model = best_model.predict(data_test)

            predictions_by_bestmodels.append(predictions_model)

        df = pd.DataFrame(predictions_by_bestmodels).transpose()
        columns = [f"model_{i}" for i in range(1, num_of_top_models + 1)]
        indexes = [f"{i}" for i in range(1, len(labels_test) + 1)]
        df.columns = columns
        df.index = indexes

        sum_byrows = df.sum(axis=1).to_list()

        predictions_top_classifiers = []
        for sum_byrow in sum_byrows:
            if sum_byrow > (num_of_top_models // 2):
                predictions_top_classifiers.append(1)
            else:
                predictions_top_classifiers.append(0)

        metrics_dict = calculate_metrics(labels_test, predictions_top_classifiers)

        metrics = [f"top{num_of_top_models}_classifiers_voting_models_{run}", f"{run}", metrics_dict["accuracy"],
                   metrics_dict["sensitivity"], metrics_dict["specificity"], metrics_dict["precision"],
                   metrics_dict["recall"], metrics_dict["f1"], metrics_dict["kappa"]]
        all_metrics.append(metrics)

    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_metrics = df_all_metrics.reset_index(drop=True)
    df_all_metrics.columns = ["name_model", "run", "accuracy", "sensitivity",
                              "specificity", "precision", "recall", "f1", "kappa"]

    df_all_metrics.to_csv(f"performance_metrics/top{num_of_top_models}_mixed_classifiers_voting_models.csv", index=False)


def single_classifier_task_specific():
    """It creates the combination models consisting of 19 smaller task-specific
    models of the same classifier type"""
    all_metrics = []
    classifiers = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier',
                   'LinearDiscriminantAnalysis', 'GaussianNB', 'SVC',
                   'DecisionTreeClassifier', 'MLPClassifier', 'ExtraTreesClassifier',
                   'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']

    for run in range(1, num_random_splits + 1):
        for classifier in classifiers:
            predictions_model_bytask = []
            for task in range(1, NUM_TASKS + 1):
                data_train = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/data_train_{run}_{task}.pkl")
                data_test = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/data_test_{run}_{task}.pkl")
                labels_train = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/labels_train_{run}_{task}.pkl")
                labels_test = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/labels_test_{run}_{task}.pkl")

                model = joblib.load(f"runs/{run}/by_tasks/task_{task}/models/{classifier}_{run}_{task}.pkl")
                model.fit(data_train, labels_train)

                prediction_model = model.predict(data_test)
                predictions_model_bytask.append(prediction_model)

            df = pd.DataFrame(predictions_model_bytask).transpose()
            columns = [f"task_{i}" for i in range(1, NUM_TASKS + 1)]
            indexes = [f"{i}" for i in range(1, len(labels_test) + 1)]
            df.columns = columns
            df.index = indexes

            sum_byrows = df.sum(axis=1).to_list()

            predictions_multiclassifier_byclassifier = []
            for sum_byrow in sum_byrows:
                if sum_byrow > (NUM_TASKS // 2):
                    predictions_multiclassifier_byclassifier.append(1)
                else:
                    predictions_multiclassifier_byclassifier.append(0)

            metrics_dict = calculate_metrics(labels_test, predictions_multiclassifier_byclassifier)

            metrics = [f"single_{classifier}_task_specific_run{run}", f"{classifier}", f"{run}", metrics_dict["accuracy"],
                       metrics_dict["sensitivity"], metrics_dict["specificity"], metrics_dict["precision"],
                       metrics_dict["recall"], metrics_dict["f1"], metrics_dict["kappa"]]
            all_metrics.append(metrics)

    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_metrics = df_all_metrics.reset_index(drop=True)
    df_all_metrics.columns = ["name_model", "classifier", "run", "accuracy", "sensitivity",
                              "specificity", "precision", "recall", "f1", "kappa"]

    df_all_metrics.to_csv("performance_metrics/single_classifier_task_specific.csv", index=False)


def select_best_model_bytasks():
    """The best models are selected by task, in total 19"""
    df_metrics = pd.read_csv("performance_metrics/allmodels_bytask_metrics.csv")
    for task in range(1, NUM_TASKS + 1):
        df_task = df_metrics[df_metrics["task"] == task]
        df_task_sorted = df_task.sort_values(by="f1", ascending=False)
        best_model_name = df_task_sorted["name_model"].iloc[0]
        best_model_run = df_task_sorted["run"].iloc[0]
        best_model_accuracy = df_task_sorted["accuracy"].iloc[0]
        best_model = joblib.load(f"runs/{best_model_run}/by_tasks/task_{task}/models/{best_model_name}.pkl")
        print(f"Task {task}: {best_model_name} --> {round(best_model_accuracy*100,2)}% accuracy")
        best_models_bytask.append(best_model)
    return best_models_bytask


def mixed_classifier_task_specific():
    """It creates the combination models consisting of the best 19 smaller task-specific
    models regardless of the type of classifiers that comprise it"""
    random.seed(123)
    all_metrics = []
    best_models_bytask = select_best_model_bytasks()
    print(f"Making {num_random_splits} multiclassifier predictions...\n")
    for run in range(1, num_random_splits + 1):
        predictions_bytask_test = []
        for task in range(1, NUM_TASKS + 1):
            data_train = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/data_train_{run}_{task}.pkl")
            data_test = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/data_test_{run}_{task}.pkl")
            labels_train = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/labels_train_{run}_{task}.pkl")
            labels_test = joblib.load(f"runs/{run}/by_tasks/task_{task}/splits/labels_test_{run}_{task}.pkl")

            best_model_bytask = best_models_bytask[task - 1]
            best_model_bytask.fit(data_train, labels_train)

            # Predict Data Test
            predictions_model_test = best_model_bytask.predict(data_test)
            predictions_bytask_test.append(predictions_model_test)

        # Sum of Rows Data Test
        df = pd.DataFrame(predictions_bytask_test).transpose()
        columns = [f"task_{i}" for i in range(1, NUM_TASKS + 1)]
        indexes = [f"{i}" for i in range(1, len(labels_test) + 1)]
        df.columns = columns
        df.index = indexes

        sum_byrows = df.sum(axis=1).to_list()
        predictions_multiclassifier_byrun = []
        for sum_byrow in sum_byrows:
            if sum_byrow > (NUM_TASKS//2):
                predictions_multiclassifier_byrun.append(1)
            else:
                predictions_multiclassifier_byrun.append(0)

        metrics_dict = calculate_metrics(labels_test, predictions_multiclassifier_byrun)

        metrics = [f"mixed_classifier_task_specific_{run}", "mixed", metrics_dict["accuracy"], metrics_dict["sensitivity"],
                   metrics_dict["specificity"], metrics_dict["precision"], metrics_dict["recall"],
                   metrics_dict["f1"], metrics_dict["kappa"]]
        all_metrics.append(metrics)

    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_metrics = df_all_metrics.reset_index(drop=True)
    df_all_metrics.columns = ["name_model", "classifier", "accuracy", "sensitivity", "specificity",
                              "precision", "recall", "f1", "kappa"]

    df_all_metrics.to_csv("performance_metrics/mixed_classifier_task_specific.csv", index=False)
