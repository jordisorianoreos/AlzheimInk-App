import pandas as pd
import numpy as np
import joblib


def app_predictor_model(user_data_list):

    # Adapts the list to a single row array
    user_data = np.array(user_data_list).reshape(1, -1)

    num_of_voting_splits = 19  # should be odd

    # The name of the models of which it is composed is written.
    top3_models = ["GaussianNB_5", "RandomForestClassifier_13", "RandomForestClassifier_15"]

    # Splits numbers are obtained for the three models
    df_top3_mixed = pd.read_csv("performance_metrics/top3_mixed_classifiers_voting_models.csv")
    df_top3_mixed.sort_values("f1", ascending=False, inplace=True)
    voting_splits = []
    for split in range(num_of_voting_splits):
        split_num = df_top3_mixed["run"].iloc[split]
        voting_splits.append(split_num)

    models_prediction = []
    list_all_models_predictions = []

    #  Predictions are made for each of the splits of the three models and a list with 0 and 1
    #  is obtained with the results of the 19 predictions
    for i, model_name in enumerate(top3_models):
        model = joblib.load(f'app_predictor_model/top3_models/{model_name}.pkl')
        predictions_by_model = []
        for split in voting_splits:
            data_train = joblib.load(f'app_predictor_model/split_{split}/data_train_{split}.pkl')
            data_test = joblib.load(f'app_predictor_model/split_{split}/data_test_{split}.pkl')
            labels_train = joblib.load(f'app_predictor_model/split_{split}/labels_train_{split}.pkl')

            if data_test.shape[1] == user_data.shape[1]:
                model.fit(data_train, labels_train)
                prediction_model = model.predict(user_data)
            else:
                print("No se han completado todas las tareas.")
            predictions_by_model.append(prediction_model[0])

        models_prediction.append(predictions_by_model)

        if sum(predictions_by_model) > (len(predictions_by_model)//2):
            list_all_models_predictions.append(1)
        else:
            list_all_models_predictions.append(0)

    # A dataframe is made with the three rows of the predictions
    df_results = pd.DataFrame(models_prediction)
    columns = [f"split_{i}" for i in range(1, num_of_voting_splits + 1)]
    indexs = [f"{model}" for model in top3_models]
    df_results.columns = columns
    df_results.index = indexs
    df_results["prediction"] = list_all_models_predictions  # adding new column
    print(df_results.to_latex())

    # The final diagnosis is established according to whether more than half
    # are of one class or the other
    if sum(list_all_models_predictions) > (len(list_all_models_predictions)//2):
        user_prediction = "Alzheimer"
    else:
        user_prediction = "Sano"

    # The percentage of predictors which have given a diagnosis of Alzheimer's disease is obtained (1).
    perc_predictors = f"{round(((df_results.sum().sum() / df_results.size) * 100), 2)}%"

    return user_prediction, perc_predictors
