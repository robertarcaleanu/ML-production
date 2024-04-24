import joblib

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
   

def inference_pipeline(input_data, model_path: str):
    """Pipeline used to predict a value

    Args:
        input_data (_type_): input data

    Returns:
        _type_: predicted result
    """
    model = joblib.load(model_path)
    data = ... # process data
    result = model.predict(data)
    
    return f"Your data has been processed: {input_data.feature1} and {input_data.feature2}"

def continuous_deployment_pipeline(data_path: str, min_accuracy: float, model_path: str) -> str:
    """In this function we train the model and save it if the accuracy is higher than the threshold

    Args:
        min_accuracy (float): minimum accuracy required to save the model

    Returns:
        _type_: _description_
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rsme = evaluate_model(model, X_test, y_test)
    # min_accuracy = 0.85
    accuracy = 0.5
    if accuracy > min_accuracy:
        joblib.dump(model, 'LinearRegression.joblib')
        message = "Model has been saved. Accuracy: {}".format(accuracy)
    else:
        message = "Model has not been saved. Accuracy: {}".format(accuracy)

    return message
