def inference_pipeline(input_data):
    """Pipeline used to predict a value

    Args:
        input_data (_type_): input data

    Returns:
        _type_: predicted result
    """
    
    return f"Your data has been processed: {input_data.feature1} and {input_data.feature2}"

def continuous_deployment_pipeline(min_accuracy):
    """In this function we train the model and save it if the accuracy is higher than the threshold

    Args:
        min_accuracy (_type_): minimum accuracy required to save the model

    Returns:
        _type_: _description_
    """
    # min_accuracy = 0.85
    accuracy = 0.5
    if accuracy > min_accuracy:
        message = "Model has been saved. Accuracy: {}".format(accuracy)
    else:
        message = "Model has not been saved. Accuracy: {}".format(accuracy)

    return message