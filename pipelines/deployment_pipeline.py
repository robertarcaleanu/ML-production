def inference_pipeline(input_data):
    
    return f"Your data has been processed: {input_data.feature1} and {input_data.feature2}"

def continuous_deployment_pipeline(min_accuracy):
    # min_accuracy = 0.85
    accuracy = 0.5
    if accuracy > min_accuracy:
        message = "Model has been saved. Accuracy: {}".format(accuracy)
    else:
        message = "Model has not been saved. Accuracy: {}".format(accuracy)

    return message