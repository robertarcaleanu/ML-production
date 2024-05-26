import logging
import uvicorn
import conf

from pydantic import BaseModel
from fastapi import FastAPI

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

def create_app():
    """Creating the FastAPI application

    Args:
        deployment_type (Literal[&quot;train&quot;, &quot;predict&quot;]): _description_
        min_accuracy (float): _description_
    """
    
    class InputData(BaseModel):
        # Define your input data schema
        payment_sequential: float
        payment_installments: float
        payment_value: float
        price: float
        freight_value: float
        product_name_lenght: float
        product_description_lenght: float
        product_photos_qty: float
        product_weight_g: float
        product_length_cm: float
        product_height_cm: float
        product_width_cm: float
        # Add more features as needed


    app = FastAPI()
    @app.post("/predict")
    def predict(input_data: InputData):
        try:
            result = inference_pipeline(
                input_data.dict(), 
                model_path=conf.MODEL_PATH)
            return {"result": result}
        except Exception as e:
            logging.error("Error in predicting the result: {}".format(e))
            raise e
        

    @app.post("/train")
    def train():
        try:
            result = continuous_deployment_pipeline(
                data_path=conf.DATA_PATH,
                min_accuracy=conf.MIN_ACCURACY,
                model_path=conf.MODEL_PATH)
            
            return {"result": result}
        except Exception as e:
            logging.error("Error in training the model: {}".format(e))
            raise e
        
    return app


def main():
    """Run the FastAPI application."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
