import logging

from typing import Literal

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)


def main(deployment_type: Literal["train", "predict"], min_accuracy: float):
    """Run the pipeline."""
    try:
        if deployment_type == "train":
            # Initialize a continuous deployment pipeline run
            continuous_deployment_pipeline(
                min_accuracy=min_accuracy,
                workers=3,
                timeout=60,
            )
    except Exception as e:
        logging.error(f"Error in deploying {deployment_type} pipeline: {e}")
        raise e
    
    try:
        if deployment_type == "predict":
            # Initialize an inference pipeline run
            inference_pipeline(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
            )
    except Exception as e:
        logging.error(f"Error in deploying {deployment_type} pipeline: {e}")
        raise e


if __name__ == "__main__":
    main(deployment_type="predict", min_accuracy=0.85)
