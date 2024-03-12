## MLOps Project

This project consists of developing a Machine Learning model and deploying it into production.

### Initial steps

1. Create a virtual environment
2. Install zenml `pip install zenml["server"]`
3. Initialize zenml `zenml init`
4. Execute zenml `zenml up`
5. Create folders:
    - src
    - pipeline
    - saved model
    - steps
6. Create `run_pipeline.py` file
7. Install required packages fro mthe `requirements.txt`
8. We define the steps (`\steps` folder)
    - ingest data
    - clean data
    - train model
    - evaluation
9. Creating a pipeline (`\pipeline` folder)
    - training pipeline



