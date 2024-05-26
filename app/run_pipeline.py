from pipelines.training_pipeline import train_pipeline

import logging

if __name__ == "__main__":
    try:
        train_pipeline(data_path='~/DS-Projects/MLOps/ML-production/datasets/olist_customers_dataset-complete.csv')
        logging.info("Executed from Linux")
    except Exception as e:
        train_pipeline(data_path='C:/Users/rober/OneDrive/Escriptori/DataSets/MLOps/ML-production/datasets/olist_customers_dataset-complete.csv')
        logging.info("Executed from Windowws")