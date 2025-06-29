import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    input_data_path: str = os.path.join('notebooks', 'spam.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")
        try:
            logging.info(f"Reading dataset from {self.ingestion_config.input_data_path}")
            df = pd.read_csv(self.ingestion_config.input_data_path, encoding='latin-1')
            logging.info('Read the dataset as dataframe')

            required_columns = ['v1', 'v2']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset missing required columns: {required_columns}")

            if not df['v1'].isin(['ham', 'spam']).all():
                raise ValueError("Target column (v1) contains unexpected values")

            logging.info("Renaming columns and dropping unnecessary ones")
            df = df.rename(columns={'v1': 'target', 'v2': 'text'})
            df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

            logging.info("Removing duplicate rows")
            df = df.drop_duplicates(keep='first')
            logging.info(f"Removed {len(df) - len(df.drop_duplicates())} duplicates")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data")

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Saved train and test data")
            logging.info("Ingestion of the data is completed.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                train_set,
                test_set
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data_path, test_data_path, train_set, test_set = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        logging.info("Transforming train data")
        train_embeddings, train_target, train_embeddings_path, train_target_path = data_transformation.initiate_data_transformation(
            train_data_path, is_train=True
        )
        preprocessor_path = os.path.join('artifacts', 'embeddings.pkl')
        label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')
        if not os.path.exists(preprocessor_path) or not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Missing embeddings.pkl or label_encoder.pkl")
        logging.info("Transforming test data")
        test_embeddings, test_target, test_embeddings_path, test_target_path = data_transformation.initiate_data_transformation(
            test_data_path, is_train=False
        )
        logging.info("Initiating model training")
        model_trainer = ModelTrainer()
        results = model_trainer.initiate_model_training(
            train_embeddings_path=train_embeddings_path,
            train_target_path=train_target_path,
            test_embeddings_path=test_embeddings_path,
            test_target_path=test_target_path
        )
        logging.info(f"Training completed with results: {results}")
    except Exception as e:
        logging.error(f"Error in main block: {str(e)}")
        raise CustomException(e, sys)