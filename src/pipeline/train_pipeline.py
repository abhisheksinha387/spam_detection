import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            logging.info("Initiating data ingestion")
            train_data_path, test_data_path, _, _ = self.data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            logging.info("Initiating data transformation for training data")
            train_embeddings, train_target, train_embeddings_path, train_target_path = self.data_transformation.initiate_data_transformation(
                train_data_path, is_train=True
            )
            
            logging.info("Initiating data transformation for test data")
            test_embeddings, test_target, test_embeddings_path, test_target_path = self.data_transformation.initiate_data_transformation(
                test_data_path, is_train=False
            )
            
            # Model Training
            logging.info("Initiating model training")
            results = self.model_trainer.initiate_model_training(
                train_embeddings_path=train_embeddings_path,
                train_target_path=train_target_path,
                test_embeddings_path=test_embeddings_path,
                test_target_path=test_target_path
            )
            
            logging.info(f"Training pipeline completed with results: {results}")
            return results
        
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        results = pipeline.run_pipeline()
        print("Training Results:", results)
    except Exception as e:
        logging.error(f"Failed to run training pipeline: {str(e)}")
        raise CustomException(e, sys)