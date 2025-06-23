import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
from src.utils import compute_text_features, generate_embeddings, validate_and_prepare_dataframe
from sentence_transformers import SentenceTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_embeddings_path: str = os.path.join('artifacts', 'train_embeddings.csv')
    test_embeddings_path: str = os.path.join('artifacts', 'test_embeddings.csv')
    train_target_path: str = os.path.join('artifacts', 'train_target.csv')
    test_target_path: str = os.path.join('artifacts', 'test_target.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, data_path, is_train=True):
        logging.info("Entering data transformation process.")
        try:
            # Read the data
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)
            logging.info("Data reading successful")

            # Validate input data
            if not all(col in df.columns for col in ['target', 'text']):
                raise ValueError("Input data missing required columns: target, text")
            if not df['target'].isin(['ham', 'spam']).all():
                raise ValueError("Target column contains unexpected values")

            # Log input columns
            logging.info(f"Input DataFrame columns: {df.columns.tolist()}")

            # Encode target
            logging.info("Encoding target column")
            df['target'] = df['target'].map({'ham': 0, 'spam': 1})

            # Feature engineering using utils
            logging.info("Computing text features")
            df = compute_text_features(df)

            # Create text embeddings
            logging.info("Generating text embeddings")
            if is_train:
                # For training, create a new SentenceTransformer model
                logging.info("Loading SentenceTransformer model for training")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                df_embeddings = pd.DataFrame(model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True))
                
                # Save the preprocessor
                logging.info("Saving preprocessor model")
                joblib.dump(model, self.data_transformation_config.preprocessor_obj_file_path)
                logging.info(f"Saved preprocessor to {self.data_transformation_config.preprocessor_obj_file_path}")
                if not os.path.exists(self.data_transformation_config.preprocessor_obj_file_path):
                    raise FileNotFoundError(f"Failed to save preprocessor to {self.data_transformation_config.preprocessor_obj_file_path}")
            else:
                # For testing, load the saved preprocessor
                df_embeddings = generate_embeddings(df['text'].tolist(), self.data_transformation_config.preprocessor_obj_file_path)

            # Combine features with embeddings
            logging.info("Combining features and embeddings")
            df_combined = pd.concat([df[['num_characters', 'num_words', 'num_sentences']], df_embeddings], axis=1)

            # Validate and prepare DataFrame using utils
            df_combined = validate_and_prepare_dataframe(df_combined)

            # Save embeddings and target
            logging.info("Saving embeddings and target")
            embeddings_path = self.data_transformation_config.train_embeddings_path if is_train else self.data_transformation_config.test_embeddings_path
            target_path = self.data_transformation_config.train_target_path if is_train else self.data_transformation_config.test_target_path      
            df_combined.to_csv(embeddings_path, index=False)
            df[['target']].to_csv(target_path, index=False)
            logging.info(f"Saved embeddings to {embeddings_path} and target to {target_path}")

            logging.info("Data transformation completed")
            return (
                df_combined,
                df['target'],
                embeddings_path,
                target_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    train_data_path = os.path.join('artifacts', 'train.csv')
    obj.initiate_data_transformation(train_data_path, is_train=True)
    test_data_path = os.path.join('artifacts', 'test.csv')
    obj.initiate_data_transformation(test_data_path, is_train=False)