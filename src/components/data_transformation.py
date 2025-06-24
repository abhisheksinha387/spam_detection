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
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataTransformationConfig:
    embeddings_file_path: str = os.path.join('artifacts', 'embeddings.pkl')
    label_encoder_file_path: str = os.path.join('artifacts', 'label_encoder.pkl')
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
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)
            logging.info("Data reading successful")

            if not all(col in df.columns for col in ['target', 'text']):
                raise ValueError("Input data missing required columns: target, text")

            logging.info("Computing text features")
            df = compute_text_features(df)

            logging.info("Generating text embeddings")
            if is_train:
                logging.info("Loading SentenceTransformer model for training")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                df_embeddings = pd.DataFrame(model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True))
                logging.info("Saving embeddings model")
                joblib.dump(model, self.data_transformation_config.embeddings_file_path)
                logging.info(f"Saved embeddings model to {self.data_transformation_config.embeddings_file_path}")
                if not os.path.exists(self.data_transformation_config.embeddings_file_path):
                    raise FileNotFoundError(f"Failed to save embeddings model")
                
                logging.info("Encoding target column and saving label encoder")
                label_encoder = LabelEncoder()
                df['target'] = label_encoder.fit_transform(df['target'])
                joblib.dump(label_encoder, self.data_transformation_config.label_encoder_file_path)
                logging.info(f"Saved label encoder to {self.data_transformation_config.label_encoder_file_path}")
                if not os.path.exists(self.data_transformation_config.label_encoder_file_path):
                    raise FileNotFoundError(f"Failed to save label encoder")
            else:
                df_embeddings = generate_embeddings(df['text'].tolist(), self.data_transformation_config.embeddings_file_path)
                label_encoder = joblib.load(self.data_transformation_config.label_encoder_file_path)
                df['target'] = label_encoder.transform(df['target'])

            logging.info("Combining features and embeddings")
            df_combined = pd.concat([df[['num_characters', 'num_words', 'num_sentences']], df_embeddings], axis=1)
            df_combined = validate_and_prepare_dataframe(df_combined)

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