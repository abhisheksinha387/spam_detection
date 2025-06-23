import pandas as pd
import numpy as np
import joblib
from src.exception import CustomException
from src.logger import logging
import sys
import os
from src.utils import compute_text_features, generate_embeddings, validate_and_prepare_dataframe

class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_dataframe(self):
        try:
            logging.info("Creating DataFrame from input text")
            data = {'text': [self.text]}
            df = pd.DataFrame(data)
            
            # Compute text features using utils
            df = compute_text_features(df)
            
            logging.info("Input data processed successfully")
            return df
        except Exception as e:
            logging.error(f"Error processing input data: {str(e)}")
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, input_df):
        try:
            logging.info("Starting prediction pipeline")
            
            # Generate embeddings using utils
            df_embeddings = generate_embeddings(input_df['text'].tolist(), self.preprocessor_path)
            
            # Combine features with embeddings
            logging.info("Combining features and embeddings for prediction")
            df_combined = pd.concat([input_df[['num_characters', 'num_words', 'num_sentences']], df_embeddings], axis=1)
            
            # Validate and prepare DataFrame using utils
            df_combined = validate_and_prepare_dataframe(df_combined)
            
            # Load the model
            model = joblib.load(self.model_path)
            
            # Make prediction
            logging.info("Making prediction")
            prediction = model.predict(df_combined)
            
            logging.info("Prediction completed successfully")
            return prediction
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)