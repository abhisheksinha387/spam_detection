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
            df = compute_text_features(df)
            logging.info("Input data processed successfully")
            return df
        except Exception as e:
            logging.error(f"Error processing input data: {str(e)}")
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'svm.pkl')
        self.embeddings_path = os.path.join('artifacts', 'embeddings.pkl')
        self.label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')

    def predict(self, input_df):
        try:
            logging.info("Starting prediction pipeline")
            df_embeddings = generate_embeddings(input_df['text'].tolist(), self.embeddings_path)
            logging.info("Combining features and embeddings for prediction")
            df_combined = pd.concat([input_df[['num_characters', 'num_words', 'num_sentences']], df_embeddings], axis=1)
            df_combined = validate_and_prepare_dataframe(df_combined)
            model = joblib.load(self.model_path)
            label_encoder = joblib.load(self.label_encoder_path)
            logging.info("Making prediction")
            prediction = model.predict(df_combined)
            result = label_encoder.inverse_transform(prediction)[0]
            logging.info(f"Prediction completed: {result}")
            return result
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)