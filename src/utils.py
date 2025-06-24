import pandas as pd
import numpy as np
import spacy
import joblib
import os
from src.exception import CustomException
from src.logger import logging
import sys

def get_artifact_path(file_name: str) -> str:
    try:
        path = os.path.join('artifacts', file_name)
        logging.info(f"Generated artifact path: {path}")
        return path
    except Exception as e:
        logging.error(f"Error generating artifact path for {file_name}: {str(e)}")
        raise CustomException(e, sys)

def compute_text_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    try:
        logging.info("Computing text features")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("en_core_web_sm model not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        df['num_characters'] = df[text_column].apply(len)
        df['num_words'] = df[text_column].apply(lambda x: len([token for token in nlp(x) if token.is_alpha]))
        df['num_sentences'] = df[text_column].apply(lambda x: len(list(nlp(x).sents)))
        logging.info("Text features computed successfully")
        return df
    except Exception as e:
        logging.error(f"Error computing text features: {str(e)}")
        raise CustomException(e, sys)

def generate_embeddings(texts: list, embeddings_path: str) -> pd.DataFrame:
    try:
        logging.info("Generating text embeddings")
        preprocessor = joblib.load(embeddings_path)
        embeddings = preprocessor.encode(texts, batch_size=32, show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings)
        logging.info("Text embeddings generated successfully")
        return df_embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise CustomException(e, sys)

def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Validating and preparing DataFrame")
        if not np.all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("Non-numerical columns found in DataFrame")
        df.columns = df.columns.astype(str)
        logging.info(f"DataFrame validated, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Error validating DataFrame: {str(e)}")
        raise CustomException(e, sys)