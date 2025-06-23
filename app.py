from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import get_artifact_path
import sys
import os
import spacy
from sentence_transformers import SentenceTransformer
import joblib

app = Flask(__name__)

# Initialize pipeline and models at startup
try:
    logging.info("Pre-loading models during startup...")
    
    # Check and load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        logging.info("Successfully loaded spaCy model")
    except OSError:
        logging.warning("spaCy model not found, downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    
    # Initialize prediction pipeline
    predict_pipeline = PredictPipeline()
    
    # Warm up the pipeline with a dummy prediction
    dummy_data = CustomData(text="test message")
    dummy_df = dummy_data.get_data_as_dataframe()
    try:
        _ = predict_pipeline.predict(dummy_df)
        logging.info("Successfully warmed up prediction pipeline")
    except Exception as e:
        logging.error(f"Pipeline warm-up failed: {str(e)}")
    
except Exception as e:
    logging.error(f"Failed to pre-load models: {str(e)}")
    # Don't crash the app, but predictions will fail until fixed

@app.route('/')
def home():
    logging.info("Accessed home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request")
    try:
        # Get input text from form
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('index.html', 
                prediction_text='Error: Please enter some text')
        
        # Create CustomData object and get DataFrame
        data = CustomData(text=text)
        input_df = data.get_data_as_dataframe()
        
        # Make prediction
        prediction = predict_pipeline.predict(input_df)
        
        # Map prediction to label
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        logging.info(f"Prediction completed: {result}")
        
        return render_template('index.html', 
            prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        logging.exception("Error during prediction")  # Log full traceback
        error_message = f"Error: {str(e)}" if len(str(e)) < 100 else "Error: Prediction failed"
        return render_template('index.html', 
            prediction_text=error_message)

@app.route('/health')
def health_check():
    try:
        # Check if models are loaded
        if 'predict_pipeline' not in globals():
            raise Exception("Models not loaded")
            
        # Check if artifacts exist
        required_files = ['model.pkl', 'preprocessor.pkl']
        for file in required_files:
            if not os.path.exists(get_artifact_path(file)):
                raise Exception(f"Missing file: {file}")
                
        return "OK", 200
    except Exception as e:
        return f"Health check failed: {str(e)}", 500

if __name__ == '__main__':
    try:
        # Ensure artifacts directory exists
        os.makedirs(os.path.dirname(get_artifact_path('model.pkl')), exist_ok=True)
        
        logging.info("Starting Flask application")
        port = int(os.environ.get('PORT', 5000))
        
        # Use gunicorn-style workers if available
        app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")
        raise CustomException(e, sys)