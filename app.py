from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import get_artifact_path
import sys
import os

app = Flask(__name__)

@app.route('/')
def home():
    logging.info("Accessed home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request")
    try:
        # Get input text from form
        text = request.form['text']
        if not text.strip():
            raise CustomException("Empty input text provided", sys)
        
        # Create CustomData object and get DataFrame
        data = CustomData(text=text)
        input_df = data.get_data_as_dataframe()
        
        # Initialize prediction pipeline and predict
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)
        
        # Map prediction to label
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        logging.info(f"Prediction completed: {result}")
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        error_message = CustomException(e, sys).error_message
        return render_template('index.html', prediction_text=f'Error: {error_message}')

if __name__ == '__main__':
    try:
        # Ensure artifacts directory exists
        os.makedirs(os.path.dirname(get_artifact_path('model.pkl')), exist_ok=True)
        logging.info("Starting Flask application")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")
        raise CustomException(e, sys)