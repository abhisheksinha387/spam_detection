from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import get_artifact_path
import sys
import os

app = Flask(__name__)

try:
    logging.info("Pre-loading models during startup...")
    predict_pipeline = PredictPipeline()
    logging.info("Successfully initialized prediction pipeline")
except Exception as e:
    logging.error(f"Failed to pre-load models: {str(e)}")

@app.route('/')
def home():
    logging.info("Accessed home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request")
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('index.html', 
                prediction_text='Error: Please enter some text')
        data = CustomData(text=text)
        input_df = data.get_data_as_dataframe()
        result = predict_pipeline.predict(input_df)
        logging.info(f"Prediction completed: {result}")
        return render_template('index.html', 
            prediction_text=f'Prediction: {result}')
    except Exception as e:
        logging.exception("Error during prediction")
        error_message = f"Error: {str(e)}" if len(str(e)) < 100 else "Error: Prediction failed"
        return render_template('index.html', 
            prediction_text=error_message)

@app.route('/health')
def health_check():
    try:
        required_files = ['svm.pkl', 'embeddings.pkl', 'label_encoder.pkl']
        for file in required_files:
            if not os.path.exists(get_artifact_path(file)):
                raise Exception(f"Missing file: {file}")
        return "OK", 200
    except Exception as e:
        return f"Health check failed: {str(e)}", 500

if __name__ == '__main__':
    try:
        os.makedirs(os.path.dirname(get_artifact_path('svm.pkl')), exist_ok=True)
        logging.info("Starting Flask application")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")
        raise CustomException(e, sys)