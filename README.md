# Video Tutorial : https://youtu.be/LguLwZd8V3w
# Working demo   : https://huggingface.co/spaces/abhisheksinha7742/spam_detector
# (Note : demo is in hugging face and uses gradio and not flask)
Below is the updated README with a **License** section added and the formatting corrected for consistency and clarity. I've also fixed minor formatting issues (e.g., inconsistent headings, code block syntax, and instructions) to ensure a polished and professional appearance. The structure remains the same, and the content is preserved unless clarification or formatting improvements were needed.

---

# 📧 Spam Detection Project

A machine learning-based web application built with **Flask** to classify text messages as **spam** or **ham** (non-spam). This project uses a **Support Vector Machine (SVM)** model trained on **SentenceTransformer** embeddings and SpaCy-based text features.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Running the Application Locally](#running-the-application-locally)
- [Dataset](#dataset)
- [License](#license)

---

## 🔍 Project Overview

This project implements a spam detection system that:

- Ingests a dataset of text messages labeled as spam or ham.
- Processes data with SentenceTransformer embeddings and SpaCy text features.
- Trains a Linear SVM classifier with GridSearchCV for hyperparameter tuning.
- Provides a Flask-based UI for text input and prediction.
- Features a modular design with separate ingestion, transformation, training, and prediction pipelines.

---

## ✨ Features

- **Data Ingestion**: Loads and splits the dataset into training and test sets.
- **Text Embeddings**: Uses `all-MiniLM-L6-v2` from SentenceTransformer for text embeddings.
- **Feature Engineering**: Extracts character, word, and sentence counts using SpaCy.
- **Model Training**: Trains a Linear SVM model with GridSearchCV for optimal performance.
- **Web Interface**: Provides a clean, minimal Flask-based UI for predictions.
- **Logging**: Includes custom logging for debugging and monitoring.

---

## 📁 Project Structure

```
spam_detection/
├── notebooks/
│   └── spam.csv                    # Dataset file
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py  # Embeddings and feature extraction
│   │   ├── model_trainer.py        # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # Prediction logic for web app
│   │   ├── train_pipeline.py       # Full training pipeline
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging utility
│   ├── utils.py                    # Utility functions
├── templates/
│   └── index.html                  # HTML template for Flask UI
├── logs/                           # Runtime logs and error logs
├── artifacts/                      # Generated files (e.g., embeddings, models)
├── app.py                          # Flask web app entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Setup file
├── .gitignore                      # Ignored files
├── README.md                       # This file
```

---

## ⚙️ Prerequisites

- **Python**: 3.8 or higher
- **Git**: For cloning the repository
- **GitHub Account**: For accessing the repository
- **Dataset**: `spam.csv` (see [Dataset](#dataset) section)

---

## 🛠️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Alternatively, if using Conda:

```bash
conda create -p venv python=3.8 -y
conda activate venv/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Prepare the Dataset

Place `spam.csv` in the `notebooks/` folder with the following structure:

- `v1`: Labels (ham/spam)
- `v2`: Text messages

Alternatively, modify `data_ingestion.py` to download the dataset from a URL (e.g., Kaggle).

### 5. Run Data Ingestion

```bash
python -m src.components.data_ingestion
```

This generates files in the `artifacts/` directory:

- `embeddings.pkl`, `label_encoder.pkl`
- `train.csv`, `test.csv`, `data.csv`
- `train_embeddings.csv`, `test_embeddings.csv`
- `train_target.csv`, `test_target.csv`

---

## 🚀 Running the Application Locally

### 1. Start the Flask App

```bash
python app.py
```

### 2. Access the App

Open [http://localhost:5000](http://localhost:5000) in your browser to:

- Enter a text message.
- Click **Predict** to classify the message as spam or ham.
- View the prediction result.

### 3. Check Logs

Runtime logs and errors are saved in the `logs/` directory for debugging.

---

## 📦 Dataset

- **Columns**:
  - `v1`: Label (ham/spam)
  - `v2`: Text message
- **Location**: `notebooks/spam.csv` or downloadable from a hosted URL
- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

