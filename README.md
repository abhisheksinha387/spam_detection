---

# 📧 Spam Detection Project

A machine learning-based web application built with **Flask** to classify text messages as **spam** or **ham** (non-spam). This project uses a **Support Vector Machine (SVM)** model trained on **SentenceTransformer** embeddings and SpaCy-based text features.

---

## 📚 Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Local Setup](#local-setup)
* [Running the Application Locally](#running-the-application-locally)
* [Deploying on Render](#deploying-on-render)
* [Dataset](#dataset)
* [Contributing](#contributing)
* [License](#license)

---

## 🔍 Project Overview

This project implements a spam detection system that:

* Ingests a dataset of text messages labeled as spam or ham.
* Processes data with SentenceTransformer embeddings and SpaCy text features.
* Trains a Linear SVM classifier with GridSearchCV for tuning.
* Offers a Flask-based UI for text input and prediction.
* Modular design: ingestion, transformation, training, and prediction pipelines.

---

## ✨ Features

* **Data Ingestion**: Loads and splits the dataset.
* **Text Embeddings**: Uses `all-MiniLM-L6-v2` from SentenceTransformer.
* **Feature Engineering**: Character, word, and sentence count via SpaCy.
* **Model Training**: Linear SVM with hyperparameter tuning using GridSearchCV.
* **Web Interface**: Clean, minimal Flask UI for predictions.
* **Logging**: Custom logs for better debugging.
* **Deployment Ready**: Easily deploy on Render.

---

## 📁 Project Structure

```
spam_detection/
├── notebooks/
│   └── spam.csv                    # Dataset file
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading & splitting
│   │   ├── data_transformation.py  # Embeddings & feature extraction
│   │   ├── model_trainer.py        # Model training & evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # Prediction logic for web app
│   │   ├── train_pipeline.py       # Full training pipeline
│   ├── exception.py                # Custom exceptions
│   ├── logger.py                   # Logging utility
│   ├── utils.py                    # Utility functions
├── templates/
│   └── index.html                  # HTML template for Flask UI
├── app.py                          # Flask web app entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Setup file
├── .gitignore                      # Ignored files
├── README.md                       # This file
```

---

## ⚙️ Prerequisites

* Python 3.8+
* Git
* A GitHub account
* (Optional) Render account for deployment
* `spam.csv` dataset

---

## 🛠️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
Windows: venv\Scripts\activate
macOs  : source venv/bin/activate
Linux  : source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Prepare Dataset

Place `spam.csv` in the `notebooks/` folder with:

* `v1` (labels: ham/spam)
* `v2` (text messages)

Alternatively, update `data_ingestion.py` to download from a URL.

---

### 5. Run the Data Ingestion

```bash
python -m src.pipeline.data_ingestion.py
```

Generates `artifacts/`:

* `embeddings.pkl`, `label_encoder.pkl`
* `train.csv`, `test.csv`, `data.csv`
* `train_embeddings.csv`, `test_embeddings.csv`
* `train_target.csv`, `test_target.csv`

---

## 🚀 Running the Application Locally

### Start the Flask App

```bash
python app.py
```

### Access the App

Open [http://localhost:5000](http://localhost:5000) in your browser to:

* Enter a message
* Click **Predict**
* View spam or ham result

### Logs

Check `logs/` for runtime logs and errors.

---

## 🌐 Deploying on Render

### 1. Push to GitHub

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

> Ensure `artifacts/` and `logs/` are excluded via `.gitignore`.

### 2. Create Web Service on Render

* Go to Render → **New** → **Web Service**
* Connect GitHub repo
* Configure:

  * **Name**: `spam-detection-app`
  * **Runtime**: Python
  * **Branch**: `main`
  * **Build Command**:

    ```bash
    pip install -r requirements.txt && python -m spacy download en_core_web_sm
    ```
  * **Start Command**:

    ```bash
    python app.py
    ```
  * **Instance Type**: Free or upgrade for persistence

### 3. Handle Artifacts and Dataset

#### ✅ Option 1: Commit Artifacts

* Temporarily remove `artifacts/` from `.gitignore`
* Commit `model.pkl`, `preprocessor.pkl`, etc.

#### ✅ Option 2: Use Render Disk

* Add disk in Render dashboard:

  * Name: `artifacts-disk`
  * Mount Path: `/app/artifacts`
  * Size: 1GB

**Update paths in code:**

```python
# data_ingestion.py
train_data_path = os.path.join('/app/artifacts', 'train.csv','test.csv','data.csv')
...
# data_transformation.py
preprocessor_obj_file_path = os.path.join('/app/artifacts', 'embeddings.pkl','label_encoder.pkl')
...
```

**Update Build Command:**

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm && cp notebooks/spam.csv /app/artifacts/spam.csv
```

**Update Start Command:**

```bash
python -m src.pipeline.train_pipeline && python app.py
```

#### ✅ Option 3: Use Public URL for Dataset

Update `data_ingestion.py`:

```python
import requests
url = "https://your-public-url/spam.csv"
response = requests.get(url)
with open('temp_spam.csv', 'wb') as f:
    f.write(response.content)
df = pd.read_csv('temp_spam.csv', encoding='latin-1')
```

---

## 📦 Dataset

* **Columns**: `v1` (ham/spam), `v2` (text)
* **Location**: `notebooks/spam.csv` or hosted URL
* **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---


