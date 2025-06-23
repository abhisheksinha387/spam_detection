#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Training model on Render..."
python src/components/data_ingestion.py

echo "Verifying artifacts..."
if [ ! -f "artifacts/model.pkl" ]; then
    echo "ERROR: model.pkl not found!"
    exit 1
fi

if [ ! -f "artifacts/preprocessor.pkl" ]; then
    echo "ERROR: preprocessor.pkl not found!"
    exit 1
fi

echo "Build completed successfully!"