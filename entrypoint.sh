#!/bin/bash
set -e

echo "=========================================================="
echo " Starting Bangla Handwritten Word Recognition System"
echo " SICIP Assignment 04 - BRAC University"
echo "=========================================================="

# Ensure artifacts directory exists
mkdir -p /app/artifacts/mlflow

# Start MLflow UI in the background on port 5000
echo "[1/2] Launching MLflow Tracking Server on port 5000..."
mlflow ui --backend-store-uri sqlite:////app/artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!
echo "MLflow started with PID $MLFLOW_PID"

# Start Streamlit application on port 8501
echo "[2/2] Launching Streamlit Web Application on port 8501..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
