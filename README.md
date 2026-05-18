# House Price Prediction

This project trains a Linear Regression model to predict house prices from housing features.

## Project Structure

```text
data/
  Housing.csv
models/
  house_price_model.pkl
src/
  train_model.py
  test_saved_model.py
requirements.txt
README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Model Training

The model was trained using Linear Regression.

### Steps

1. Load housing dataset
2. Clean missing values
3. Encode categorical columns
4. Split dataset into training and testing data
5. Train Linear Regression model
6. Evaluate using MAE, RMSE, and R2
7. Track experiment using MLflow
8. Save trained model using joblib

### Run Training

```bash
python src/train_model.py
```

### Metrics

- MAE: 970043.4039201637
- RMSE: 1324506.960091438
- R2 Score: 0.6529242642153188

## MLflow

After training, start the MLflow UI:

```bash
python -m mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

The experiment name is:

```text
House Price Prediction - Linear Regression
```

## Test Saved Model

```bash
python src/test_saved_model.py
```
## Backend API Documentation (Flask) — Developed by Member 2

The backend is a Flask-powered microservice built to load our trained Linear Regression model binary and expose a highly responsive predictive web endpoint.

### Local Initialization
To start the API development server locally:
```bash
python src/app.py