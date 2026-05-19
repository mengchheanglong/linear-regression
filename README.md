# House Price Prediction

A full-stack machine learning application that predicts house prices using a Linear Regression model, served through a Flask REST API, and presented through a Streamlit web interface.

## Team Member Responsibilities

| Member | Role | Deliverables |
|--------|------|--------------|
| Member 1 | ML Model | `src/train_model.py`, `data/Housing.csv`, `models/house_price_model.pkl`, MLflow tracking |
| Member 2 | Flask API | `src/app.py`, `/` health check, `POST /predict` endpoint |
| Member 3 | Streamlit UI + Deployment | `streamlit_app.py`, `requirements.txt`, `render.yaml`, README, Render deployment notes |

## Project Structure

```text
data/
  Housing.csv
models/
  house_price_model.pkl
mlruns/
src/
  app.py
  test_saved_model.py
  train_model.py
streamlit_app.py
render.yaml
requirements.txt
README.md
```

## Setup

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Full Run Instructions

Run the complete application in this order.

### Step 1 - Train the Model

Skip this step if `models/house_price_model.pkl` already exists.

```bash
python src/train_model.py
```

This runs several Linear Regression experiments on `data/Housing.csv`, logs them to MLflow, and saves the best-performing model to `models/house_price_model.pkl`.

### Step 2 - Start the Flask API

Open a new terminal and run:

```bash
python src/app.py
```

The API starts at `http://localhost:5000`.

Verify it is running:

```bash
curl http://localhost:5000/
```

Expected response:

```json
{"status": "healthy", "message": "House Price Prediction API is running.", "model_loaded": true}
```

### Step 3 - Start the Streamlit App

Open another new terminal and run:

```bash
streamlit run streamlit_app.py
```

Streamlit opens automatically in your browser at `http://localhost:8501`.

## API Reference

### `GET /`

Health check.

Response:

```json
{"status": "healthy", "message": "House Price Prediction API is running.", "model_loaded": true}
```

### `POST /predict`

Predict house price from features.

Request body:

```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}
```

Response:

```json
{"predicted_price": 13142500.0}
```

## Model Training Details

- Algorithm: Linear Regression
- Experiment strategy: compare baseline Linear Regression, interaction features, log-transformed target, and degree-2 polynomial numeric features
- Selected final pipeline: Linear Regression with pairwise interaction features on numeric columns
- Preprocessing: median imputation plus pairwise interaction features for numerics, most-frequent imputation plus one-hot encoding for categoricals
- Train/test split: 80/20 with `random_state=42`

### Metrics

| Metric | Value |
|--------|-------|
| MAE    | 960,206 |
| RMSE   | 1,300,236 |
| R2     | 0.666 |

### View MLflow Experiments

```bash
python -m mlflow ui --port 5001
```

Open `http://127.0.0.1:5001`.

## Test Saved Model

```bash
python src/test_saved_model.py
```

## Render Deployment

This repo includes `render.yaml` for two web services:

- `linear-regression-api`
- `linear-regression-streamlit`

### Flask Service

- Build command: `pip install -r requirements.txt`
- Start command: `python src/app.py`

### Streamlit Service

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

### Required Environment Variables

- `API_URL` for the Streamlit service
  Example: `https://your-api-service.onrender.com/predict`

### Notes

- The Flask app reads the port from `$PORT`, which is required by Render.
- The Streamlit app reads `API_URL` from the environment, so it can call the deployed Flask API.
- Ensure `models/house_price_model.pkl` is committed so Render can load it at runtime.
