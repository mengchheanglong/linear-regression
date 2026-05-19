# House Price Prediction

A full-stack machine learning application that predicts house prices using a Linear Regression model, served through a Flask REST API, and presented via a Streamlit web interface.

---

## Team Member Responsibilities

| Member | Role | Deliverables |
|--------|------|--------------|
| Member 1 | ML Model | `src/train_model.py`, `data/Housing.csv`, `models/house_price_model.pkl`, MLflow tracking |
| Member 2 | Flask API | `src/app.py` — `/` health check, `POST /predict` endpoint |
| Member 3 | Streamlit UI + Deployment | `streamlit_app.py`, `requirements.txt`, README, Render deployment notes |

---

## Project Structure

```text
data/
  Housing.csv              # Raw housing dataset
models/
  house_price_model.pkl    # Trained model binary
mlruns/                    # MLflow experiment artifacts
src/
  train_model.py           # Model training script (Member 1)
  test_saved_model.py      # Model sanity-check script
  app.py                   # Flask REST API (Member 2)
streamlit_app.py           # Streamlit frontend (Member 3)
requirements.txt
README.md
```

---

## Setup

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Full Run Instructions

Follow these steps **in order** to run the complete application.

### Step 1 — Train the Model (Member 1)

> Skip if `models/house_price_model.pkl` already exists.

```bash
python src/train_model.py
```

This trains the Linear Regression model on `data/Housing.csv`, logs the experiment to MLflow, and saves the model to `models/house_price_model.pkl`.

### Step 2 — Start the Flask API (Member 2)

Open a **new terminal** and run:

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

### Step 3 — Start the Streamlit App (Member 3)

Open another **new terminal** and run:

```bash
streamlit run streamlit_app.py
```

Streamlit opens automatically in your browser at `http://localhost:8501`.

Fill in the house details and click **Predict Price**.

---

## API Reference

### `GET /`

Health check.

**Response:**
```json
{"status": "healthy", "message": "House Price Prediction API is running.", "model_loaded": true}
```

### `POST /predict`

Predict house price from features.

**Request body (JSON):**
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

**Response:**
```json
{"predicted_price": 13142500.00}
```

---

## Model Training Details

- **Algorithm:** Linear Regression (scikit-learn Pipeline with preprocessing)
- **Preprocessing:** Median imputation for numerics, mode imputation + OneHotEncoding for categoricals
- **Train/Test Split:** 80% / 20% (random_state=42)

### Metrics

| Metric | Value |
|--------|-------|
| MAE    | 970,043 |
| RMSE   | 1,324,507 |
| R²     | 0.653 |

### View MLflow Experiments

```bash
python -m mlflow ui
```

Open `http://127.0.0.1:5000` (start the MLflow UI *before* the Flask API to avoid port conflict, or use `--port 5001`).

---

## Test Saved Model

```bash
python src/test_saved_model.py
```

---

## Deployment on Render

### Architecture on Render

Deploy the **Flask API** as a Web Service and the **Streamlit app** as a separate Web Service.

### Flask API — Render Web Service

1. Push the repository to GitHub.
2. In Render, create a new **Web Service** and connect your repository.
3. Set the following:

   | Field | Value |
   |-------|-------|
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `python src/app.py` |
   | **Environment** | Python 3 |

4. Add any required environment variables (none needed by default).
5. After deploy, Render provides a public URL like `https://your-api.onrender.com`.

### Streamlit App — Render Web Service

1. Create a second **Web Service** in Render from the same repository.
2. Set the following:

   | Field | Value |
   |-------|-------|
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
   | **Environment** | Python 3 |

3. Add an environment variable `API_URL` pointing to your deployed Flask API URL if you parameterize it (optional — see note below).

> **Note:** For production, replace the hardcoded `API_URL = "http://localhost:5000/predict"` in `streamlit_app.py` with `os.environ.get("API_URL", "http://localhost:5000/predict")` so the Streamlit app can point to the deployed Flask service.

### Important Notes for Render

- The free tier spins down after inactivity — the first request may be slow (cold start).
- Ensure `models/house_price_model.pkl` is committed to the repository so Render can access it at runtime.
- The Flask app binds to `0.0.0.0` and uses port `5000` by default — Render overrides the port via `$PORT` if needed. Update `app.run(port=int(os.environ.get("PORT", 5000)))` for full Render compatibility.
