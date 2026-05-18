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

## Presentation Script

My responsibility was model training and MLflow tracking.

First, I loaded the housing price dataset using pandas. Then I separated the data into input features and target price.

I cleaned the data by handling missing values. For numerical columns, I used the median value. For categorical columns, I used the most frequent value and converted text values using OneHotEncoder.

After that, I split the dataset into 80% training data and 20% testing data. Then I trained a Linear Regression model.

To evaluate the model, I used MAE, RMSE, and R2 score. MAE and RMSE measure prediction error, while R2 shows how well the model explains the house price.

Finally, I used MLflow to track the experiment. I logged the parameters, metrics, and trained model. I also saved the model as house_price_model.pkl so the Flask API can use it for prediction.
