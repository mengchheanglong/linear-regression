import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/Housing.csv"
MODEL_PATH = "models/house_price_model.pkl"
TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # 1. Load Dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2. Choose Target Column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns are: {df.columns.tolist()}"
        )

    # 3. Separate Input and Output
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 4. Identify Column Types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numeric columns:", numeric_features)
    print("Categorical columns:", categorical_features)

    # 5. Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 6. Create Linear Regression Model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 7. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Training data size:", X_train.shape)
    print("Testing data size:", X_test.shape)

    # 8. MLflow Experiment Setup
    mlflow.set_experiment("House Price Prediction - Linear Regression")

    # 9. Train, Evaluate, Track
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Model Evaluation Results")
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("target_column", TARGET_COLUMN)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(model, "linear_regression_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved to {MODEL_PATH}")
        print("Experiment logged in MLflow.")


if __name__ == "__main__":
    main()
