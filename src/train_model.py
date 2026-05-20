import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


DATA_PATH = "data/Housing.csv"
MODEL_PATH = "models/house_price_model.pkl"
TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "House Price Prediction - Linear Regression"
CV_FOLDS = 5


def build_model(numeric_features, categorical_features, numeric_strategy):
    # Clean missing values in numeric columns using median imputation.
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if numeric_strategy == "interactions":
        numeric_steps.append(
            (
                "feature_engineering",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            )
        )
    elif numeric_strategy == "polynomial":
        numeric_steps.append(
            (
                "feature_engineering",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            )
        )

    numeric_transformer = Pipeline(steps=numeric_steps)

    # Encode categorical features after filling missing values with the
    # most frequent category.
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def build_candidates(numeric_features, categorical_features):
    baseline = build_model(numeric_features, categorical_features, "baseline")
    interactions = build_model(numeric_features, categorical_features, "interactions")
    polynomial = build_model(numeric_features, categorical_features, "polynomial")

    return [
        {
            "name": "baseline",
            "numeric_feature_engineering": "none",
            "target_transform": "none",
            "model": baseline,
        },
        {
            "name": "interactions",
            "numeric_feature_engineering": "pairwise_interactions",
            "target_transform": "none",
            "model": interactions,
        },
        {
            "name": "log_interactions",
            "numeric_feature_engineering": "pairwise_interactions",
            "target_transform": "log1p",
            "model": TransformedTargetRegressor(
                regressor=interactions,
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        },
        {
            "name": "polynomial_degree_2",
            "numeric_feature_engineering": "polynomial_degree_2",
            "target_transform": "none",
            "model": polynomial,
        },
    ]


def evaluate_candidate(candidate, X_train, X_test, y_train, y_test):
    model = candidate["model"]

    # Compare multiple experiment variants using 5-fold cross-validation
    # before checking their test-set performance.
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=None,
    )

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_test = r2_score(y_test, y_pred_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    r2_train = r2_score(y_train, y_pred_train)

    return {
        "mae": mae_test,
        "rmse": rmse_test,
        "r2": r2_test,
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "cv_mae": float(-cv_scores["test_mae"].mean()),
        "cv_rmse": float(-cv_scores["test_rmse"].mean()),
        "cv_r2": float(cv_scores["test_r2"].mean()),
    }


def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns are: {df.columns.tolist()}"
        )

    # Separate the input features from the target price column.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Identify numeric and categorical columns so they can be preprocessed
    # correctly inside the model pipeline.
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numeric columns:", numeric_features)
    print("Categorical columns:", categorical_features)

    # 4. Split train/test: 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Training data size:", X_train.shape)
    print("Testing data size:", X_test.shape)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # 5. Train Linear Regression and compare multiple experiment variants.
    candidates = build_candidates(numeric_features, categorical_features)
    best_result = None

    for candidate in candidates:
        with mlflow.start_run(run_name=candidate["name"]):
            metrics = evaluate_candidate(candidate, X_train, X_test, y_train, y_test)

            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_param("candidate_name", candidate["name"])
            mlflow.log_param(
                "numeric_feature_engineering",
                candidate["numeric_feature_engineering"],
            )
            mlflow.log_param("target_transform", candidate["target_transform"])
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("target_column", TARGET_COLUMN)
            mlflow.log_param("cv_folds", CV_FOLDS)

            mlflow.log_metric("MAE", metrics["mae"])
            mlflow.log_metric("RMSE", metrics["rmse"])
            mlflow.log_metric("R2", metrics["r2"])
            mlflow.log_metric("CV_MAE", metrics["cv_mae"])
            mlflow.log_metric("CV_RMSE", metrics["cv_rmse"])
            mlflow.log_metric("CV_R2", metrics["cv_r2"])

            mlflow.sklearn.log_model(candidate["model"], candidate["name"])

            print(
                f"{candidate['name']}: "
                f"Train R2={metrics['r2_train']:.4f}, Test R2={metrics['r2']:.4f} | "
                f"Train MAE={metrics['mae_train']:.2f}, Test MAE={metrics['mae']:.2f} | "
                f"Train RMSE={metrics['rmse_train']:.2f}, Test RMSE={metrics['rmse']:.2f} | "
                f"CV_R2={metrics['cv_r2']:.4f}"
            )

            current_result = {
                "candidate": candidate,
                "metrics": metrics,
            }

            if best_result is None or metrics["cv_rmse"] < best_result["metrics"]["cv_rmse"]:
                best_result = current_result

    best_candidate = best_result["candidate"]
    best_model = best_candidate["model"]
    best_metrics = best_result["metrics"]

    print("\nBest model selected:")
    print("Candidate:", best_candidate["name"])
    print("MAE:", best_metrics["mae"])
    print("RMSE:", best_metrics["rmse"])
    print("R2 Score:", best_metrics["r2"])
    print("CV MAE:", best_metrics["cv_mae"])
    print("CV RMSE:", best_metrics["cv_rmse"])
    print("CV R2 Score:", best_metrics["cv_r2"])

    # 6. Save best model after refitting it on the full dataset.
    best_model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"Best model saved to {MODEL_PATH}")
    print("All experiments logged in MLflow.")


if __name__ == "__main__":
    main()
