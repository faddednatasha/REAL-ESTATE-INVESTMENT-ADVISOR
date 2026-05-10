import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

from data_preprocessing import (
    load_data, handle_missing_and_duplicates, engineer_features,
    encode_categoricals, scale_features, get_feature_columns
)


def prepare_data(filepath: str):
    """Full pipeline: load → clean → engineer → encode → split."""
    df = load_data(filepath)
    df = handle_missing_and_duplicates(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    features = get_feature_columns()
    X = df[features]

    # Targets
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5yr"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42
    )
    return X_train, X_test, yc_train, yc_test, yr_train, yr_test, encoders, df


def train_classifier(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier and log with MLflow."""
    mlflow.set_experiment("Real_Estate_Classification")
    with mlflow.start_run(run_name="XGBoost_Classifier"):
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "xgb_classifier")

        print(f"[Classifier] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    return model, {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}


def train_regressor(X_train, X_test, y_train, y_test):
    """Train Random Forest regressor and log with MLflow."""
    mlflow.set_experiment("Real_Estate_Regression")
    with mlflow.start_run(run_name="RF_Regressor"):
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "rf_regressor")

        print(f"[Regressor] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def save_artifacts(clf_model, reg_model, encoders, output_dir="models"):
    """Persist trained models and encoders to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/classifier.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    with open(f"{output_dir}/regressor.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open(f"{output_dir}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print(f"✅ Models and encoders saved to '{output_dir}/'")


def load_artifacts(model_dir="models"):
    """Load saved models and encoders from disk."""
    with open(f"{model_dir}/classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(f"{model_dir}/regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(f"{model_dir}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return clf, reg, encoders


if __name__ == "__main__":
    DATA_PATH = "india_housing_prices.csv"
    print("📦 Preparing data...")
    X_train, X_test, yc_train, yc_test, yr_train, yr_test, encoders, df = prepare_data(DATA_PATH)

    print("🧠 Training classifier...")
    clf_model, clf_metrics = train_classifier(X_train, X_test, yc_train, yc_test)

    print("📈 Training regressor...")
    reg_model, reg_metrics = train_regressor(X_train, X_test, yr_train, yr_test)

    save_artifacts(clf_model, reg_model, encoders)
    print("\n✅ Training complete!")
