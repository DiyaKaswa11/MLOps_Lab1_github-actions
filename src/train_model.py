import argparse
import os
import pickle

import mlflow
from joblib import dump

from model_utils import N_ESTIMATORS, build_models, compute_metrics, load_and_split_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # Load real data and split into train/test sets
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Save the held-out test set for the evaluation script
    os.makedirs("data", exist_ok=True)
    with open("data/test_data.pickle", "wb") as f:
        pickle.dump((X_test, y_test), f)

    base_model, calibrated_model = build_models()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("breast_cancer_random_forest")

    with mlflow.start_run(run_name=f"run_{timestamp}"):
        mlflow.log_params({
            "dataset": "Breast Cancer Wisconsin",
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
            "model": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "calibration": "sigmoid (Platt scaling), 5-fold CV",
        })

        base_model.fit(X_train, y_train)
        calibrated_model.fit(X_train, y_train)

        base_metrics = compute_metrics(base_model, X_test, y_test)
        calibrated_metrics = compute_metrics(calibrated_model, X_test, y_test)
        mlflow.log_metrics({f"base_{k}": v for k, v in base_metrics.items()})
        mlflow.log_metrics({f"calibrated_{k}": v for k, v in calibrated_metrics.items()})

    # Save both models with a timestamp-based version
    os.makedirs("models", exist_ok=True)
    dump(base_model, f"models/model_{timestamp}_rf.joblib")
    dump(calibrated_model, f"models/model_{timestamp}_rf_calibrated.joblib")
    print(f"Saved models for version {timestamp}")
