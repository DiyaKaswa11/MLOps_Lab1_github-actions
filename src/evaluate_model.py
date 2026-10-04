import argparse
import json
import os
import pickle

from joblib import load

from model_utils import compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp

    try:
        base_model = load(f"models/model_{timestamp}_rf.joblib")
        calibrated_model = load(f"models/model_{timestamp}_rf_calibrated.joblib")
    except FileNotFoundError as e:
        raise ValueError(f"Could not find trained models for version {timestamp}") from e

    with open("data/test_data.pickle", "rb") as f:
        X_test, y_test = pickle.load(f)

    metrics = {
        "version": timestamp,
        "dataset": "Breast Cancer Wisconsin",
        "test_samples": int(len(y_test)),
        "random_forest": compute_metrics(base_model, X_test, y_test),
        "calibrated_random_forest": compute_metrics(calibrated_model, X_test, y_test),
    }

    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{timestamp}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
