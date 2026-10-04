"""Shared helpers for training and evaluating the breast cancer classifier."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_ESTIMATORS = 100


def load_and_split_data(test_size=0.2):
    """Loads the Breast Cancer Wisconsin dataset and splits it into train and test sets."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def build_models():
    """Returns a random forest and a calibrated random forest (Platt scaling)."""
    base_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    calibrated_model = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        method="sigmoid",
        cv=5,
    )
    return base_model, calibrated_model


def compute_metrics(model, X, y):
    """Returns accuracy, F1, ROC-AUC and Brier score for a fitted model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1_score": round(float(f1_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y, y_prob)), 4),
    }
