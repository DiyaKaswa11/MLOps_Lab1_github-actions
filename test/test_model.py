import numpy as np

from src.model_utils import build_models, compute_metrics, load_and_split_data


def test_data_split_shapes():
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert X_train.shape == (455, 30)
    assert X_test.shape == (114, 30)
    assert set(np.unique(y_train)) == {0, 1}


def test_split_is_stratified():
    _, _, y_train, y_test = load_and_split_data()
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_models_output_valid_predictions():
    X_train, X_test, y_train, _ = load_and_split_data()
    for model in build_models():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        assert probs.min() >= 0 and probs.max() <= 1
        assert set(np.unique(model.predict(X_test))) <= {0, 1}


def test_compute_metrics():
    X_train, X_test, y_train, y_test = load_and_split_data()
    base_model, _ = build_models()
    base_model.fit(X_train, y_train)
    metrics = compute_metrics(base_model, X_test, y_test)
    assert set(metrics) == {"accuracy", "f1_score", "roc_auc", "brier_score"}
    for value in metrics.values():
        assert 0 <= value <= 1
    assert metrics["accuracy"] > 0.9
