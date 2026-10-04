# MLOps Lab 1 – Model Training, Versioning and Calibration with GitHub Actions

Based on Lab 2 of the Github_Labs folder in the MLOps course (IE7305). Every push to `main` runs unit tests, then trains a Random Forest classifier and a calibrated version of it, evaluates both on a held-out test set, and commits the timestamp-versioned models and metrics back to the repository.

## Project Structure

    .
    ├── .github/workflows/
    │   └── model_retraining_on_push.yml   # CI/CD pipeline
    ├── src/
    │   ├── model_utils.py                 # Data loading, model building, metrics
    │   ├── train_model.py                 # Trains both models, logs runs to MLflow
    │   └── evaluate_model.py              # Evaluates both models on the test set
    ├── test/
    │   └── test_model.py                  # Unit tests
    ├── models/                            # Versioned models (added by the pipeline)
    ├── metrics/                           # Versioned metrics (added by the pipeline)
    └── requirements.txt

## Changes from the original lab

### Data and model
- Replaced the synthetic `make_classification` data with the real **Breast Cancer Wisconsin** dataset (569 samples, 30 features, built into scikit-learn). The original also picked a random sample size between 0 and 2000, which could crash on 0.
- Added a stratified **80/20 train/test split**. The original evaluated the model on newly generated data from the same generator instead of a held-out test set.
- Moved shared code into `src/model_utils.py` so training, evaluation and tests all use the same functions.

### Calibration
- The original README describes model calibration, but there was no calibration code. I added a calibrated Random Forest using `CalibratedClassifierCV` with sigmoid (Platt scaling) and 5-fold cross-validation, and the pipeline now trains and compares both models.

### Evaluation
- Expanded metrics from F1 only to **accuracy, F1, ROC-AUC and Brier score** (the Brier score measures probability quality, which is what calibration is meant to improve).
- Metrics for both models are saved as versioned JSON files and shown in the GitHub Actions run summary.

### MLflow
- The original used the `./mlruns` file store, which fails on current MLflow versions. Switched to the recommended SQLite backend (`sqlite:///mlflow.db`). Each run logs dataset details, model parameters and metrics for both models.

### CI/CD pipeline
- Replaced the two original workflows, which used paths from the course repo, outdated action versions and an undefined variable, with one working workflow.
- Added a **test job** with 4 pytest tests; training only runs if all tests pass.
- Added `permissions: contents: write` so the workflow can push models and metrics back to the repo.
- Triggers: push to `main`, a weekly schedule (Sundays at midnight UTC), and manual runs. Pushes that only change models, metrics or the README do not trigger retraining.
- Added pip caching to speed up runs.

## Results

Test set of 114 samples:

| Model | Accuracy | F1 | ROC-AUC | Brier score |
|---|---|---|---|---|
| Random Forest | 0.9561 | 0.9655 | 0.9937 | 0.0321 |
| Calibrated Random Forest | 0.9561 | 0.9655 | 0.9927 | 0.0337 |

Calibration did not improve the model here: the Brier score went slightly up (lower is better). The Random Forest's probabilities were already well calibrated on this dataset, so Platt scaling added little. Calibration matters more for models whose raw probabilities are badly skewed.

## How to Run Locally

    python3 -m venv lab_02
    source lab_02/bin/activate
    pip install -r requirements.txt
    pytest -v
    python src/train_model.py --timestamp localtest
    python src/evaluate_model.py --timestamp localtest
