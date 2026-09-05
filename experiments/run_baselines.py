"""E0: Random Forest baseline on complete (unimputed) data.

Establishes the reference performance before any missingness/imputation is
introduced. CV is used only on the training split; the held-out test set is
scored once, never used for tuning (master prompt section 23).
"""

import json
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import METRICS_DIR, RANDOM_SEED
from src.data_loader import load_raw_data
from src.preprocessing import train_test_split_stratified


def run_baseline():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    runtime_seconds = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = {
        "experiment_id": "E0_complete_RF_baseline",
        "dataset": "heart_failure_clinical_records",
        "seed": RANDOM_SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "model_params": model.get_params(),
        "cv_f1_mean": float(cv_f1_scores.mean()),
        "cv_f1_std": float(cv_f1_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "runtime_seconds": runtime_seconds,
        "success": True,
    }

    # model_params contains non-JSON-serializable values (e.g. None is fine,
    # but keep it simple and just stringify anything odd).
    result["model_params"] = {k: str(v) for k, v in result["model_params"].items()}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "e0_complete_rf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    r = run_baseline()
    print(json.dumps({k: v for k, v in r.items() if k != "model_params"}, indent=2))
