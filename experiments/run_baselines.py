"""E0: Random Forest baseline on complete (unimputed) data.

Establishes the reference performance before any missingness/imputation is
introduced. CV is used only on the training split; the held-out test set is
scored once, never used for tuning (master prompt section 23).
"""

import json

from src.config import METRICS_DIR
from src.data_loader import load_raw_data
from src.prediction import RF_PARAMS, train_evaluate_rf
from src.preprocessing import train_test_split_stratified


def run_baseline():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    metrics = train_evaluate_rf(X_train, y_train, X_test, y_test)

    result = {
        "experiment_id": "E0_complete_RF_baseline",
        "dataset": "heart_failure_clinical_records",
        "seed": RF_PARAMS["random_state"],
        "model_params": {k: str(v) for k, v in RF_PARAMS.items()},
        "success": True,
        **metrics,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "e0_complete_rf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    r = run_baseline()
    print(json.dumps({k: v for k, v in r.items() if k != "model_params"}, indent=2))
