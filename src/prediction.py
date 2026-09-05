"""Random Forest training/evaluation, shared by the complete-data baseline
(E0) and every imputed-data prediction experiment (E1-E3). Centralizing this
keeps the RF config and metric set identical across experiments, which is
required for a fair comparison (master prompt section 23)."""

from __future__ import annotations

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import RANDOM_SEED

RF_PARAMS = {"n_estimators": 200, "random_state": RANDOM_SEED}


def train_evaluate_rf(X_train, y_train, X_test, y_test, seed: int = RANDOM_SEED) -> dict:
    """Fit an RF on the training split (5-fold stratified CV for a
    training-side F1 estimate) and score once on the held-out test split.
    Test data must never influence model selection (section 23). `seed`
    defaults to the project seed but is overridable for repeated-seed
    experiments (master prompt section 29)."""
    model = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    runtime_seconds = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "cv_f1_mean": float(cv_f1_scores.mean()),
        "cv_f1_std": float(cv_f1_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "runtime_seconds": runtime_seconds,
    }
