"""E6 pipeline leakage guard: GA fitness must only ever see the imputed
training split, never the test split, even though run_e6_cell masks and
imputes both (master prompt section 17)."""

from unittest.mock import patch

import experiments.run_e6 as run_e6
from src.data_loader import load_raw_data
from src.preprocessing import train_test_split_stratified


def test_run_e6_cell_ga_never_sees_test_rows():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    seen = {}

    def fake_run_ga(X, y, seed=None, population_size=None, generations=None):
        seen["index"] = set(X.index)
        return {"selected_features": list(X.columns[:3]), "best_fitness": 0.0, "generation_log": []}

    with patch.object(run_e6, "run_ga", fake_run_ga):
        run_e6.run_e6_cell("MCAR", 0.10, "mean", X_train, X_test, y_train, y_test)

    assert seen["index"] == set(X_train.index)
    assert seen["index"].isdisjoint(set(X_test.index))
