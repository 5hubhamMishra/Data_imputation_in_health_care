import numpy as np
import pandas as pd

from src.imputation import apply_imputer, fit_imputer, impute_mean, score_imputation


def test_fit_imputer_uses_train_statistics_only():
    train = pd.DataFrame({"age": [10.0, 20.0, np.nan, 40.0]})
    test = pd.DataFrame({"age": [np.nan, 100.0]})

    imputer = fit_imputer("mean", train, columns=["age"])
    train_mean = train["age"].mean()  # 23.333...

    filled_test = apply_imputer(imputer, test, columns=["age"])
    assert filled_test.loc[0, "age"] == train_mean  # not test's own (non-NaN) mean of 100.0
    assert filled_test.loc[1, "age"] == 100.0  # untouched observed value


def test_impute_mean_matches_manual_fillna():
    df = pd.DataFrame({"age": [10.0, np.nan, 30.0]})
    imputed = impute_mean(df, columns=["age"])
    assert imputed.loc[1, "age"] == 20.0  # mean of 10 and 30


def test_score_imputation_only_scores_hidden_cells():
    imputed = pd.DataFrame({"age": [1.0, 2.0, 3.0]})
    ground_truth = pd.DataFrame({"age": [np.nan, 2.5, np.nan]})  # only index 1 was hidden
    scores = score_imputation(imputed, ground_truth)
    assert scores["age"]["mae"] == 0.5
