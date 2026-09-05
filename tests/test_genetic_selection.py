import inspect

import numpy as np
import pandas as pd

from src.genetic_selection import _fitness, _init_population, _mutate, _repair, run_ga


def test_no_zero_feature_chromosomes_in_init_population():
    rng = np.random.default_rng(0)
    population = _init_population(n_features=12, size=200, rng=rng)
    assert (population.sum(axis=1) > 0).all()


def test_repair_flips_a_bit_when_all_zero():
    rng = np.random.default_rng(0)
    chromosome = np.zeros(12, dtype=bool)
    _repair(chromosome, rng)
    assert chromosome.sum() == 1


def test_fitness_and_run_ga_have_no_test_set_parameter():
    # Structural leakage guard: the test set cannot influence GA fitness if
    # there is nowhere to pass it in.
    for fn in (_fitness, run_ga):
        params = set(inspect.signature(fn).parameters)
        assert "X_test" not in params and "y_test" not in params


def _toy_data(n=60, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((X["f0"] + rng.normal(scale=0.1, size=n) > 0).astype(int))
    return X, y


def test_run_ga_returns_nonempty_selected_features():
    X, y = _toy_data()
    result = run_ga(X, y, seed=1)
    assert 1 <= len(result["selected_features"]) <= X.shape[1]
    assert len(result["generation_log"]) > 0


def test_mutation_probability_matches_configured_rate():
    rng = np.random.default_rng(0)
    n_features = 1000
    chromosome = np.zeros(n_features, dtype=bool)
    mutated = _mutate(chromosome.copy(), mutation_probability=0.3, rng=rng)
    flip_rate = mutated.sum() / n_features
    assert 0.2 < flip_rate < 0.4
