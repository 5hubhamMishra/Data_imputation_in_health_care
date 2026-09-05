"""Genetic Algorithm for feature selection (master prompt sections 25-26).

Binary chromosome, one bit per feature (1 = selected). Fitness is mean
Stratified 5-fold CV F1 computed on the training split only, with a small
feature-count penalty to prefer smaller subsets when CV performance is
comparable:

    fitness = mean_cv_f1 - GA_LAMBDA * (n_selected / n_total)

`_fitness` and `run_ga` take only training data as arguments - there is no
test-set parameter anywhere in this module, so the held-out test set cannot
leak into GA fitness by construction (verified by
tests/test_genetic_selection.py). Reuses `prediction.RF_PARAMS` so the model
configuration matches every other experiment in this repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    GA_CROSSOVER_PROBABILITY,
    GA_ELITE_COUNT,
    GA_FITNESS_N_ESTIMATORS,
    GA_GENERATIONS,
    GA_LAMBDA,
    GA_POPULATION_SIZE,
    GA_TOURNAMENT_SIZE,
    RANDOM_SEED,
)
from src.prediction import RF_PARAMS


def _repair(chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Prevent zero-feature chromosomes by flipping one random bit on."""
    if not chromosome.any():
        chromosome[rng.integers(len(chromosome))] = True
    return chromosome


def _init_population(n_features: int, size: int, rng: np.random.Generator) -> np.ndarray:
    population = rng.random((size, n_features)) < 0.5
    for row in population:
        _repair(row, rng)
    return population


def _fitness(chromosome: np.ndarray, X_train: pd.DataFrame, y_train: pd.Series, seed: int, cache: dict) -> float:
    """Mean 5-fold stratified CV F1 on X_train/y_train restricted to the
    selected columns, minus a feature-count penalty. Cached by chromosome
    since elitism/convergence make the same chromosome reappear often."""
    key = tuple(chromosome.tolist())
    if key in cache:
        return cache[key]

    columns = X_train.columns[chromosome]
    model = RandomForestClassifier(**{**RF_PARAMS, "n_estimators": GA_FITNESS_N_ESTIMATORS, "random_state": seed})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1 = cross_val_score(model, X_train[columns], y_train, cv=cv, scoring="f1").mean()

    feature_ratio = chromosome.sum() / len(chromosome)
    fitness = float(cv_f1 - GA_LAMBDA * feature_ratio)
    cache[key] = fitness
    return fitness


def _tournament_select(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(population), size=tournament_size)
    winner = idx[np.argmax(fitnesses[idx])]
    return population[winner].copy()


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Single-point crossover - simplest scheme that still recombines
    building blocks; uniform crossover is unnecessary complexity for a
    12-bit chromosome."""
    if rng.random() < GA_CROSSOVER_PROBABILITY:
        point = rng.integers(1, len(parent_a))
        child_a = np.concatenate([parent_a[:point], parent_b[point:]])
        child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    else:
        child_a, child_b = parent_a.copy(), parent_b.copy()
    return child_a, child_b


def _mutate(chromosome: np.ndarray, mutation_probability: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(len(chromosome)) < mutation_probability
    chromosome[flips] = ~chromosome[flips]
    return chromosome


def run_ga(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_SEED,
    population_size: int = GA_POPULATION_SIZE,
    generations: int = GA_GENERATIONS,
) -> dict:
    """Run one GA feature-selection search using only X_train/y_train.

    `population_size`/`generations` default to the master-prompt-specified
    30/30 (used by E5's reported complete-data results). A caller may pass
    smaller values to cut cost for a large combinatorial sweep (e.g. E6's
    9-cell mechanism x missingness x GA run) - this only changes search
    thoroughness for that caller, not the E5 baseline or its documented
    parameters.

    Returns the best chromosome/selected features found across all
    generations, plus a per-generation log (best/mean fitness, selected
    feature count) for convergence plots and stability analysis.
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    mutation_probability = 1.0 / n_features
    cache: dict = {}

    population = _init_population(n_features, population_size, rng)
    generation_log = []
    best_chromosome, best_fitness = None, -np.inf

    for generation in range(generations):
        fitnesses = np.array([_fitness(c, X_train, y_train, seed, cache) for c in population])

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_chromosome = population[gen_best_idx].copy()

        generation_log.append({
            "generation": generation,
            "best_fitness": float(fitnesses.max()),
            "mean_fitness": float(fitnesses.mean()),
            "selected_feature_count": int(population[gen_best_idx].sum()),
        })

        elite_idx = np.argsort(fitnesses)[-GA_ELITE_COUNT:]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < population_size:
            parent_a = _tournament_select(population, fitnesses, GA_TOURNAMENT_SIZE, rng)
            parent_b = _tournament_select(population, fitnesses, GA_TOURNAMENT_SIZE, rng)
            child_a, child_b = _crossover(parent_a, parent_b, rng)
            for child in (child_a, child_b):
                _mutate(child, mutation_probability, rng)
                _repair(child, rng)
                next_population.append(child)

        population = np.array(next_population[:population_size])

    return {
        "seed": seed,
        "best_chromosome": best_chromosome,
        "best_fitness": best_fitness,
        "selected_features": list(X_train.columns[best_chromosome]),
        "generation_log": generation_log,
    }
