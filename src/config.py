"""Central configuration for the heart failure imputation/GA pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "heart_failure_clinical_records_dataset.csv"

TARGET_COLUMN = "DEATH_EVENT"

# Columns that must never be masked by the missingness framework or dropped
# by preprocessing/feature selection.
PROTECTED_COLUMNS = [TARGET_COLUMN]

# Continuous numeric features eligible for controlled missingness + mean/
# median/KNN imputation. Excludes the 5 binary clinical flags (anaemia,
# diabetes, high_blood_pressure, sex, smoking): mean/median imputation on a
# 0/1 flag is not a meaningful reconstruction target, so this cycle scopes
# masking/imputation to genuinely continuous-valued columns.
CONTINUOUS_FEATURES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

RANDOM_SEED = 42
TEST_SIZE = 0.2

PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

# Genetic Algorithm feature-selection defaults (master prompt section 25).
# mutation_probability is not fixed here: it is set to 1/n_features at
# runtime, per the master prompt's guidance.
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 30
GA_CROSSOVER_PROBABILITY = 0.8
GA_TOURNAMENT_SIZE = 3
GA_ELITE_COUNT = 2
# Small feature-count penalty in the GA fitness function:
# fitness = mean_cv_f1 - GA_LAMBDA * selected_feature_ratio. 0.02 means a
# full swing in feature ratio (0 -> 1) costs at most 0.02 F1 - small
# relative to this dataset's typical CV-F1 noise (~0.05-0.1), so it nudges
# toward parsimony without dominating the fitness signal.
GA_LAMBDA = 0.02
GA_SEEDS = [42, 123, 2026, 7, 99]
# GA fitness evaluation uses fewer trees than the final RF models
# (prediction.RF_PARAMS uses 200): CV-F1 ranking between feature subsets is
# stable well before 200 trees, and fitness is evaluated up to population x
# generations times per GA run, so this cuts the dominant per-run cost by
# ~5x. The final all-features-vs-GA comparison still uses the full 200-tree
# RF_PARAMS for both sides, so reported performance metrics are unaffected.
GA_FITNESS_N_ESTIMATORS = 50
