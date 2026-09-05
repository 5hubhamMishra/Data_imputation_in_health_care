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
