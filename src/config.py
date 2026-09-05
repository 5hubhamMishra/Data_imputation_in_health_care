"""Central configuration for the heart failure imputation/GA pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "heart_failure_clinical_records_dataset.csv"

TARGET_COLUMN = "DEATH_EVENT"

# Columns that must never be masked by the missingness framework or dropped
# by preprocessing/feature selection.
PROTECTED_COLUMNS = [TARGET_COLUMN]

RANDOM_SEED = 42
TEST_SIZE = 0.2

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
