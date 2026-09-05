"""Load and validate the raw heart failure dataset. Never writes to data/raw/."""

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN

EXPECTED_SHAPE = (299, 13)
EXPECTED_COLUMNS = {
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
    "sex", "smoking", "time", "DEATH_EVENT",
}


def load_raw_data() -> pd.DataFrame:
    """Load the untouched raw CSV and fail clearly if it no longer matches
    the verified reference shape/columns (master prompt section 15: fail
    clearly if the raw dataset unexpectedly changes)."""
    df = pd.read_csv(RAW_DATA_PATH)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")
    if set(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns {set(df.columns)}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    return df
