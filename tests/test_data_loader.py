from src.config import RANDOM_SEED, TARGET_COLUMN
from src.data_loader import EXPECTED_SHAPE, load_raw_data
from src.preprocessing import train_test_split_stratified


def test_load_raw_data_shape_and_target():
    df = load_raw_data()
    assert df.shape == EXPECTED_SHAPE
    assert TARGET_COLUMN in df.columns
    assert df.isna().sum().sum() == 0


def test_split_is_stratified_and_reproducible():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.05  # stratified => similar class ratio

    X_train2, X_test2, y_train2, y_test2 = train_test_split_stratified(df)
    assert list(X_train.index) == list(X_train2.index)  # same seed => same split


def test_split_excludes_target_from_features():
    df = load_raw_data()
    X_train, X_test, _, _ = train_test_split_stratified(df)
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
