import numpy as np
import pytest
from titanic.data import load_titanic, FEATURE_COLS


@pytest.fixture(scope="module")
def split():
    return load_titanic()


def test_shapes(split):
    n_train, n_test = len(split.X_train), len(split.X_test)
    assert n_train + n_test == 891
    assert split.X_train.shape[1] == len(FEATURE_COLS)
    assert split.X_test.shape[1] == len(FEATURE_COLS)
    assert split.y_train.shape == (n_train,)
    assert split.y_test.shape == (n_test,)


def test_dtypes(split):
    for arr in (split.X_train, split.X_test, split.X_train_scaled, split.X_test_scaled):
        assert arr.dtype == np.float32
    assert split.y_train.dtype == np.float32
    assert split.y_test.dtype == np.float32


def test_no_nans(split):
    for arr in (split.X_train, split.X_test, split.X_train_scaled, split.X_test_scaled):
        assert not np.isnan(arr).any()


def test_labels_binary(split):
    assert set(np.unique(split.y_train)).issubset({0.0, 1.0})
    assert set(np.unique(split.y_test)).issubset({0.0, 1.0})


def test_scaled_has_zero_mean(split):
    # StandardScaler fitted on train — train mean should be ~0
    assert abs(split.X_train_scaled.mean()) < 0.1


def test_no_leakage_imputation(split):
    # Age column (index 2) must have no NaNs after pipeline
    assert not np.isnan(split.X_train[:, 2]).any()
    assert not np.isnan(split.X_test[:, 2]).any()
