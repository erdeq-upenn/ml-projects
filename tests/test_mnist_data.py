import numpy as np
import pytest
from mnist.data import load_mnist


@pytest.fixture(scope="module")
def split():
    return load_mnist()


def test_shapes(split):
    assert split.X_train.shape == (60000, 784)
    assert split.X_test.shape == (10000, 784)
    assert split.y_train.shape == (60000,)
    assert split.y_test.shape == (10000,)


def test_dtypes(split):
    for arr in (split.X_train, split.X_test, split.X_train_scaled, split.X_test_scaled):
        assert arr.dtype == np.float32
    assert split.y_train.dtype == np.int64
    assert split.y_test.dtype == np.int64


def test_no_nans(split):
    for arr in (split.X_train, split.X_test, split.X_train_scaled, split.X_test_scaled):
        assert not np.isnan(arr).any()


def test_pixel_range(split):
    # Raw pixels normalized to [0, 1]
    assert split.X_train.min() >= 0.0
    assert split.X_train.max() <= 1.0


def test_labels_range(split):
    assert split.y_train.min() == 0
    assert split.y_train.max() == 9
    assert len(np.unique(split.y_train)) == 10


def test_scaled_has_zero_mean(split):
    assert abs(split.X_train_scaled.mean()) < 0.1
