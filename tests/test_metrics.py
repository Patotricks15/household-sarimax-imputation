import numpy as np
import pytest

from forecaster.evaluation.metrics import mae, mase, rmse, seasonal_scale


def test_mae_and_rmse_ignore_unobserved_actuals():
    y = np.array([1.0, np.nan, 3.0])
    pred = np.array([2.0, 100.0, 1.0])
    assert mae(y, pred) == pytest.approx(1.5)
    assert rmse(y, pred) == pytest.approx(np.sqrt(2.5))


def test_all_unobserved_gives_nan():
    assert np.isnan(mae(np.full(3, np.nan), np.ones(3)))


def test_seasonal_scale_is_mean_abs_seasonal_difference():
    train = np.tile(np.array([0.0, 2.0]), 6)
    assert seasonal_scale(train, period=2) == pytest.approx(0.0)
    train2 = np.array([1.0, 2.0, 4.0, 4.0])
    assert seasonal_scale(train2, period=2) == pytest.approx(np.mean([3.0, 2.0]))


def test_mase_is_nan_for_constant_training_window():
    assert np.isnan(mase(np.ones(3), np.zeros(3), np.full(10, 5.0), period=2))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.ones(3), np.ones(2))
