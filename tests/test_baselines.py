import numpy as np
import pandas as pd
import pytest

from forecaster.models.baselines import seasonal_naive


def test_seasonal_naive_repeats_the_last_cycle_exactly():
    cycle = np.arange(24, dtype=float)
    train = np.tile(cycle, 5)
    np.testing.assert_allclose(seasonal_naive(train, 48, period=24), np.tile(cycle, 2))


def test_seasonal_naive_uses_last_observed_value_per_slot():
    train = np.tile(np.arange(24, dtype=float), 3)
    train[24 * 2 + 3] = np.nan
    pred = seasonal_naive(train, 24, period=24)
    assert pred[3] == pytest.approx(3.0)


def test_seasonal_naive_rejects_empty_input():
    with pytest.raises(ValueError):
        seasonal_naive(np.array([]), 5)
