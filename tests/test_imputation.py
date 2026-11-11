import numpy as np
import pandas as pd
import pytest

from forecaster.models.imputation import causal_ffill_fill, hour_of_week_fill, kalman_fill


def test_causal_ffill_uses_only_past_values():
    y = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    filled = causal_ffill_fill(y)
    np.testing.assert_allclose(filled[1:], [1.0, 1.0, 1.0, 4.0, 4.0])


def test_causal_ffill_prefix_does_not_depend_on_later_values():
    y = np.array([2.0, np.nan, 3.0, np.nan, np.nan, 9.0])
    base = causal_ffill_fill(y[:4])
    changed = y.copy()
    changed[4:] = 1000.0
    np.testing.assert_allclose(causal_ffill_fill(changed)[:4], base)


def test_hour_of_week_fill_uses_matching_slots():
    index = pd.date_range("2007-01-01", periods=24 * 14, freq="h")
    y = np.tile(np.arange(24, dtype=float), 14)
    y[24 * 7 + 5] = np.nan
    filled = hour_of_week_fill(y, index)
    assert filled[24 * 7 + 5] == pytest.approx(5.0)


def test_hour_of_week_fill_falls_back_when_a_slot_is_fully_hidden():
    index = pd.date_range("2007-01-01", periods=24 * 14, freq="h")
    y = np.tile(np.arange(24, dtype=float), 14)
    slot_mask = (index.day_of_week.to_numpy() == 2) & (index.hour.to_numpy() == 7)
    y[slot_mask] = np.nan
    filled = hour_of_week_fill(y, index)
    assert np.all(np.isfinite(filled))
    assert np.all(filled[slot_mask] == pytest.approx(np.nanmean(y)))


def test_kalman_fill_keeps_observed_values_and_removes_gaps(series):
    index, y = series
    filled = kalman_fill(y, index)
    observed = np.isfinite(y)
    np.testing.assert_array_equal(filled[observed], y[observed])
    assert np.all(np.isfinite(filled))


def test_kalman_fill_recovers_a_gap_in_a_clean_seasonal_series():
    index = pd.date_range("2007-01-01", periods=24 * 21, freq="h")
    hours = np.arange(index.size)
    truth = 2.0 + np.sin(2 * np.pi * hours / 24)
    y = truth.copy()
    y[300:312] = np.nan
    filled = kalman_fill(y, index)
    assert np.mean(np.abs(filled[300:312] - truth[300:312])) < 0.25
