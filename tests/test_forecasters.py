import numpy as np
import pandas as pd

from forecaster.models.sarimax import forecast_kalman_sarimax


def _seasonal_with_gaps(n: int = 24 * 21, seed: int = 2):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2007-01-01", periods=n, freq="h")
    hours = np.arange(n)
    y = 2.0 + np.sin(2 * np.pi * hours / 24) + 0.5 * np.cos(2 * np.pi * hours / 168)
    y = y + rng.normal(0, 0.05, n)
    y[rng.random(n) < 0.05] = np.nan
    return index, y


def test_sarimax_forecast_is_finite_and_non_negative():
    index, y = _seasonal_with_gaps()
    train_index, future_index = index[:-24], index[-24:]
    pred = forecast_kalman_sarimax(y[:-24], train_index, future_index)
    assert pred.shape == (24,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)


def test_sarimax_forecast_tracks_the_seasonal_signal():
    index, y = _seasonal_with_gaps()
    train_index, future_index = index[:-24], index[-24:]
    pred = forecast_kalman_sarimax(y[:-24], train_index, future_index)
    truth = 2.0 + np.sin(2 * np.pi * np.arange(len(y) - 24, len(y)) / 24)
    assert np.mean(np.abs(pred - truth)) < 0.5
