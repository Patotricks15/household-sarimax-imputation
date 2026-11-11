import os

# Single-threaded BLAS: SARIMAX fits are much faster without thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest


def make_household_series(n_hours: int = 24 * 24, seed: int = 0):
    """Hourly series with daily and weekly seasonality, observation noise, and gaps."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2007-01-01", periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    daily = 1.5 * np.sin(2 * np.pi * hours / 24)
    weekly = 0.8 * np.cos(2 * np.pi * hours / 168)
    y = 2.0 + daily + weekly + rng.normal(0, 0.15, n_hours)
    gaps = rng.random(n_hours) < 0.08
    gaps[100:112] = True
    y = y.copy()
    y[gaps] = np.nan
    return index, y


@pytest.fixture
def series():
    return make_household_series()


@pytest.fixture
def small_backtest_args():
    return {"horizon": 24, "n_folds": 3, "window": 24 * 10}
