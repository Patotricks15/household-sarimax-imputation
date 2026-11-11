import numpy as np
import pandas as pd

from forecaster.evaluation.run import evaluate_backtest
from forecaster.models.baselines import seasonal_naive


def _series(n: int = 24 * 40, seed: int = 1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2007-01-01", periods=n, freq="h")
    hours = np.arange(n)
    values = 1.0 + np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.1, n)
    return index, values


def test_forecasters_never_see_test_period_values():
    index, values = _series()
    seen: list[np.ndarray] = []

    def spy(train, train_index, future_index):
        seen.append(np.asarray(train).copy())
        return seasonal_naive(train, len(future_index), period=24)

    perturbed = values.copy()
    perturbed[-24:] += 50.0

    evaluate_backtest(values, index, horizon=24, n_folds=3, window=24 * 10, forecasters={"spy": spy})
    clean_seen = list(seen)
    seen.clear()
    evaluate_backtest(perturbed, index, horizon=24, n_folds=3, window=24 * 10, forecasters={"spy": spy})

    assert len(clean_seen) == len(seen) == 3
    for clean, dirty in zip(clean_seen, seen):
        np.testing.assert_array_equal(clean, dirty)


def test_forecaster_window_is_bounded_by_origin():
    index, values = _series()
    lengths: list[int] = []

    def spy(train, train_index, future_index):
        lengths.append(len(train))
        assert train_index[-1] < future_index[0]
        return np.zeros(len(future_index))

    evaluate_backtest(values, index, horizon=24, n_folds=2, window=24 * 10, forecasters={"spy": spy})
    assert all(length <= 24 * 10 for length in lengths)


def test_shuffling_training_time_hurts_a_seasonal_forecast():
    index, values = _series()
    train, future = values[:-24], values[-24:]
    ordered = seasonal_naive(train, 24, period=24)
    rng = np.random.default_rng(3)
    shuffled = seasonal_naive(rng.permutation(train), 24, period=24)
    ordered_err = np.mean(np.abs(future - ordered))
    shuffled_err = np.mean(np.abs(future - shuffled))
    assert shuffled_err > ordered_err
