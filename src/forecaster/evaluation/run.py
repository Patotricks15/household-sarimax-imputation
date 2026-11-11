from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from forecaster.evaluation.backtest import rolling_origin_folds
from forecaster.evaluation.metrics import mae, mase, rmse
from forecaster.features.calendar import calendar_features
from forecaster.models.baselines import seasonal_naive
from forecaster.models.imputation import causal_ffill_fill, hour_of_week_fill, kalman_fill
from forecaster.models.sarimax import fit_sarimax, forecast_kalman_sarimax

# (train values, train index, future index) -> forecast of length len(future index)
Forecaster = Callable[[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex], np.ndarray]


def _seasonal_naive(train, train_index, future_index):
    return seasonal_naive(train, len(future_index), period=24)


def _sarimax_calendar(train, train_index, future_index):
    return forecast_kalman_sarimax(train, train_index, future_index, use_calendar=True)


def _sarimax_no_calendar(train, train_index, future_index):
    return forecast_kalman_sarimax(train, train_index, future_index, use_calendar=False)


def _sarimax_ffill_calendar(train, train_index, future_index):
    filled = causal_ffill_fill(train)
    exog_train = calendar_features(train_index).to_numpy()
    exog_future = calendar_features(future_index).to_numpy()
    result = fit_sarimax(filled, exog_train)
    mean = result.get_forecast(steps=len(future_index), exog=exog_future).predicted_mean
    return np.clip(np.asarray(mean, dtype=float), 0.0, None)


DEFAULT_FORECASTERS: dict[str, Forecaster] = {
    "seasonal_naive_24": _seasonal_naive,
    "sarimax_kalman_calendar": _sarimax_calendar,
    "sarimax_kalman_no_calendar": _sarimax_no_calendar,
    "sarimax_ffill_calendar": _sarimax_ffill_calendar,
}


def evaluate_backtest(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    horizon: int,
    n_folds: int,
    window: int,
    forecasters: Mapping[str, Forecaster] = DEFAULT_FORECASTERS,
) -> pd.DataFrame:
    """One row per (model, fold). Metrics use only test hours that were observed.

    Each forecaster receives only the trailing ``window`` hours before its fold
    origin. Test actuals are never passed to a model.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[0] != len(index):
        raise ValueError("values and index must have the same length")
    rows: list[dict] = []
    for fold_no, fold in enumerate(rolling_origin_folds(values.size, horizon, n_folds)):
        start = max(0, fold.train_end - window)
        train = values[start : fold.train_end]
        train_index = index[start : fold.train_end]
        actual = values[fold.test_start : fold.test_end]
        future_index = index[fold.test_start : fold.test_end]
        for name, forecast in forecasters.items():
            pred = np.asarray(forecast(train, train_index, future_index), dtype=float)
            if pred.shape != (horizon,) or not np.all(np.isfinite(pred)):
                raise ValueError(f"{name} returned an invalid forecast in fold {fold_no}")
            rows.append(
                {
                    "model": name,
                    "fold": fold_no,
                    "test_start": future_index[0],
                    "n_observed_test": int(np.isfinite(actual).sum()),
                    "mae": mae(actual, pred),
                    "rmse": rmse(actual, pred),
                    "mase": mase(actual, pred, train, period=24),
                }
            )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across folds for each model."""
    grouped = rows.groupby("model", sort=False)
    out = grouped[["mae", "rmse", "mase"]].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


def masked_imputation_check(
    train: np.ndarray,
    train_index: pd.DatetimeIndex,
    frac: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Hide a fraction of observed training hours and measure each imputer's error on them.

    Everything here is inside the training window, so no test data is used.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    train = np.asarray(train, dtype=float)
    observed = np.flatnonzero(np.isfinite(train))
    rng = np.random.default_rng(seed)
    hidden = rng.choice(observed, size=max(1, int(frac * observed.size)), replace=False)
    masked = train.copy()
    truth = train[hidden]
    masked[hidden] = np.nan

    imputers = {
        "kalman_smoother": lambda x: kalman_fill(x, train_index),
        "causal_ffill": causal_ffill_fill,
        "hour_of_week_mean": lambda x: hour_of_week_fill(x, train_index),
    }
    rows = []
    for name, impute in imputers.items():
        filled = impute(masked)
        pred = filled[hidden]
        rows.append({"imputer": name, "n_hidden": int(hidden.size), "mae": mae(truth, pred), "rmse": rmse(truth, pred)})
    return pd.DataFrame(rows)
