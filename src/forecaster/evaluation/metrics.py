from __future__ import annotations

import numpy as np


def _observed(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only positions where the actual value was observed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("actuals and forecasts must have the same shape")
    keep = np.isfinite(y_true)
    return y_true[keep], y_pred[keep]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual, pred = _observed(y_true, y_pred)
    return float(np.mean(np.abs(actual - pred))) if actual.size else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual, pred = _observed(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - pred) ** 2))) if actual.size else float("nan")


def seasonal_scale(train: np.ndarray, period: int = 24) -> float:
    """Mean absolute seasonal-naive error on the training window, using only observed pairs."""
    train = np.asarray(train, dtype=float)
    diffs = np.abs(train[period:] - train[:-period])
    diffs = diffs[np.isfinite(diffs)]
    return float(diffs.mean()) if diffs.size else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, train: np.ndarray, period: int = 24) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE. NaN when the scale is zero."""
    scale = seasonal_scale(train, period)
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale
