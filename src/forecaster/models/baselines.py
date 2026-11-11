from __future__ import annotations

import numpy as np


def seasonal_naive(train: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    """Repeat the most recent observed value for each position in the cycle.

    Positions are indexed from the start of ``train``, so the forecast for step
    ``h`` uses the slot ``(len(train) + h) % period``. Slots with no observation
    fall back to the observed training mean.
    """
    train = np.asarray(train, dtype=float)
    if train.size == 0 or horizon < 1:
        raise ValueError("need a non-empty training window and a positive horizon")
    last = np.full(period, np.nan)
    for i, value in enumerate(train):
        if not np.isnan(value):
            last[i % period] = value
    fallback = float(np.nanmean(train)) if np.any(~np.isnan(train)) else 0.0
    slots = (train.size + np.arange(horizon)) % period
    out = last[slots]
    return np.where(np.isnan(out), fallback, out)
