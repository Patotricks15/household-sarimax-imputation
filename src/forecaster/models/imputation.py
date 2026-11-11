from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.features.calendar import calendar_features
from forecaster.models.sarimax import fit_sarimax


def kalman_fill(train: np.ndarray, train_index: pd.DatetimeIndex) -> np.ndarray:
    """Replace NaN gaps with the Kalman smoothed estimate from the SARIMAX fitted on this window.

    The smoother conditions on the whole training window, including later
    training hours, but never on anything after the window's end. Observed
    values are returned unchanged.
    """
    train = np.asarray(train, dtype=float)
    result = fit_sarimax(train, calendar_features(train_index).to_numpy())
    smoothed = np.asarray(result.smoother_results.smoothed_forecasts[0], dtype=float)
    out = train.copy()
    gaps = np.isnan(out)
    out[gaps] = smoothed[gaps]
    return out


def causal_ffill_fill(train: np.ndarray) -> np.ndarray:
    """Forward-fill gaps from past observations only.

    Leading gaps, which have no past value, fall back to the mean of the observed
    training window.
    """
    series = pd.Series(np.asarray(train, dtype=float))
    filled = series.ffill()
    fallback = float(series.mean())
    return filled.fillna(fallback).to_numpy()


def hour_of_week_fill(train: np.ndarray, train_index: pd.DatetimeIndex) -> np.ndarray:
    """Fill gaps with the mean of observed training hours at the same hour of the week."""
    train = np.asarray(train, dtype=float)
    slot = train_index.day_of_week.to_numpy() * 24 + train_index.hour.to_numpy()
    frame = pd.DataFrame({"slot": slot, "y": train})
    profile = frame.groupby("slot")["y"].mean()
    fill = pd.Series(slot).map(profile).to_numpy(dtype=float)
    overall = float(np.nanmean(train)) if np.any(np.isfinite(train)) else 0.0
    fill = np.where(np.isnan(fill), overall, fill)
    out = train.copy()
    gaps = np.isnan(out)
    out[gaps] = fill[gaps]
    return out
