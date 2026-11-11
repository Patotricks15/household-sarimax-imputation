from __future__ import annotations

import numpy as np
import pandas as pd

WEEK_HOURS = 168
# A Monday, so hour-of-week 0 is Monday 00:00 and the phase is stable across batch and serving.
ANCHOR = pd.Timestamp("2000-01-03")
FOURIER_ORDER = 2


def calendar_features(index: pd.DatetimeIndex, order: int = FOURIER_ORDER) -> pd.DataFrame:
    """Weekly Fourier terms and a weekend flag. Known in advance for any future hour."""
    hours = np.rint((index - ANCHOR) / pd.Timedelta(hours=1)).astype(float)
    cols: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * hours / WEEK_HOURS
        cols[f"weekly_fourier_sin_{k}"] = np.sin(angle)
        cols[f"weekly_fourier_cos_{k}"] = np.cos(angle)
    cols["is_weekend"] = (index.day_of_week >= 5).astype(float)
    return pd.DataFrame(cols, index=index)


def missing_hour(valid_minutes: np.ndarray, min_valid_minutes: int = 30) -> np.ndarray:
    """Diagnostic indicator: 1 where the hour was blanked for too few valid minutes."""
    return (np.asarray(valid_minutes) < min_valid_minutes).astype(float)
