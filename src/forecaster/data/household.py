from __future__ import annotations

from pathlib import Path

import pandas as pd

MINUTES_PER_HOUR = 60


def load_minutes(path: str | Path) -> pd.Series:
    """Read the minute-level Global_active_power series (kW). '?' marks missing."""
    frame = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        usecols=["Date", "Time", "Global_active_power"],
        dtype={"Global_active_power": float},
        low_memory=False,
    )
    stamps = pd.to_datetime(frame["Date"] + " " + frame["Time"], format="%d/%m/%Y %H:%M:%S")
    series = pd.Series(frame["Global_active_power"].to_numpy(), index=pd.DatetimeIndex(stamps), name="kw")
    if not series.index.is_monotonic_increasing:
        raise ValueError("minute timestamps must be sorted")
    return series


def to_hourly(minutes: pd.Series, min_valid_minutes: int = 30) -> pd.DataFrame:
    """Hourly mean kW, NaN where fewer than ``min_valid_minutes`` minutes were observed.

    Partial hours are blanked rather than averaged, because a mean over a few
    minutes is a biased estimate of the hour.
    """
    if not 1 <= min_valid_minutes <= MINUTES_PER_HOUR:
        raise ValueError("min_valid_minutes must be between 1 and 60")
    grouped = minutes.resample("h")
    valid = grouped.count()
    mean = grouped.mean().where(valid >= min_valid_minutes)
    return pd.DataFrame({"kw": mean, "valid_minutes": valid.astype(int)})
