from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecaster.features.calendar import calendar_features

ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 24)


def fit_sarimax(y: np.ndarray, exog: np.ndarray | None = None):
    """Fit a fixed-order SARIMAX. NaN entries in ``y`` are handled by the Kalman filter."""
    model = SARIMAX(
        y,
        exog=exog,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False, maxiter=200)


def forecast_kalman_sarimax(
    train: np.ndarray,
    train_index: pd.DatetimeIndex,
    future_index: pd.DatetimeIndex,
    use_calendar: bool = True,
) -> np.ndarray:
    """Forecast with the Kalman-filtered SARIMAX fitted on the raw, gap-containing window."""
    exog_train = calendar_features(train_index).to_numpy() if use_calendar else None
    exog_future = calendar_features(future_index).to_numpy() if use_calendar else None
    result = fit_sarimax(train, exog_train)
    mean = result.get_forecast(steps=len(future_index), exog=exog_future).predicted_mean
    return np.clip(np.asarray(mean, dtype=float), 0.0, None)
