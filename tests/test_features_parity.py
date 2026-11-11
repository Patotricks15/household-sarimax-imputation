import numpy as np
import pandas as pd

from forecaster.features.calendar import calendar_features


def test_batch_and_serving_calendar_features_match():
    full = pd.date_range("2007-01-01", periods=24 * 30, freq="h")
    origin = 24 * 20
    batch = calendar_features(full).iloc[origin:]
    serving = calendar_features(full[origin:])
    pd.testing.assert_frame_equal(batch, serving)


def test_weekly_phase_is_anchored_to_monday_midnight():
    index = pd.DatetimeIndex([pd.Timestamp("2000-01-03 00:00:00"), pd.Timestamp("2000-01-10 00:00:00")])
    features = calendar_features(index, order=1)
    np.testing.assert_allclose(features["weekly_fourier_sin_1"].to_numpy(), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(features["weekly_fourier_cos_1"].to_numpy(), [1.0, 1.0], atol=1e-12)


def test_weekend_flag_matches_calendar():
    index = pd.DatetimeIndex(["2007-01-05", "2007-01-06", "2007-01-07", "2007-01-08"])
    np.testing.assert_array_equal(calendar_features(index)["is_weekend"].to_numpy(), [0.0, 1.0, 1.0, 0.0])
