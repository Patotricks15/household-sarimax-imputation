import numpy as np
import pandas as pd
import pytest

from forecaster.features.calendar import calendar_features, missing_hour
from forecaster.features.spec import load_declared_features

REGISTRY = "conf/features.yaml"


def test_declared_calendar_features_match_implementation():
    declared = load_declared_features(REGISTRY)
    implemented = set(calendar_features(pd.date_range("2007-01-01", periods=5, freq="h")).columns)
    calendar_declared = {n for n in declared if not n.startswith("missing_")}
    assert calendar_declared == implemented


def test_missingness_feature_is_declared_as_diagnostic():
    declared = load_declared_features(REGISTRY)
    assert "missing_hour" in declared


def test_duplicate_feature_names_are_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("features:\n  - name: a\n  - name: a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_declared_features(path)


def test_missing_hour_flags_partial_hours():
    flags = missing_hour(np.array([60, 29, 30, 0]), min_valid_minutes=30)
    np.testing.assert_array_equal(flags, [0.0, 1.0, 0.0, 1.0])


def test_calendar_index_is_regular_and_sorted():
    index = pd.date_range("2007-01-01", periods=48, freq="h")
    features = calendar_features(index)
    assert features.index.equals(index)
    assert features.index.is_monotonic_increasing
    assert np.all(np.isfinite(features.to_numpy()))
