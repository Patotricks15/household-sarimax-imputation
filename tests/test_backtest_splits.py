import pytest

from forecaster.evaluation.backtest import rolling_origin_folds


def test_folds_are_ordered_and_end_at_n():
    folds = list(rolling_origin_folds(n=100, horizon=10, n_folds=3))
    assert [f.test_end for f in folds] == [80, 90, 100]
    assert all(f.test_end - f.test_start == 10 for f in folds)


def test_test_windows_do_not_overlap():
    folds = list(rolling_origin_folds(n=200, horizon=24, n_folds=5))
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.test_end <= nxt.test_start


def test_forecast_gap_is_respected():
    for f in rolling_origin_folds(n=100, horizon=5, n_folds=4, gap=3):
        assert f.test_start - f.train_end == 3


def test_origins_are_never_shuffled():
    folds = list(rolling_origin_folds(n=120, horizon=10, n_folds=4))
    assert [f.train_end for f in folds] == sorted(f.train_end for f in folds)


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        list(rolling_origin_folds(n=20, horizon=10, n_folds=3))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        list(rolling_origin_folds(n=50, horizon=0, n_folds=2))
