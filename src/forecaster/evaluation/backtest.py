from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_end: int  # exclusive index into the series
    test_start: int
    test_end: int  # exclusive


def rolling_origin_folds(
    n: int,
    horizon: int,
    n_folds: int,
    step: int | None = None,
    gap: int = 0,
    min_train: int = 1,
) -> Iterator[Fold]:
    """Yield rolling-origin folds that never let test data leak into training.

    The last fold ends at ``n``. Earlier origins move back by ``step`` (default:
    ``horizon``). ``gap`` skips observations between train and test.
    """
    if horizon < 1 or n_folds < 1:
        raise ValueError("horizon and n_folds must be positive")
    step = horizon if step is None else step
    for k in range(n_folds - 1, -1, -1):
        test_end = n - k * step
        test_start = test_end - horizon
        train_end = test_start - gap
        if train_end < min_train or test_start < 0:
            raise ValueError("series too short for the requested folds")
        yield Fold(train_end=train_end, test_start=test_start, test_end=test_end)
