from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Single-threaded BLAS: SARIMAX fits are much faster without thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from forecaster.data.household import load_minutes, to_hourly  # noqa: E402
from forecaster.evaluation.run import evaluate_backtest, masked_imputation_check, summarize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", type=Path, required=True, help="path to household_power_consumption.txt")
    parser.add_argument("--horizon", type=int, default=168, help="forecast horizon in hours")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-hours", type=int, default=1344, help="training window (default 8 weeks)")
    parser.add_argument("--min-valid-minutes", type=int, default=30)
    parser.add_argument("--imputation-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/backtest.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hourly = to_hourly(load_minutes(args.txt), min_valid_minutes=args.min_valid_minutes)
    values = hourly["kw"].to_numpy()
    index = hourly.index
    print(f"hours: {values.size}, observed: {int((~hourly['kw'].isna()).sum())}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = evaluate_backtest(values, index, args.horizon, args.folds, args.window_hours)
        cutoff = values.size - args.folds * args.horizon
        imputation = masked_imputation_check(
            values[cutoff - args.window_hours : cutoff],
            index[cutoff - args.window_hours : cutoff],
            frac=args.imputation_frac,
            seed=args.seed,
        )

    print(summarize(rows).round(3).to_string(index=False))
    print()
    print(imputation.round(4).to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)
    imputation.to_csv(args.out.with_name("imputation_check.csv"), index=False)
    print(f"\nPer-fold rows written to {args.out}")


if __name__ == "__main__":
    main()
