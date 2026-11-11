# Household Load: SARIMAX with Kalman Imputation

Hourly forecasting of a single household's active power, with a principled
treatment of missing measurements. Gaps are filled by a Kalman smoother over the
training window only, and the forecasters are benchmarked in a rolling-origin
backtest against a seasonal-naive baseline.

## Dataset

- **Individual household electric power consumption** (UCI, mirrored on Kaggle as
  [`uciml/electric-power-consumption-data-set`](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)).
- One household in Sceaux (France), minute-level, December 2006 to November 2010.
- Target: `Global_active_power` (kW). Missing values are encoded as `?`.
- Check the Kaggle page for the license before redistributing.

The file is not committed. Download it with the Kaggle API, passing the token
inline so it is never written to disk:

```bash
KAGGLE_API_TOKEN=<your-token> .venv/bin/kaggle datasets download -d uciml/electric-power-consumption-data-set -p data --unzip
```

## Cleaning and aggregation

Implemented in [`household.py`](src/forecaster/data/household.py).

- Minute rows are read with `?` as missing (about 1.25% of minutes).
- Minutes are aggregated to hourly mean kW.
- An hour with fewer than `--min-valid-minutes` (default 30) observed minutes is
  set to missing rather than averaged. A mean over a few minutes is a biased
  estimate of the hour.
- The hourly count is kept in `valid_minutes` and used as a diagnostic
  (`missing_hour`).

## Imputation (no leakage)

Gaps are handled inside each training window only, implemented in
[`imputation.py`](src/forecaster/models/imputation.py).

- **Kalman smoother (used by the SARIMAX models):** the state-space model is
  fitted on the raw, gap-containing window. The Kalman filter skips missing
  observations, so no values are invented before fitting. Smoothed estimates
  are used only to fill gaps for the imputation check.
- **Causal forward fill:** the alternative. It uses only past observations, with
  leading gaps set to the observed window mean.
- **Hour-of-week mean:** the alternative. It uses observed training hours at the
  same hour of the week, with the overall observed mean for empty slots.

The smoother conditions on the whole training window, which ends at the fold
origin. It never sees anything after the origin, and test actuals never reach a
model. Metrics count only observed test hours.

## Models

Let `y_t` be hourly kW, with daily period `m = 24` and weekly period `168`.

- **Seasonal naive (m = 24):** `hat y_{T+h} = y_{T+h-24k}`, the last observed
  value in the same hour-of-day slot.
- **SARIMAX (1,0,1)(1,0,1)_24 with constant:** the fixed order is not selected
  per fold, to keep the backtest affordable. Stationarity and invertibility are
  enforced. The forecast is clipped at 0.
- **Calendar exogenous terms** (`sarimax_kalman_calendar`): weekly Fourier terms
  of order 2 and a weekend flag. These are known in advance for any future hour,
  so they are valid regressors. The phase is anchored to a Monday so batch and
  serving features agree.
- Variants: `sarimax_kalman_no_calendar` (no exogenous terms) and
  `sarimax_ffill_calendar` (causal forward fill before fitting).

## Metrics

- **MAE / RMSE:** over observed test hours only.
- **MASE:** `MAE / s`, where `s` is the in-sample MAE of a period-24 seasonal naive
  over the training window. Values below 1 beat the in-sample seasonal naive.

## Backtest protocol

Rolling origin with `--folds 5` folds of `--horizon 168` hours. Each forecaster
receives only the trailing `--window-hours` (default 1344, eight weeks) before the
fold origin, with `forecaster(train, train_index, future_index)`.

Tests check that:

- a forecaster's training windows do not change when the actuals of the
  current fold's test period and later are altered;
- forecasters receive only the bounded pre-origin window;
- Kalman imputation keeps observed values and fills every gap;
- batch and serving calendar features agree for the same origin;
- shuffling training time degrades a seasonal forecast.

## Results

Real data, `python scripts/run_backtest.py --txt data/household_power_consumption.txt`.
Hourly series: 34,589 hours, of which 34,157 are observed after the
30-minute threshold. Full output is in `outputs/run.log` (git-ignored).

Forecasting, mean and std across 5 folds of 168 hours:

| Model | MAE | RMSE | MASE |
|---|---|---|---|
| sarimax_kalman_no_calendar | **0.554** (±0.093) | **0.730** | **0.942** |
| sarimax_kalman_calendar | 0.559 (±0.091) | 0.738 | 0.951 |
| sarimax_ffill_calendar | 0.561 (±0.087) | 0.740 | 0.955 |
| seasonal_naive_24 | 0.564 (±0.111) | 0.800 | 0.962 |

Imputation check: 10% of observed training hours hidden at random in the first
window, then reconstructed and compared with the true values.

| Imputer | Hidden hours | MAE | RMSE |
|---|---|---|---|
| kalman_smoother | 125 | **0.281** | **0.376** |
| hour_of_week_mean | 125 | 0.456 | 0.649 |
| causal_ffill | 125 | 0.484 | 0.706 |

Reading:

- The Kalman smoother is clearly the best imputer. It roughly halves the MAE of
  causal forward fill, which is the main reason to prefer it.
- The SARIMAX forecasters are at most slightly better than seasonal naive on
  MASE, and the differences are well inside the fold standard deviation. No
  forecaster is clearly superior.
- Calendar exogenous terms did not help on this run. A plausible reason is that the
  seasonal component already captures most of the daily pattern. This is a
  hypothesis, not something the backtest tested. This is an honest negative result.
- Five folds is a small sample. Repeat with other windows and horizons before
  drawing a firm conclusion.

## Repository layout

```
conf/features.yaml                    declared feature registry
src/forecaster/data/household.py      minute loading and hourly aggregation
src/forecaster/features/              weekly Fourier and weekend calendar features, missingness flag
src/forecaster/models/                SARIMAX, Kalman and causal imputers, seasonal naive
src/forecaster/evaluation/            rolling-origin folds, metrics, backtest and imputation check
scripts/run_backtest.py               CLI
scripts/check_neural_deps.py          fails if a neural forecasting dependency appears
tests/                                unit, leakage and parity tests
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/check_neural_deps.py
```

Tests and the CLI pin BLAS to one thread. SARIMAX fits are several times slower
under thread oversubscription.

## Scope

This repository covers the forecasting method, imputation, backtest, tests, CI and
this README. Terraform, Floci, architecture diagrams and LaTeX are out of scope
and are not included.
