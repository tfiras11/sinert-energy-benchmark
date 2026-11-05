# SInERT Data Pipeline — Phase 1 & 2

Code for the project *"Benchmark multi-critères et Modélisation IA
pour la prédiction énergétique et l'estimation de la facturation de SUP'COM"*.

## Layout

```
sg_project/
├── config.py              paths, meter blocks, variable map, thresholds, calendars
├── ingestion.py           step 1 — load & concatenate the raw VMU-C exports (CSV+XLSX)
├── cleaning.py            step 2 — 15-min grid, counter deltas, imputation, flags
├── behavior.py            step 4 — energy totals, peaks, base load, activation, variability
├── visualize.py           step 3 — PNG figure set
├── structure.py           step 5 — 11-variable table + 672-step sliding windows
├── run_phase1.py          CLI: ingestion + cleaning
├── run_phase1_analysis.py CLI: figures + behaviour parameters + structuring
├── run_phase2.py          CLI: benchmark (baselines, SARIMAX, ML locally)
├── phase2/
│   ├── data_protocol.py   chronological splits, scalers, Colab npz export
│   ├── baselines.py       seasonal persistence t-96 / t-672
│   ├── sarimax_model.py   SARIMAX (2,0,2) + calendar Fourier exog, walk-forward
│   ├── ml_models.py       RandomForest (trained locally)
│   ├── benchmark.py       metrics engine (MAE/RMSE/R2/MAPE/PDE) + result store
│   ├── colab_benchmark.py ALL Colab-trained models: XGBoost, LightGBM, SVR,
│   │                      SARIMAX, and the 7 DL architectures (PyTorch)
│   └── merge_colab.py     merge Colab results back into the benchmark table
└── output/                parquet tables, reports, figures, windows, benchmark
```

## Usage

```bash
python -I sg_project/run_phase1.py                  # steps 1 & 2 (~2 min)
python -I sg_project/run_phase1_analysis.py         # steps 3, 4 & 5 (~10 s)
python -I sg_project/run_phase2.py                  # local benchmark part
python -I sg_project/run_phase2.py --only sarimax   # re-run one section
```

## Phase 2 — benchmark protocol

Task for every model: **1-day-ahead forecasting**
input = past week (672 steps × 11 channels) → output = next day
(96 steps × 4 targets: ΔEa, ΔEr per step; Pmax, Smax per step).

Chronological split (no shuffling): **train** ≤ 2025-05-31 (1,059 days) ·
**val** 2025-06-01→10-31 (153) · **test** 2025-11-01→2026-01-31 (92, 100 % real
data — no imputed days). Scaling fit on train only; `windows_scaled.npz`
bundles everything Colab needs.

Metrics (physical units): MAE/RMSE/R² on the 96-step profiles; MAPE on daily
energy sums; PDE (peak demand error) on daily peaks; n_params + training time.
Every model's val/test predictions are stored in
`output/benchmark/predictions/<model>.npz` for Phase 3 (ΔFacture) and Phase 5.

**Where each model runs** (local machine: 5.9 GB RAM — 0.9 GB free):

| Runs locally | Runs on Colab (`phase2/colab_benchmark.py`) |
|---|---|
| naive_96, naive_672, RandomForest | SARIMAX (walk-forward), XGBoost, LightGBM, SVR, LSTM, Bi-LSTM, GRU, CNN-1D, TCN/ResNet-1D, Transformer, CNN-LSTM |

Colab workflow: upload `windows_scaled.npz` + `colab_benchmark.py` →
`!python colab_benchmark.py --models ml,sarimax,dl --epochs 60` → download
`colab_results.csv` + `colab_predictions.npz` into `output/benchmark/` →
`python -I merge_colab.py`. The merged `benchmark_results.csv` then contains
every model, and Phase 3/5 consume the single prediction store.

## What step 1 (ingestion) does

- Recursively discovers every `Daily Report_Main AC meter.csv` (≈ 690 files,
  July 2022 → January 2026).
- Normalizes headers (raw ones carry trailing spaces), parses
  `Date (DD-MM-YYYY) + Time (HH:MM:SS)`, drops the empty trailing `24:00:00` row.
- Keeps, for each of the **14 meter blocks**, the 6 variables of the cahier de
  charges: `kWh [ABS]`, `kvarh [ABS]`, `kW sys [AVG]`, `kvar sys [AVG]`,
  `kW dmd [MAX]`, `kVA dmd [MAX]`.
- Concatenates everything into one wide frame indexed by timestamp
  (`{block}__{var}` columns), sorts, drops duplicate timestamps.

## What step 2 (cleaning & imputation) does

- Re-indexes on a strict 15-minute grid and flags absent slots
  (`flag_slot_missing`).
- **Cumulative counters** → per-step energy: first difference; spans of missing
  readings ≤ 2 days are spread evenly (totals preserved); counter resets
  (negative jumps) are excluded and logged. Counters are renamed to
  `E_kWh` / `Q_kvarh` (these are the ΔEa / ΔEr targets of Phase 2).
- **Direct measurements** — imputation by gap duration:
  | Gap | Rule |
  |---|---|
  | ≤ 1 h | linear (time) interpolation |
  | ≤ 2 days | seasonal-naive: same time previous week (t−672), fallback previous day (t−96) |
  | > 2 days | left as NaN, counted in the report |
- Adds `*_imputed` boolean flags for the main meter (EM_VIRTUAL).
- **Defective-value rule** (cahier de charges 1.2): demand peaks reading exactly
  ~0 while the meter reports P_avg > 1 kW are reporting glitches
  (55 cases on EM_VIRTUAL, more on sub-meters) → set to NaN and imputed;
  genuine shutdowns (P_avg ≈ 0) are preserved.
- Derives `EM_VIRTUAL__PF_derived = P / √(P²+Q²)` — every raw PF column of the
  virtual meter is empty in the exports.

## Outputs

| File | Content |
|---|---|
| `raw_ingested.parquet` | raw values as read (NaN where `---`) |
| `cleaned_15min.parquet` | cleaned/imputed table, 15-min grid |
| `cleaning_report.csv` | per block/variable stats (NaN, imputed, resets, long gaps) |
| `ingest_log.csv` | per-file bookkeeping (rows kept/dropped, missing columns) |
| `summary.txt` | run summary + verification lines |

## Phase 1.3 — Visualisation (`output/figures/`)

10 PNG figures: daily/monthly energy, weekday & seasonal & academic profiles,
load heatmap, base-vs-peak evolution, derived PF timeline, annual boxplots,
per-block stacked annual energy.

## Phase 1.4 — Behaviour parameters (`output/behavior/`)

| File | Content |
|---|---|
| `energy_{daily,weekly,monthly,annual}.csv` | energy totals per meter block |
| `daily_parameters.csv` | per day: E, Pmin + hour (base load), Pmax/Smax + hours (peaks), peak duration (steps with P ≥ 90 % of daily max), activation/deactivation hours (P ≥ min + 15 % of daily range), std/var/CV, calendar flags |
| `monthly_parameters.csv` | monthly aggregation of the above |
| `profile_*.csv` | mean ± std 15-min profiles by weekday, STEG season, academic period |

Calendars (documented assumptions in `config.py`): tariff seasons per the STEG
contract (hiver Sep–May / été Jun–Aug); academic periods = study weeks outside
summer break (Jul-Aug) and winter break (20 Dec–6 Jan).

## Phase 1.5 — Structuring (`output/structured/`)

- `features_15min.parquet` — the **11 variables** of the cahier de charges:
  4 physical (E_abs kWh [ABS], Q_abs kvarh [ABS], P_max kW dmd, S_max kVA dmd)
  + 7 calendar/cyclical (hour sin/cos, dow sin/cos, doy sin/cos, is_weekend).
- `windows_daily.npz` — samples for 1-day-ahead forecasting:
  `X (1266, 672, 11)` float32 (one week of history, 7 days × 96 steps),
  `Y (1266, 96, 4)` float32 (next day's per-step ΔEa, ΔEr, Pmax, Smax),
  `sample_time`, `features`, `targets`, metadata. Samples overlapping missing
  data are dropped (38 dropped, mostly the August 2025 outage).

## Verified ground truths

- Sept 2025 EM_VIRTUAL consumption was cross-checked against raw counter
  differences (cleaned total within +0.4 % of the raw reference).
- The real STEG bill of 09/2026 was used as an order-of-magnitude ground
  truth for the site consumption (EM_VIRTUAL ≈ the TGBT_TR2 feed).
- `EM_VIRTUAL` and `TGBT_TR2` are identical feeds (checked on Sept 2025).

## Known data-coverage facts

- **Analysis starts 2022-07-01** (`ANALYSIS_START` in `config.py`): the
  EM_VIRTUAL counter produced nothing before 2022-06-28, so June 2022
  (only 3 partial days of data) is dropped from the cleaned data.
- November 2023 was exported as `.xlsx` (column names carry a trailing index).

## Long-gap imputation (August 2025 outage)

August 2025 is absent from the exports (the folder named `August2025` actually
contains May 2025 files — mislabelled on OneDrive). With
`IMPUTE_LONG_GAPS = True` (analysis stage only), the gap is filled by a
**pattern-based method** (`cleaning.impute_long_gaps_by_pattern`):

1. **Reference shapes**: mean per-step value by (weekday, time-of-day) over the
   same calendar month of every other year with >= 80 % valid data
   (Aug 2022, 2023, 2024 — all complete).
2. **Energies (E, Q)**: the shapes are scaled to conserve **exactly** the true
   energy of the gap, read from the cumulative counter span
   (last reading before → first reading after = **67,920 kWh / 14,745 kvarh**).
   The level is a measurement, never a guess.
3. **P_max / S_max** (no counter exists): shapes scaled by the ratio of the
   neighbouring-month mean level (Jul+Sep 2025 vs Jul+Sep of reference years):
   ratio_P = 1.25, ratio_S = 1.24 (documented approximation).
4. **P_avg / Q_avg** derived from the imputed energies (P = E / Δt), keeping
   the measurement identities consistent.
5. Every imputed step carries the flag `EM_VIRTUAL__is_imputed`
   (`features_15min.parquet` column `is_imputed`); window samples overlapping
   the gap are flagged `sample_imputed` in `windows_daily.npz` so Phase 2 can
   report metrics with and without them.

Result: the feature table is 100 % NaN-free; `cleaned_15min.parquet` (raw
cleaning output, gap left as NaN) and `cleaned_15min_imputed.parquet` are both
kept for transparency.
