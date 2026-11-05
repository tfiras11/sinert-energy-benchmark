"""Phase 1 - Step 5: structuring of the sliding windows (cahier de charges).

The specification asks for sliding windows of **672 steps** (7 days × 96
steps/day) described by **11 characteristic variables**:

* 4 physical quantities (main meter, EM_VIRTUAL):
    - ``E_abs_kWh``    : active energy counter (kWh [ABS])
    - ``Q_abs_kvarh``  : reactive energy counter (kVARh [ABS])
    - ``P_max_kW``     : max active demand (kW dmd [MAX])
    - ``S_max_kVA``    : max apparent demand (kVA dmd [MAX])
* 7 calendar & cyclical variables:
    - hour sin/cos, day-of-week sin/cos, day-of-year sin/cos, is_weekend

Two artefacts are produced:

1. ``features_15min.parquet`` — the canonical 11-column table on the 15-min
   grid (input of every Phase 2 model);
2. ``windows_daily.npz`` — materialised samples for 1-day-ahead forecasting:
   ``X`` (n, 672, 11) float32, ``Y`` (n, 96, 4) float32 with the next day's
   per-step ΔEa (kWh/step), ΔEr (kvarh/step), P_max (kW), S_max (kVA), plus a
   ``sample_time`` index. Samples overlapping missing data are dropped and
   counted (the August 2025 outage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MAIN_BLOCK, MEDIUM_GAP_STEPS, STEP_MIN

logger = logging.getLogger(__name__)

INPUT_STEPS = 7 * 96          # 672 — one week of history
HORIZON_STEPS = 96            # 1 day ahead

PHYS_FEATURES = ["E_abs_kWh", "Q_abs_kvarh", "P_max_kW", "S_max_kVA"]
CAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "doy_sin", "doy_cos", "is_weekend"]
FEATURES = PHYS_FEATURES + CAL_FEATURES

TARGETS = ["dEa_kWh", "dEr_kvarh", "P_max_kW", "S_max_kVA"]


# --------------------------------------------------------------------------- #
def build_feature_table(cleaned: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble the 11-variable table on the strict 15-min grid.

    ``cleaned`` must be the cleaned 15-min table (grid index); ``raw`` the raw
    ingestion output, used to recover the cumulative counters: reindexing them
    on the grid then interpolating over gaps <= 2 days reproduces exactly the
    even energy spread used by the cleaning step.
    """
    idx = cleaned.index

    e_abs = raw["EM_VIRTUAL__kWh_abs"].reindex(idx) \
        .interpolate(method="linear", limit=MEDIUM_GAP_STEPS)
    q_abs = raw["EM_VIRTUAL__kvarh_abs"].reindex(idx) \
        .interpolate(method="linear", limit=MEDIUM_GAP_STEPS)

    p_max = cleaned[f"{MAIN_BLOCK}__P_max"]
    s_max = cleaned[f"{MAIN_BLOCK}__S_max"]

    feats = pd.DataFrame(index=idx)
    feats["E_abs_kWh"] = e_abs
    feats["Q_abs_kvarh"] = q_abs
    feats["P_max_kW"] = p_max
    feats["S_max_kVA"] = s_max

    h = idx.hour + idx.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    dow = idx.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    doy = idx.dayofyear
    leap = idx.is_leap_year
    year_len = np.where(leap, 366.0, 365.0)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / year_len)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / year_len)
    feats["is_weekend"] = (dow >= 5).astype("float64")

    # quality flag (NOT part of the 11 modelling variables)
    if f"{MAIN_BLOCK}__is_imputed" in cleaned.columns:
        feats["is_imputed"] = cleaned[f"{MAIN_BLOCK}__is_imputed"].astype("float64")

    feats.index.name = "timestamp"
    logger.info("Feature table: %d rows x %d cols (%.2f%% rows with NaN)",
                len(feats), feats.shape[1], 100 * feats[PHYS_FEATURES].isna().any(axis=1).mean())
    cols = FEATURES + (["is_imputed"] if "is_imputed" in feats.columns else [])
    return feats[cols]


# --------------------------------------------------------------------------- #
def build_windows(feats: pd.DataFrame, targets_src: pd.DataFrame,
                  stride: int = 96, input_steps: int = INPUT_STEPS,
                  horizon: int = HORIZON_STEPS
                  ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, int, np.ndarray]:
    """Materialise sliding-window samples for 1-day-ahead forecasting.

    Parameters
    ----------
    feats : feature table (11 columns + optional ``is_imputed``, 15-min grid)
    targets_src : cleaned 15-min table providing the target columns
        (``EM_VIRTUAL__E_kWh``, ``EM_VIRTUAL__Q_kvarh``, ``...__P_max``, ``...__S_max``)
    stride : step between consecutive samples (96 = one sample per day, at 00:00)

    If ``feats`` carries an ``is_imputed`` column (long-gap pattern
    imputation), samples whose input window or target overlaps an imputed
    step are flagged ``sample_imputed=True`` so Phase 2 can exclude them or
    report metrics with and without them.

    Returns
    -------
    X (n, input_steps, 11) float32, Y (n, horizon, 4) float32,
    sample_time (n,) — timestamps of the first *forecast* step,
    n_dropped, sample_imputed (n,) bool.
    """
    phys = feats[PHYS_FEATURES].to_numpy(dtype="float32")
    cal = feats[CAL_FEATURES].to_numpy(dtype="float32")
    X_src = np.concatenate([phys, cal], axis=1)

    imp = (feats["is_imputed"] > 0).to_numpy() if "is_imputed" in feats.columns \
        else np.zeros(len(feats), dtype=bool)

    t_e = targets_src[f"{MAIN_BLOCK}__E_kWh"].to_numpy(dtype="float32")
    t_q = targets_src[f"{MAIN_BLOCK}__Q_kvarh"].to_numpy(dtype="float32")
    t_p = targets_src[f"{MAIN_BLOCK}__P_max"].to_numpy(dtype="float32")
    t_s = targets_src[f"{MAIN_BLOCK}__S_max"].to_numpy(dtype="float32")

    n_total = len(feats)
    starts = np.arange(input_steps, n_total - horizon + 1, stride)
    starts = starts[(starts % 96 == 0)]  # keep targets aligned on midnight

    xs, ys, times, flags = [], [], [], []
    n_dropped = 0
    for t0 in starts:
        x_win = X_src[t0 - input_steps: t0]
        y_e = t_e[t0: t0 + horizon]
        y_q = t_q[t0: t0 + horizon]
        y_p = t_p[t0: t0 + horizon]
        y_s = t_s[t0: t0 + horizon]
        if np.isnan(x_win[:, :4]).any() or np.isnan(y_e).any() or np.isnan(y_q).any() \
                or np.isnan(y_p).any() or np.isnan(y_s).any():
            n_dropped += 1
            continue
        xs.append(x_win)
        ys.append(np.stack([y_e, y_q, y_p, y_s], axis=1))
        times.append(feats.index[t0])
        flags.append(bool(imp[t0 - input_steps: t0 + horizon].any()))

    X = np.asarray(xs, dtype="float32")
    Y = np.asarray(ys, dtype="float32")
    sample_time = pd.DatetimeIndex(times, name="sample_time")
    sample_imputed = np.asarray(flags, dtype=bool)
    logger.info("Windows: %d samples kept, %d dropped (NaN overlap), "
                "%d flagged imputed, X%s Y%s (%.1f MB)",
                len(X), n_dropped, int(sample_imputed.sum()), X.shape, Y.shape,
                (X.nbytes + Y.nbytes) / 1e6)
    return X, Y, sample_time, n_dropped, sample_imputed
