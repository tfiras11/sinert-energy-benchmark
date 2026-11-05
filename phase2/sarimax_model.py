"""Phase 2 - statistical family: SARIMAX with calendar exogenous variables.

The cahier de charges asks for SARIMAX with exogenous weather/calendar
variables. No weather series is available in the dataset, so the exogenous
block is the calendar: Fourier harmonics of the daily and weekly cycles plus
a weekend indicator (documented limitation - weather can be added later).

Protocol: one SARIMAX per target fitted ONCE on the continuous 15-min train
series, then forecast recursively over val+test with known future exog, and
sliced per forecast day to align with the window samples. Multi-step
recursion means errors accumulate with horizon - reported honestly.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from config import STEP_MIN

logger = logging.getLogger(__name__)

SARIMAX_ORDER = (2, 0, 2)
TARGET_COLUMNS = {
    "dEa_kWh": "EM_VIRTUAL__E_kWh",
    "dEr_kvarh": "EM_VIRTUAL__Q_kvarh",
    "P_max_kW": "EM_VIRTUAL__P_max",
    "S_max_kVA": "EM_VIRTUAL__S_max",
}


def calendar_exog(index: pd.DatetimeIndex) -> np.ndarray:
    """Fourier + weekend exogenous block for a DatetimeIndex."""
    h = index.hour + index.minute / 60.0
    dow = index.dayofweek
    doy = index.dayofyear
    year_len = np.where(index.is_leap_year, 366.0, 365.0)
    cols = []
    for k in (1, 2, 3):                       # daily harmonics
        cols += [np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)]
    for k in (1, 2):                          # weekly harmonics
        cols += [np.sin(2 * np.pi * k * dow / 7), np.cos(2 * np.pi * k * dow / 7)]
    cols += [np.sin(2 * np.pi * doy / year_len), np.cos(2 * np.pi * doy / year_len),
             (dow >= 5).astype(float)]
    return np.stack(cols, axis=1).astype("float64")


def fit_forecast_sarimax(cleaned: pd.DataFrame, sample_time: pd.DatetimeIndex,
                         masks: dict[str, np.ndarray], y_mean: np.ndarray,
                         y_std: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Fit one SARIMAX per target; walk-forward day-ahead predictions.

    Protocol: fit once on the train series, then for every val/test day
    forecast the 96 next steps and append the ACTUAL observations to the
    state filter (``results.append(..., refit=False)`` - one Kalman pass,
    no re-estimation). This is the standard honest evaluation for statistical
    models: each prediction is conditioned on all observed history, and the
    long-horizon drift of a single recursive forecast is avoided.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    idx = cleaned.index
    exog_all = pd.DataFrame(
        calendar_exog(idx),
        index=idx, columns=[f"x{i}" for i in range(calendar_exog(idx).shape[1])])
    train_end_ts = pd.Timestamp("2025-05-31 23:45:00")
    train_mask_idx = idx <= train_end_ts

    # forecast days in chronological order, with their split label
    days = sample_time[masks["val"] | masks["test"]]
    day_split = np.where(masks["val"][masks["val"] | masks["test"]], "val", "test")

    pred = {m: np.full((int(masks[m].sum()), 96, 4), np.nan, dtype="float32")
            for m in ("val", "test")}
    total_params, total_time = 0, 0.0
    step = pd.Timedelta(minutes=STEP_MIN)

    for k, (tgt, col) in enumerate(TARGET_COLUMNS.items()):
        series = cleaned[col].astype("float64")
        t0 = time.perf_counter()
        model = SARIMAX(series[train_mask_idx], exog=exog_all[train_mask_idx],
                        order=SARIMAX_ORDER,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=200)
        total_params += len(res.params)
        logger.info("SARIMAX %s: fitted in %.1fs (llf=%.1f, converged=%s)",
                    tgt, time.perf_counter() - t0, res.llf, res.mle_retvals.get("converged"))

        # ---- walk-forward over the forecast days ---------------------------
        ctr = {"val": 0, "test": 0}
        for d, sp in zip(days, day_split):
            fidx = pd.date_range(d, periods=96, freq=step)
            fc = res.get_forecast(steps=96, exog=exog_all.loc[fidx])
            pred[sp][ctr[sp], :, k] = \
                np.asarray(fc.predicted_mean, dtype="float32").ravel()[:96]
            ctr[sp] += 1
            res = res.append(series.loc[fidx], exog=exog_all.loc[fidx], refit=False)
        total_time += time.perf_counter() - t0

    pred["n_params"] = total_params
    pred["train_time_s"] = total_time
    return pred
