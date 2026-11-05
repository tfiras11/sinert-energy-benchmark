"""Phase 1 - Step 2: cleaning, gap handling and imputation.

Strategy (per the cahier de charges: "imputation adaptée ... selon la durée des
données manquantes"):

* Timestamps are re-indexed on a strict 15-minute grid; slots absent from the
  raw files are flagged (``flag_slot_missing``).
* Cumulative counters (kWh / kvarh [ABS]) are turned into per-step energy by
  first difference. When a counter is missing over a span <= 2 days, the energy
  of the whole span is spread evenly over the missing steps (totals preserved).
  Counter resets (negative jumps) are detected and excluded.
* Direct measurements (P, Q, Pmax, Smax) are imputed by gap length:
    - <= 1 h    : linear (time) interpolation
    - <= 2 days : seasonal-naive (same time previous week, fallback previous day)
    - >  2 days : left as NaN and counted in the report
* A derived power factor (P / sqrt(P² + Q²)) is added for the main meter,
  since every raw PF column of EM_VIRTUAL is empty (``---``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    ANALYSIS_START,
    BLOCKS,
    COL,
    MAIN_BLOCK,
    MEDIUM_GAP_STEPS,
    SHORT_GAP_STEPS,
    STEPS_PER_DAY,
    STEPS_PER_WEEK,
    STEP_MIN,
    COUNTER_VARS,
    MEASUREMENT_VARS,
)

logger = logging.getLogger(__name__)

RESET_TOL = 1e-3  # negative counter jump below this magnitude is float noise

# Counters become per-step energy -> renamed to reflect the new semantics
COUNTER_OUT_NAME = {"kWh_abs": "E_kWh", "kvarh_abs": "Q_kvarh"}


# --------------------------------------------------------------------------- #
# Regularisation on the 15-minute grid
# --------------------------------------------------------------------------- #
def regularize(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deduplicate, sort, trim to ANALYSIS_START and reindex on a 15-min grid.

    Returns the regularised frame with a boolean ``flag_slot_missing`` column
    and the number of slots that were absent from the raw files.
    """
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if ANALYSIS_START:  # e.g. June 2022 is almost entirely NaN -> dropped
        n_before = len(df)
        df = df[df.index >= pd.Timestamp(ANALYSIS_START)]
        logger.info("ANALYSIS_START=%s: dropped %d rows before it",
                    ANALYSIS_START, n_before - len(df))

    full_idx = pd.date_range(df.index.min(), df.index.max(),
                             freq=f"{STEP_MIN}min")
    missing_ts = full_idx.difference(df.index)
    out = df.reindex(full_idx)
    out.index.name = "timestamp"

    flag = pd.Series(False, index=out.index, name="flag_slot_missing")
    flag.loc[missing_ts] = True
    out["flag_slot_missing"] = flag

    logger.info("Grid: %s -> %s | %d slots | %d missing slots",
                out.index.min(), out.index.max(), len(out), len(missing_ts))
    return out, len(missing_ts)


# --------------------------------------------------------------------------- #
# Counter handling (kWh / kvarh [ABS] -> energy per step)
# --------------------------------------------------------------------------- #
def counter_to_energy(counter: pd.Series, max_missing_steps: int = MEDIUM_GAP_STEPS
                      ) -> tuple[pd.Series, dict]:
    """Convert a cumulative counter into per-step energy (kWh / kvarh).

    * consecutive valid readings            -> simple difference
    * missing span <= ``max_missing_steps`` -> total energy of the span spread
      evenly over its steps (consumption totals are preserved)
    * negative jumps (counter reset/swap)   -> excluded, counted in ``info``
    * spans across longer gaps              -> left as NaN, counted in ``info``
    """
    v = counter.to_numpy(dtype="float64")
    n = len(v)
    energy = np.full(n, np.nan)

    info = {"resets": 0, "long_gap_steps": 0}

    # --- simple consecutive differences -----------------------------------
    diff = np.diff(v)                       # NaN wherever a neighbour is NaN
    energy[1:] = diff
    neg = diff < -RESET_TOL                 # counter reset / replacement
    energy[1:][neg] = np.nan
    info["resets"] = int(neg.sum())

    # --- spans across missing readings -------------------------------------
    valid_pos = np.flatnonzero(~np.isnan(v))
    for a, b in zip(valid_pos[:-1], valid_pos[1:]):
        gap = b - a
        if gap <= 1:
            continue
        total = v[b] - v[a]
        if gap <= max_missing_steps + 1 and total >= -RESET_TOL:
            energy[a + 1: b + 1] = total / gap
        else:
            info["long_gap_steps"] += gap

    out = pd.Series(energy, index=counter.index, name=counter.name)
    return out, info


# --------------------------------------------------------------------------- #
# Measurement imputation (gap-length rules)
# --------------------------------------------------------------------------- #
def _nan_run_lengths(is_nan: pd.Series) -> pd.Series:
    """Length of the NaN run each element belongs to (0 where not NaN)."""
    boundary = is_nan.ne(is_nan.shift(fill_value=False)).cumsum()
    run_len = is_nan.groupby(boundary).transform("size")
    return run_len.where(is_nan, other=0)


def impute_measurement(s: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Impute a direct measurement series according to its gap duration.

    Returns ``(filled, imputed_mask, counts)``.
    """
    raw_nan = s.isna()
    counts = {"linear": 0, "seasonal": 0, "long_gap": int(raw_nan.sum())}
    if not raw_nan.any():
        return s.copy(), pd.Series(False, index=s.index, name=f"{s.name}_imputed"), counts

    run_len = _nan_run_lengths(raw_nan)
    filled = s.copy()

    # --- short gaps (<= 1 h): linear interpolation -------------------------
    lin = s.interpolate(method="linear", limit=SHORT_GAP_STEPS)
    mask_short = raw_nan & (run_len <= SHORT_GAP_STEPS)
    filled[mask_short] = lin[mask_short]
    counts["linear"] = int(mask_short.sum())

    # --- medium gaps (<= 2 days): seasonal naive ---------------------------
    week = s.shift(STEPS_PER_WEEK)   # same weekday, same time, one week earlier
    day = s.shift(STEPS_PER_DAY)     # same time, one day earlier
    seasonal = week.where(week.notna(), day)
    mask_med = raw_nan & (run_len > SHORT_GAP_STEPS) & (run_len <= MEDIUM_GAP_STEPS)
    filled[mask_med] = seasonal[mask_med]
    counts["seasonal"] = int((mask_med & seasonal.notna()).sum())

    # --- long gaps (> 2 days): left as NaN ---------------------------------
    still_nan = filled.isna() & raw_nan
    counts["long_gap"] = int(still_nan.sum())

    mask = filled.notna() & raw_nan
    return filled, mask.rename(f"{s.name}_imputed"), counts


# --------------------------------------------------------------------------- #
# Derived power factor (raw PF columns of EM_VIRTUAL are empty)
# --------------------------------------------------------------------------- #
def derive_power_factor(p: pd.Series, q: pd.Series) -> pd.Series:
    """PF = P / sqrt(P² + Q²); NaN where inputs are missing or both are zero."""
    denom = np.sqrt(p * p + q * q)
    pf = p / denom.where(denom > 0)
    return pf.clip(-1.0, 1.0).rename("EM_VIRTUAL__PF_derived")


# --------------------------------------------------------------------------- #
# Long-gap imputation by seasonal pattern (e.g. the August 2025 outage)
# --------------------------------------------------------------------------- #
def _contiguous_nan_runs(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(start, end, length) of every contiguous NaN run of *s*."""
    mask = s.isna()
    if not mask.any():
        return []
    idx = mask.index.to_series()
    grp = (mask != mask.shift(fill_value=False)).cumsum()[mask]
    runs = pd.DataFrame({"ts": idx[mask]}).groupby(grp).agg(
        start=("ts", "min"), end=("ts", "max"), n=("ts", "size"))
    return [(r.start, r.end, int(r.n)) for r in runs.itertuples()]


def _dow_tod_shape(s: pd.Series) -> pd.Series:
    """Mean of *s* indexed by (dayofweek, HH:MM) - the weekly shape."""
    ref = s.dropna()
    key = pd.MultiIndex.from_arrays(
        [ref.index.dayofweek, ref.index.strftime("%H:%M")])
    return ref.groupby(key).mean()


def _month_slice(df: pd.DataFrame, col: str, year: int, month: int) -> pd.Series:
    """All steps of one calendar month (empty Series if the month is absent)."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(minutes=STEP_MIN)
    return df.loc[start:end, col]


def _lookup(shape: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    key = pd.MultiIndex.from_arrays([idx.dayofweek, idx.strftime("%H:%M")])
    return shape.reindex(key).to_numpy(dtype="float64")


def impute_long_gaps_by_pattern(cleaned: pd.DataFrame, raw: pd.DataFrame,
                                block: str = MAIN_BLOCK
                                ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fill NaN runs longer than ``MEDIUM_GAP_STEPS`` with a seasonal method.

    Method (documented for the weekly review):
    * reference shapes : mean value by (weekday, time-of-day) over the same
      calendar month of every other year whose month is >= 80 % valid;
    * per-step energies (E, Q) are scaled to conserve EXACTLY the true energy
      of the gap, read from the cumulative counter span (last valid reading
      before the run -> first valid reading after it) - the level is a
      measurement, never a guess;
    * P_max / S_max have no counter: their shapes are scaled by the ratio of
      the mean level of the neighbouring months (gap month +/- 1) between the
      gap year and the reference years (documented approximation).

    Returns ``(cleaned_imputed, raw_imputed, infos)``; adds a
    ``{block}__is_imputed`` boolean column and fills the counter NaNs of the
    gap in the raw frame so the feature table sees continuous counters.
    """
    e_col = COL.format(block=block, var="E_kWh")
    q_col = COL.format(block=block, var="Q_kvarh")
    p_col = COL.format(block=block, var="P_max")
    s_col = COL.format(block=block, var="S_max")
    ce_col = "EM_VIRTUAL__kWh_abs"
    cq_col = "EM_VIRTUAL__kvarh_abs"

    cleaned = cleaned.copy()
    raw = raw.copy()
    is_imputed = pd.Series(False, index=cleaned.index)
    infos: list[dict] = []

    # the raw frame has no rows where files are missing (e.g. Aug 2025);
    # add empty grid rows so the gap counters can be filled below
    missing_idx = cleaned.index.difference(raw.index)
    if len(missing_idx):
        empty = pd.DataFrame(np.nan, index=missing_idx, columns=raw.columns)
        raw = pd.concat([raw, empty]).sort_index()
        raw.index.name = "timestamp"
        logger.info("Added %d empty grid rows to the raw frame (gap fill)", len(missing_idx))

    for t_start, t_end, n_steps in _contiguous_nan_runs(cleaned[e_col]):
        if n_steps <= MEDIUM_GAP_STEPS:
            continue
        gap_idx = cleaned.loc[t_start:t_end].index
        month = t_start.month

        # ---- reference years: same calendar month, >= 80 % valid -----------
        ref_years = []
        for y in sorted(set(cleaned.index.year)):
            if y == t_start.year:
                continue
            m = _month_slice(cleaned, e_col, y, month)
            if m.notna().mean() >= 0.8:
                ref_years.append(y)
        if not ref_years:
            logger.warning("no reference year for gap %s..%s - skipped",
                           t_start, t_end)
            continue

        def ref_series(col: str) -> pd.Series:
            return pd.concat([_month_slice(cleaned, col, y, month)
                              for y in ref_years])

        shapes = {v: _dow_tod_shape(ref_series(c)) for v, c in
                  (("E", e_col), ("Q", q_col), ("P", p_col), ("S", s_col))}

        # ---- true energies from the counter span ---------------------------
        rc = raw[ce_col].dropna()
        before = rc[rc.index < t_start].iloc[-1]
        after = rc[rc.index >= t_end].iloc[0]
        span_e = float(after - before)
        rq = raw[cq_col].dropna()
        span_q = float(rq[rq.index >= t_end].iloc[0]
                       - rq[rq.index < t_start].iloc[-1])

        # ---- energies: distribute the known totals -------------------------
        e_filled = pd.Series(_lookup(shapes["E"], gap_idx), index=gap_idx)
        e_filled *= span_e / e_filled.sum()
        q_filled = pd.Series(_lookup(shapes["Q"], gap_idx), index=gap_idx)
        q_filled *= span_q / q_filled.sum()

        # ---- demand peaks: neighbour-month level ratio ---------------------
        def neighbour_level(col: str, year: int) -> float:
            parts = []
            for m in (month - 1, month + 1):
                yy = year if 1 <= m <= 12 else (year + 1 if m == 13 else year - 1)
                mm = 1 if m == 13 else (12 if m == 0 else m)
                parts.append(_month_slice(cleaned, col, yy, mm))
            s = pd.concat(parts).dropna()
            return float(s.mean()) if len(s) else float("nan")

        ratio_p = (neighbour_level(p_col, t_start.year)
                   / np.nanmean([neighbour_level(p_col, y) for y in ref_years]))
        ratio_s = (neighbour_level(s_col, t_start.year)
                   / np.nanmean([neighbour_level(s_col, y) for y in ref_years]))

        cleaned.loc[gap_idx, e_col] = e_filled
        cleaned.loc[gap_idx, q_col] = q_filled
        cleaned.loc[gap_idx, p_col] = _lookup(shapes["P"], gap_idx) * ratio_p
        cleaned.loc[gap_idx, s_col] = _lookup(shapes["S"], gap_idx) * ratio_s
        # mean powers derived from the imputed energies (P = E / step duration),
        # keeping the measurement identities consistent for the whole gap
        cleaned.loc[gap_idx, COL.format(block=block, var="P_avg")] = \
            e_filled * (60.0 / STEP_MIN)
        cleaned.loc[gap_idx, COL.format(block=block, var="Q_avg")] = \
            q_filled * (60.0 / STEP_MIN)
        is_imputed.loc[gap_idx] = True

        # ---- rebuild the counters inside the gap ---------------------------
        for c_col, filled in ((ce_col, e_filled), (cq_col, q_filled)):
            c = raw[c_col]
            fill_idx = c.loc[t_start:t_end].index[c.loc[t_start:t_end].isna()]
            base = c[c.index < t_start].dropna().iloc[-1]
            raw.loc[fill_idx, c_col] = (base + filled.cumsum()
                                        .reindex(fill_idx)).to_numpy()

        infos.append({
            "gap_start": str(t_start), "gap_end": str(t_end), "n_steps": n_steps,
            "reference_years": ref_years,
            "true_energy_kwh": round(span_e, 1),
            "true_reactive_kvarh": round(span_q, 1),
            "ratio_P": round(float(ratio_p), 3), "ratio_S": round(float(ratio_s), 3),
        })
        logger.info("Long gap %s -> %s (%d steps): E=%.0f kWh, Q=%.0f kvarh, "
                    "refs=%s, ratio_P=%.2f, ratio_S=%.2f",
                    t_start.date(), t_end.date(), n_steps, span_e, span_q,
                    ref_years, ratio_p, ratio_s)

    cleaned[f"{block}__is_imputed"] = is_imputed
    return cleaned, raw, infos


# --------------------------------------------------------------------------- #
# Full cleaning pipeline
# --------------------------------------------------------------------------- #
def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the whole step-2 cleaning pipeline.

    Returns
    -------
    df_clean : DataFrame
        15-min grid with, per block: per-step energy (kWh_abs/kvarh_abs
        converted), imputed measurements, plus quality flags for the main
        meter and a derived PF for EM_VIRTUAL.
    report : DataFrame
        One row per block/variable with the cleaning statistics.
    """
    grid, n_missing_slots = regularize(df_raw)

    flag_cols: dict[str, pd.Series] = {}
    report_rows: list[dict] = []

    for block in BLOCKS:
        for var in COUNTER_VARS + MEASUREMENT_VARS:
            col = COL.format(block=block, var=var)
            if col not in grid.columns:      # defensive: schema drift
                logger.warning("column %s absent from grid - skipped", col)
                continue
            s = grid[col]
            n_raw_nan = int(s.isna().sum())

            if var in COUNTER_VARS:
                energy, info = counter_to_energy(s)
                out_col = COL.format(block=block, var=COUNTER_OUT_NAME[var])
                grid = grid.drop(columns=[col])
                grid[out_col] = energy      # per-step energy replaces the counter
                report_rows.append({
                    "block": block, "variable": COUNTER_OUT_NAME[var],
                    "n_raw_nan": n_raw_nan,
                    "n_imputed": 0, "n_long_gap_nan": info["long_gap_steps"],
                    "n_counter_resets": info["resets"],
                })
                if info["resets"]:
                    logger.warning("%s: %d counter reset(s) excluded", col, info["resets"])
            else:
                s = grid[col]
                # defective-value rule (cahier de charges 1.2): a demand peak of
                # exactly ~0 while the meter reports normal mean power is a
                # reporting glitch (58 cases, e.g. P_avg = 161 kW & P_max = 0).
                # Genuine shutdowns are preserved: P_avg must exceed 1 kW.
                n_defective = 0
                if var in ("P_max", "S_max"):
                    p_avg_col = COL.format(block=block, var="P_avg")
                    if p_avg_col in grid.columns:
                        defective = (s < 0.5) & (grid[p_avg_col] > 1.0)
                        n_defective = int(defective.sum())
                        if n_defective:
                            logger.warning("%s: %d defective zero reading(s) "
                                           "-> NaN + imputation", col, n_defective)
                            s = s.mask(defective)
                            grid[col] = s
                filled, mask, counts = impute_measurement(s)
                grid[col] = filled
                if block == MAIN_BLOCK:
                    flag_cols[f"{col}_imputed"] = mask.astype(bool)
                report_rows.append({
                    "block": block, "variable": var, "n_raw_nan": n_raw_nan,
                    "n_imputed": counts["linear"] + counts["seasonal"],
                    "n_long_gap_nan": counts["long_gap"],
                    "n_counter_resets": 0,
                    "n_defective_zero": n_defective,
                })

    # derived PF for the main meter
    p = grid[COL.format(block=MAIN_BLOCK, var="P_avg")]
    q = grid[COL.format(block=MAIN_BLOCK, var="Q_avg")]
    grid["EM_VIRTUAL__PF_derived"] = derive_power_factor(p, q)

    # quality flags (main meter + grid) appended
    for name, col_flag in flag_cols.items():
        grid[name] = col_flag

    report = pd.DataFrame(report_rows)
    logger.info("Cleaning done: %d rows x %d columns", len(grid), grid.shape[1])
    return grid, report
