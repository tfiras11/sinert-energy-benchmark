"""Phase 1 - Step 4: consumption & behaviour parameter extraction.

All definitions below are documented so they can be defended at the weekly
review, and every threshold lives in ``config.py``.

* **Cumulative energy (kWh)**    : daily / weekly (ISO) / monthly / annual sums
  of the per-step energy ``E_kWh`` for every meter block.
* **Base load (creux)**          : per day, minimum of ``P_avg`` and the hour
  it occurs.
* **Peak demand (pointe)**       : per day, max of ``P_max`` (kW) and ``S_max``
  (kVA), hour of occurrence, and peak duration = number of 15-min steps with
  ``P_avg >= PEAK_DURATION_FRAC * daily Pmax``.
* **Average consumption**        : mean power profiles by day-of-week class
  (weekday / Saturday / Sunday), by tariff season (STEG contract) and by
  academic period (study week vs university holiday - assumption in config).
* **Site activation**            : per day, first and last step where
  ``P_avg >= daily_min + ACTIVATION_MARGIN * (daily_max - daily_min)``.
* **Variability**                : per day std, variance and coefficient of
  variation of ``P_avg``; monthly aggregates as well.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    ACTIVATION_MARGIN,
    BLOCKS,
    COL,
    MAIN_BLOCK,
    PEAK_DURATION_FRAC,
    SUMMER_BREAK,
    WINTER_BREAK,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _month_day_range(ts: pd.Timestamp, span: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """True if *ts* falls inside a (start_md, end_md) month/day span (wrap-aware)."""
    (m1, d1), (m2, d2) = span
    md = (ts.month, ts.day)
    if (m1, d1) <= (m2, d2):
        return (m1, d1) <= md <= (m2, d2)
    return md >= (m1, d1) or md <= (m2, d2)   # spans wrapping New Year


def academic_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-timestep flags: is_study_week, is_summer_break, is_winter_break."""
    ts = pd.Series(index, index=index)
    summer = ts.apply(lambda t: _month_day_range(t, SUMMER_BREAK))
    winter = ts.apply(lambda t: _month_day_range(t, WINTER_BREAK))
    return pd.DataFrame({
        "is_summer_break": summer.to_numpy(),
        "is_winter_break": winter.to_numpy(),
        "is_study_week": ~(summer | winter).to_numpy(),
    }, index=index)


def tariff_season(index: pd.DatetimeIndex) -> pd.Series:
    """STEG MT tariff season label per timestamp ('hiver' / 'été')."""
    months = pd.Series(index.month, index=index)
    return months.isin((9, 10, 11, 12, 1, 2, 3, 4, 5)).map(
        {True: "hiver", False: "été"}).rename("tariff_season")


# --------------------------------------------------------------------------- #
# 1) Cumulative energy totals per block
# --------------------------------------------------------------------------- #
def energy_totals(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily / weekly / monthly / annual energy sums for every meter block."""
    cols = {b: COL.format(block=b, var="E_kWh") for b in BLOCKS}
    cols = {b: c for b, c in cols.items() if c in df.columns}
    e = df[list(cols.values())].rename(columns={c: b for b, c in cols.items()})

    out = {
        "daily": e.resample("D").sum(min_count=1),
        "weekly": e.resample("W-MON", label="left", closed="left").sum(min_count=1),
        "monthly": e.resample("MS").sum(min_count=1),
        "annual": e.resample("YS").sum(min_count=1),
    }
    return out


# --------------------------------------------------------------------------- #
# 2) Daily behaviour parameters (main meter)
# --------------------------------------------------------------------------- #
def _hour_of_extreme(s: pd.Series, which: str) -> pd.Series:
    """Per calendar day, the HH:MM of the first extreme (min/max) of *s*.

    Days entirely NaN (outages) yield None instead of raising.
    """
    def _f(g: pd.Series):
        if g.notna().sum() == 0:
            return None
        idx = g.idxmin() if which == "min" else g.idxmax()
        return idx.strftime("%H:%M")
    return s.groupby(s.index.date).apply(_f)


def _first_last_true(mask: pd.Series, which: str) -> pd.Series:
    """Per calendar day, HH:MM of the first/last True of a boolean mask."""
    def _f(g: pd.Series):
        if not g.any():
            return None
        idx = g.index[0] if which == "first" else g.index[-1]
        return idx.strftime("%H:%M")
    sub = mask[mask]                      # keep only True steps
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.groupby(sub.index.date).apply(_f)


def daily_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """One row per day with peaks, base load, activation hours, variability."""
    p = df[COL.format(block=MAIN_BLOCK, var="P_avg")]
    pmax = df[COL.format(block=MAIN_BLOCK, var="P_max")]
    smax = df[COL.format(block=MAIN_BLOCK, var="S_max")]
    e = df[COL.format(block=MAIN_BLOCK, var="E_kWh")]

    daily = pd.DataFrame(index=pd.to_datetime(sorted(set(p.index.date))))

    daily["E_kWh"] = e.resample("D").sum(min_count=1)
    daily["P_min_kW"] = p.resample("D").min(min_count=1)
    daily["P_min_hour"] = _hour_of_extreme(p, "min")
    daily["P_max_kW"] = pmax.resample("D").max(min_count=1)
    daily["P_max_hour"] = _hour_of_extreme(pmax, "max")
    daily["S_max_kVA"] = smax.resample("D").max(min_count=1)
    daily["S_max_hour"] = _hour_of_extreme(smax, "max")

    # peak duration: steps with P_avg >= frac * daily max(P_avg)
    pmax_avg = p.resample("D").transform("max")
    above = p >= PEAK_DURATION_FRAC * pmax_avg
    daily["peak_duration_steps"] = above.groupby(p.index.date).sum().astype("float64")

    # activation / deactivation hours
    pmin = p.resample("D").transform("min")
    active = p >= pmin + ACTIVATION_MARGIN * (pmax_avg - pmin)
    daily["activation_hour"] = _first_last_true(active, "first")
    daily["deactivation_hour"] = _first_last_true(active, "last")
    daily["active_steps"] = active.groupby(p.index.date).sum().astype("float64")

    # variability
    g = p.groupby(p.index.date)
    daily["P_mean_kW"] = g.mean()
    daily["P_std_kW"] = g.std()
    daily["P_var_kW2"] = g.var()
    daily["P_cv"] = daily["P_std_kW"] / daily["P_mean_kW"]

    # calendar context
    daily["weekday"] = daily.index.day_name()
    daily["is_weekend"] = daily.index.dayofweek >= 5
    cal = academic_flags(daily.index)
    daily = pd.concat([daily, cal, tariff_season(daily.index)], axis=1)

    logger.info("Daily parameters: %d days", len(daily))
    return daily


# --------------------------------------------------------------------------- #
# 3) Monthly aggregates of the behaviour parameters
# --------------------------------------------------------------------------- #
def monthly_parameters(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation of the daily behaviour table."""
    m = daily.resample("MS").agg({
        "E_kWh": "sum",
        "P_max_kW": "max",
        "S_max_kVA": "max",
        "P_min_kW": "min",
        "P_std_kW": "mean",
        "P_cv": "mean",
        "peak_duration_steps": "sum",
        "active_steps": "sum",
        "is_study_week": "mean",
    })
    m["peak_duration_h"] = m["peak_duration_steps"] / 4.0
    m["active_hours"] = m["active_steps"] / 4.0
    return m


# --------------------------------------------------------------------------- #
# 4) Average power profiles (per class)
# --------------------------------------------------------------------------- #
def average_profiles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean 15-min power profiles by day class (main meter).

    Returns
    -------
    dict with keys ``weekday_class`` (rows: time of day, columns:
    Mon..Sun), ``season`` (rows: time of day, columns: hiver/été),
    ``academic`` (rows: time of day, columns: study/summer/winter break).
    Each value also carries a ``_std`` twin frame with the std deviation.
    """
    p = df[COL.format(block=MAIN_BLOCK, var="P_avg")].dropna()
    time_of_day = p.index.strftime("%H:%M")

    prof: dict[str, pd.DataFrame] = {}

    dow = p.index.dayofweek
    grp = pd.DataFrame({"tod": time_of_day, "dow": dow, "P": p.to_numpy()})
    mean = grp.pivot_table(index="tod", columns="dow", values="P", aggfunc="mean")
    std = grp.pivot_table(index="tod", columns="dow", values="P", aggfunc="std")
    mean.columns = std.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    prof["weekday_class_mean"], prof["weekday_class_std"] = mean.sort_index(), std.sort_index()

    seas = tariff_season(p.index).to_numpy()
    grp = pd.DataFrame({"tod": time_of_day, "season": seas, "P": p.to_numpy()})
    prof["season_mean"] = grp.pivot_table(index="tod", columns="season", values="P", aggfunc="mean").sort_index()
    prof["season_std"] = grp.pivot_table(index="tod", columns="season", values="P", aggfunc="std").sort_index()

    acad = academic_flags(p.index)
    label = np.where(acad["is_summer_break"], "summer_break",
            np.where(acad["is_winter_break"], "winter_break", "study_week"))
    grp = pd.DataFrame({"tod": time_of_day, "acad": label, "P": p.to_numpy()})
    prof["academic_mean"] = grp.pivot_table(index="tod", columns="acad", values="P", aggfunc="mean").sort_index()
    prof["academic_std"] = grp.pivot_table(index="tod", columns="acad", values="P", aggfunc="std").sort_index()

    return prof
