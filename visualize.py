"""Phase 1 - Step 3: data visualisation (PNG figures for the report/dashboard).

All figures describe the main meter (EM_VIRTUAL) unless stated otherwise.
Missing periods (e.g. August 2025) appear as gaps, never as fabricated data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavior import academic_flags, tariff_season
from config import BLOCKS, COL, MAIN_BLOCK

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
})


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("figure written: %s", path.name)


# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, daily: pd.DataFrame, out_dir: Path) -> None:
    """Render the Phase 1.3 figure set into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    e = df[COL.format(block=MAIN_BLOCK, var="E_kWh")]
    p = df[COL.format(block=MAIN_BLOCK, var="P_avg")]
    pf = df["EM_VIRTUAL__PF_derived"]

    # 1 - daily energy over the whole period ---------------------------------
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.fill_between(daily.index, daily["E_kWh"] / 1e3, color="#2b6cb0", lw=0.4)
    ax.set_title("SUP'COM main meter (EM_VIRTUAL) — daily consumption")
    ax.set_ylabel("kWh ×1000")
    _save(fig, out_dir, "01_daily_energy.png")

    # 2 - monthly energy bars -------------------------------------------------
    monthly = e.resample("MS").sum(min_count=1)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(monthly.index, monthly.values / 1e3, width=24, color="#2b6cb0")
    ax.set_title("Monthly energy — EM_VIRTUAL")
    ax.set_ylabel("MWh")
    _save(fig, out_dir, "02_monthly_energy.png")

    # 3 - mean daily profile per day of week ---------------------------------
    grp = pd.DataFrame({"tod": p.index.strftime("%H:%M"),
                        "dow": p.index.dayofweek, "P": p.to_numpy()}).dropna()
    mean = grp.pivot_table(index="tod", columns="dow", values="P", aggfunc="mean").sort_index()
    mean.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in mean.columns:
        ax.plot(x, mean[c], lw=1.2, label=c)
    ax.set_xticks(x[::16]); ax.set_xticklabels(mean.index[::16], rotation=45)
    ax.set_title("Mean 15-min power profile by day of week (all years)")
    ax.set_ylabel("kW"); ax.legend(ncol=7, fontsize=8)
    _save(fig, out_dir, "03_profile_weekday.png")

    # 4 - seasonal profiles (STEG tariff seasons) ----------------------------
    seas = pd.DataFrame({"tod": p.index.strftime("%H:%M"),
                         "s": tariff_season(p.index).to_numpy(),
                         "P": p.to_numpy()}).dropna()
    m = seas.pivot_table(index="tod", columns="s", values="P", aggfunc="mean").sort_index()
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in m.columns:
        ax.plot(x, m[c], lw=1.4, label=f"{c} (tariff)")
    ax.set_xticks(x[::16]); ax.set_xticklabels(m.index[::16], rotation=45)
    ax.set_title("Mean power profile — STEG tariff seasons")
    ax.set_ylabel("kW"); ax.legend()
    _save(fig, out_dir, "04_profile_season.png")

    # 5 - academic-period profiles -------------------------------------------
    acad = academic_flags(p.index)
    label = np.where(acad["is_summer_break"], "summer_break",
            np.where(acad["is_winter_break"], "winter_break", "study_week"))
    grp = pd.DataFrame({"tod": p.index.strftime("%H:%M"),
                        "a": label, "P": p.to_numpy()}).dropna()
    m = grp.pivot_table(index="tod", columns="a", values="P", aggfunc="mean").sort_index()
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in m.columns:
        ax.plot(x, m[c], lw=1.4, label=c)
    ax.set_xticks(x[::16]); ax.set_xticklabels(m.index[::16], rotation=45)
    ax.set_title("Mean power profile — academic periods (assumed calendar)")
    ax.set_ylabel("kW"); ax.legend()
    _save(fig, out_dir, "05_profile_academic.png")

    # 6 - week heatmap (day x time) for the last full year -------------------
    last_year = p.loc["2025-02-01":"2026-01-31"]
    hm = pd.DataFrame({"d": last_year.index.date,
                       "tod": last_year.index.strftime("%H:%M"),
                       "P": last_year.to_numpy()}).dropna()
    piv = hm.pivot_table(index="d", columns="tod", values="P").sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(piv.to_numpy().T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, len(piv), 0, 96])
    ax.set_yticks([0, 24, 48, 72, 96])
    ax.set_yticklabels(["00h", "06h", "12h", "18h", "24h"])
    ticks = np.linspace(0, len(piv) - 1, 9).astype(int)
    ax.set_xticks(ticks); ax.set_xticklabels([str(piv.index[t]) for t in ticks],
                                             rotation=30, ha="right")
    fig.colorbar(im, ax=ax, label="kW")
    ax.set_title("Load heatmap — Feb 2025 → Jan 2026 (days × time of day)")
    ax.grid(False)
    _save(fig, out_dir, "06_heatmap_year.png")

    # 7 - base load & peak evolution ------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(daily.index, daily["P_max_kW"], lw=0.7, label="daily peak (kW)")
    ax.plot(daily.index, daily["P_min_kW"], lw=0.7, color="#c53030", label="daily base (kW)")
    ax.legend(); ax.set_ylabel("kW")
    ax.set_title("Daily peak vs base load — EM_VIRTUAL")
    _save(fig, out_dir, "07_base_peak_evolution.png")

    # 8 - derived power factor ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 3.2))
    pf_d = pf.resample("D").agg(["mean", "count"])
    pf_m = pf_d["mean"].where(pf_d["count"] >= 24)   # days with >= 6 h of data
    ax.plot(pf_m.index, pf_m, lw=0.7, color="#276749")
    ax.axhline(0.90, color="#c05621", ls="--", lw=1, label="cos φ = 0.90 (contract reference)")
    ax.axhline(0.80, color="#c53030", ls=":", lw=1, label="cos φ = 0.80 (penalty threshold)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Derived power factor (daily mean) — EM_VIRTUAL")
    _save(fig, out_dir, "08_power_factor.png")

    # 9 - annual daily-energy boxplots ----------------------------------------
    grp = pd.DataFrame({"y": daily.index.year, "E": daily["E_kWh"]}).dropna()
    data = [g["E"].to_numpy() / 1e3 for _, g in grp.groupby("y")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=sorted(grp["y"].unique()))
    ax.set_ylabel("kWh ×1000")
    ax.set_title("Distribution of daily energy per year")
    _save(fig, out_dir, "09_daily_energy_boxplot.png")

    # 10 - per-block annual energy (all meters) -------------------------------
    cols = {b: COL.format(block=b, var="E_kWh") for b in BLOCKS}
    cols = {b: c for b, c in cols.items() if c in df.columns}
    ann = df[list(cols.values())].rename(columns={c: b for b, c in cols.items()}) \
        .resample("YS").sum(min_count=1) / 1e3
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(ann))
    cmap = plt.get_cmap("tab20")
    for i, b in enumerate(ann.columns):
        vals = ann[b].fillna(0).to_numpy()
        ax.bar(ann.index.year, vals, bottom=bottom, label=b, color=cmap(i % 20))
        bottom += vals
    ax.set_ylabel("MWh"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Annual energy by meter block (stacked)")
    _save(fig, out_dir, "10_annual_by_block.png")
