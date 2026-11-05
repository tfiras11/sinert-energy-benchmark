"""Phase 1 analysis runner: visualisation (1.3), behaviour parameters (1.4)
and window structuring (1.5).

Usage:
    python -I run_phase1_analysis.py [--out-dir PATH]

Reads ``cleaned_15min.parquet`` and ``raw_ingested.parquet`` produced by
``run_phase1.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from behavior import (  # noqa: E402
    average_profiles,
    daily_parameters,
    energy_totals,
    monthly_parameters,
)
from cleaning import impute_long_gaps_by_pattern  # noqa: E402
from config import COL, MAIN_BLOCK  # noqa: E402
from structure import (  # noqa: E402
    CAL_FEATURES,
    FEATURES,
    HORIZON_STEPS,
    INPUT_STEPS,
    PHYS_FEATURES,
    TARGETS,
    build_feature_table,
    build_windows,
)
from visualize import make_figures  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_phase1_analysis")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 - steps 3, 4 & 5")
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = ap.parse_args()

    out = args.out_dir
    beh_dir = out / "behavior"
    fig_dir = out / "figures"
    str_dir = out / "structured"
    for d in (beh_dir, fig_dir, str_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cleaned = pd.read_parquet(out / "cleaned_15min.parquet")
    raw = pd.read_parquet(out / "raw_ingested.parquet")
    logger.info("Loaded cleaned (%s) and raw tables", cleaned.shape)

    # ---------------- optional long-gap pattern imputation ------------------
    gap_infos = []
    if config.IMPUTE_LONG_GAPS:
        cleaned, raw, gap_infos = impute_long_gaps_by_pattern(cleaned, raw)
        cleaned.to_parquet(out / "cleaned_15min_imputed.parquet")
        logger.info("Imputed tables saved (cleaned_15min_imputed.parquet)")

    # ---------------- step 4: behaviour parameters -------------------------
    totals = energy_totals(cleaned)
    for name, tbl in totals.items():
        tbl.to_csv(beh_dir / f"energy_{name}.csv")

    daily = daily_parameters(cleaned)
    daily.to_csv(beh_dir / "daily_parameters.csv")
    monthly = monthly_parameters(daily)
    monthly.to_csv(beh_dir / "monthly_parameters.csv")

    profiles = average_profiles(cleaned)
    for name, tbl in profiles.items():
        tbl.to_csv(beh_dir / f"profile_{name}.csv")

    # ---------------- step 3: figures --------------------------------------
    make_figures(cleaned, daily, fig_dir)

    # ---------------- step 5: structuring ----------------------------------
    feats = build_feature_table(cleaned, raw)
    feats.to_parquet(str_dir / "features_15min.parquet")
    X, Y, sample_time, n_dropped, sample_imputed = build_windows(feats, cleaned, stride=96)
    np.savez_compressed(
        str_dir / "windows_daily.npz",
        X=X, Y=Y, sample_time=sample_time.values, sample_imputed=sample_imputed,
        features=np.array(FEATURES), phys=np.array(PHYS_FEATURES),
        cal=np.array(CAL_FEATURES), targets=np.array(TARGETS),
        input_steps=INPUT_STEPS, horizon=HORIZON_STEPS, stride=96,
    )
    pd.DataFrame({"sample_time": sample_time}).to_csv(
        str_dir / "windows_index.csv", index=False)

    # ---------------- summary ----------------------------------------------
    pf = cleaned["EM_VIRTUAL__PF_derived"].dropna()
    study = daily[daily["is_study_week"]]
    lines = [
        "=" * 72,
        "PHASE 1 - ANALYSIS SUMMARY (steps 3, 4, 5)",
        "=" * 72,
        f"Period           : {cleaned.index.min()} -> {cleaned.index.max()}",
        f"Days analysed    : {len(daily)} ({int(daily['is_study_week'].sum())} study, "
        f"{int(daily['is_summer_break'].sum())} summer break, "
        f"{int(daily['is_winter_break'].sum())} winter break)",
        "",
        "Long-gap imputation (pattern-based, flagged):",
    ]
    if gap_infos:
        for gi in gap_infos:
            lines.append(
                f"  gap {gi['gap_start'][:10]} -> {gi['gap_end'][:10]} "
                f"({gi['n_steps']} steps): E={gi['true_energy_kwh']:,.0f} kWh, "
                f"Q={gi['true_reactive_kvarh']:,.0f} kvarh (counter-conserved), "
                f"refs={gi['reference_years']}, ratio_P={gi['ratio_P']}, "
                f"ratio_S={gi['ratio_S']}")
    else:
        lines.append("  none (IMPUTE_LONG_GAPS=False or no long gap found)")
    lines += [
        "",
        "Annual energy - EM_VIRTUAL (kWh):",
        totals["annual"][MAIN_BLOCK].round(0).to_string(),
        "",
        f"Base load  : median daily min = {daily['P_min_kW'].median():.1f} kW "
        f"(median hour {daily['P_min_hour'].mode().iloc[0]})",
        f"Peak demand: mean daily Pmax = {daily['P_max_kW'].mean():.1f} kW "
        f"(median hour {daily['P_max_hour'].mode().iloc[0]}), "
        f"max = {daily['P_max_kW'].max():.1f} kW on {daily['P_max_kW'].idxmax().date()}",
        f"Apparent   : max Smax = {daily['S_max_kVA'].max():.1f} kVA "
        f"on {daily['S_max_kVA'].idxmax().date()}",
        f"Activation : median {study['activation_hour'].mode().iloc[0]} / "
        f"deactivation {study['deactivation_hour'].mode().iloc[0]} (study weeks)",
        f"Variability: mean daily std = {daily['P_std_kW'].mean():.1f} kW, "
        f"mean CV = {daily['P_cv'].mean():.2f}",
        "",
        "Derived power factor (billing-relevant):",
        f"  mean = {pf.mean():.3f} | share of steps PF < 0.90: "
        f"{(pf < 0.90).mean():.2%} | PF < 0.80: {(pf < 0.80).mean():.2%} | "
        f"PF < 0.60: {(pf < 0.60).mean():.2%}",
        "",
        "Structuring (Phase 1.5):",
        f"  features_15min.parquet : {len(feats):,} rows x {feats.shape[1]} cols "
        f"(11 modelling vars + is_imputed flag)",
        f"  windows_daily.npz      : X{X.shape}, Y{Y.shape} float32 | "
        f"{len(sample_time)} samples kept, {n_dropped} dropped, "
        f"{int(sample_imputed.sum())} flagged imputed (excludable in Phase 2)",
        f"Elapsed: {time.perf_counter() - t0:.1f} s",
    ]
    text = "\n".join(lines)
    (out / "analysis_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
