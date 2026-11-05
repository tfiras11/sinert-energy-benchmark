"""Phase 2 runner - non-deep-learning benchmark (baselines, SARIMAX, ML).

Usage:
    python -I run_phase2.py [--out-dir PATH] [--no-sarimax] [--no-ml]

Produces (in output/benchmark/):
    windows_scaled.npz      Colab-ready scaled dataset (DL models)
    benchmark_results.csv   metrics table (DL rows merged later by merge_dl.py)
    predictions/<model>.npy val+test predictions, physical units
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "phase2"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark import Benchmark  # noqa: E402
from data_protocol import flatten, prepare_data  # noqa: E402
from ml_models import MODELS  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_phase2")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 - benchmark (non-DL)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--no-sarimax", action="store_true")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--only", choices=["baselines", "ml", "sarimax"], default=None,
                    help="run a single section on top of the existing results")
    args = ap.parse_args()

    bench = Benchmark(args.out_dir)
    if args.only:
        bench.load_existing()
    t0 = time.perf_counter()

    data = prepare_data(args.out_dir)
    X, Y, masks = data["X"], data["Y"], data["masks"]
    st, y_mean, y_std = data["sample_time"], data["y_mean"], data["y_std"]

    y_true = {m: Y[mask] for m, mask in
              (("val", masks["val"]), ("test", masks["test"]))}
    Xtr, Ytr = flatten(X[masks["train"]]), Y[masks["train"]].reshape(len(Y[masks["train"]]), -1)

    # ------------------------------------------------------------------ #
    run_all = args.only is None
    if run_all or args.only == "baselines":
        logger.info("=== baselines ===")
        from baselines import naive_predictions
        # drop rows of the section being re-run (dedupe)
        bench.rows = [r for r in bench.rows if r.get("family") != "Baseline"]
        for name, preds in naive_predictions(X, st, masks).items():
            bench.register(name, "Baseline", y_true, preds,
                           n_params=0, train_time_s=0.0,
                           note="seasonal persistence (" + name.split("_")[1] + ")")
        bench.save()

    # ------------------------------------------------------------------ #
    if (run_all or args.only == "ml") and not args.no_ml:
        logger.info("=== machine learning family ===")
        bench.rows = [r for r in bench.rows if r.get("family") != "Machine Learning"]
        Ytr_scaled = data["Ys"][masks["train"]].reshape(len(Xtr), -1)
        for name, fn in MODELS.items():
            logger.info("--- %s ---", name)
            pred_scaled, n_params, dt = fn(Xtr, Ytr_scaled,
                                           flatten(X[masks["val"] | masks["test"]]))
            pred_phys = pred_scaled.reshape(-1, 96, 4) * y_std + y_mean
            preds = {"val": pred_phys[:int(masks["val"].sum())].astype("float32"),
                     "test": pred_phys[int(masks["val"].sum()):].astype("float32")}
            bench.register(name, "Machine Learning", y_true, preds,
                           n_params=n_params, train_time_s=dt)
            bench.save()   # incremental: partial results survive interruption

    # ------------------------------------------------------------------ #
    if (run_all or args.only == "sarimax") and not args.no_sarimax:
        logger.info("=== SARIMAX ===")
        from sarimax_model import fit_forecast_sarimax
        cleaned = pd.read_parquet(args.out_dir / "cleaned_15min_imputed.parquet")
        sar = fit_forecast_sarimax(cleaned, st, masks, y_mean, y_std)
        preds = {m: sar[m] for m in ("val", "test")}
        bench.register("SARIMAX", "Statistical", y_true, preds,
                       n_params=sar["n_params"], train_time_s=sar["train_time_s"],
                       note="order (2,0,2) + calendar Fourier exog")
        bench.save()

    # ------------------------------------------------------------------ #
    df = bench.save()
    cols = ["model", "family", "n_params", "train_time_s"] + \
           [c for c in df.columns if c.endswith("_test")]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].round(3).to_string(index=False))
    logger.info("Phase 2 (non-DL) done in %.1fs -> %s",
                time.perf_counter() - t0, bench.out_dir)


if __name__ == "__main__":
    main()
