"""Merge the Colab results (colab_results.csv + colab_predictions.npz) into
the local benchmark table (benchmark_results.csv + predictions/).

Usage (after downloading the two Colab files into output/benchmark/):
    python -I merge_colab.py [--out-dir PATH]

Rows are de-duplicated by model name (local rows are kept if a model exists
in both). Every model's predictions end up in benchmark/predictions/<m>.npz
so Phase 3 (delta-facture) and Phase 5 (dashboard) see one consistent store.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    ap.add_argument("--colab-dir", type=Path, default=None,
                    help="folder holding colab_results.csv + colab_predictions.npz "
                         "(default: <out-dir>/benchmark)")
    args = ap.parse_args()
    bench = args.out_dir / "benchmark"
    cdir = args.colab_dir or bench

    results = pd.read_csv(bench / "benchmark_results.csv")
    colab = pd.read_csv(cdir / "colab_results.csv")

    d = np.load(cdir / "colab_predictions.npz")
    pred_dir = bench / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key in d.files:
        if "__" not in key:
            continue
        name = key.split("__")[0]
        np.savez_compressed(pred_dir / f"{name}.npz",
                            **{k: d[k] for k in d.files if k.startswith(name + "__")})

    new_models = colab["model"].tolist()
    merged = pd.concat([results[~results["model"].isin(new_models)], colab],
                       ignore_index=True)
    merged.to_csv(bench / "benchmark_results.csv", index=False)
    print(f"merged {len(new_models)} Colab rows -> {len(merged)} models total")
    cols = ["model", "family", "train_time_s", "RMSE_dEa_kWh_test",
            "MAPE_dEa_kWh_daily_test", "PDE_Pmax_test"]
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].sort_values("RMSE_dEa_kWh_test").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
