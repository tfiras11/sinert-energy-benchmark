"""Phase 2 - shared data protocol for the benchmark.

Task (identical for every model, DL or not):
    input  X : past week  = 672 steps x 11 channels (windows_daily.npz)
    target Y : next day   = 96 steps  x 4 targets  (dEa, dEr, Pmax, Smax)

Chronological split (no shuffling - time series):
    train : sample_time <= TRAIN_END      (default 2025-05-31)
    val   : TRAIN_END <  t <= VAL_END     (default 2025-10-31)
    test  : t > VAL_END                   (default up to 2026-01-31)
The test period contains no imputed days, so final metrics are quoted on real
measurements only. The 39 imputed-flagged samples fall in val (August 2025).

``prepare_data`` writes ``windows_scaled.npz`` - the single file needed on
Colab to train the deep-learning models with the exact same protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_END = "2025-05-31"
VAL_END = "2025-10-31"

TARGETS = ["dEa_kWh", "dEr_kvarh", "P_max_kW", "S_max_kVA"]


# --------------------------------------------------------------------------- #
def load_windows(out_dir: Path) -> dict:
    """Load the Phase 1 window artefacts."""
    d = np.load(Path(out_dir) / "structured" / "windows_daily.npz", allow_pickle=False)
    return {
        "X": d["X"], "Y": d["Y"],
        "sample_time": pd.DatetimeIndex(d["sample_time"]),
        "sample_imputed": d["sample_imputed"].astype(bool),
        "features": [str(x) for x in d["features"]],
        "targets": [str(x) for x in d["targets"]],
    }


def make_splits(sample_time: pd.DatetimeIndex,
                train_end: str = TRAIN_END, val_end: str = VAL_END
                ) -> dict[str, np.ndarray]:
    """Boolean masks train / val / test on the sample axis (chronological)."""
    te = pd.Timestamp(train_end) + pd.Timedelta(hours=23, minutes=59)
    ve = pd.Timestamp(val_end) + pd.Timedelta(hours=23, minutes=59)
    t = pd.DatetimeIndex(sample_time)
    return {
        "train": np.asarray(t <= te),
        "val": np.asarray((t > te) & (t <= ve)),
        "test": np.asarray(t > ve),
    }


# --------------------------------------------------------------------------- #
def prepare_data(out_dir: Path, train_end: str = TRAIN_END,
                 val_end: str = VAL_END) -> dict:
    """Load windows, split chronologically, fit scalers on train, save npz.

    Returns a dict with everything the benchmark needs:
    ``Xs / Ys`` (scaled float32), masks, sample metadata and scaler params.
    """
    w = load_windows(out_dir)
    X, Y = w["X"].astype("float32"), w["Y"].astype("float32")
    masks = make_splits(w["sample_time"], train_end, val_end)

    x_mean = X[masks["train"]].reshape(-1, X.shape[2]).mean(0)
    x_std = X[masks["train"]].reshape(-1, X.shape[2]).std(0) + 1e-8
    y_mean = Y[masks["train"]].reshape(-1, Y.shape[2]).mean(0)
    y_std = Y[masks["train"]].reshape(-1, Y.shape[2]).std(0) + 1e-8

    Xs = (X - x_mean) / x_std
    Ys = (Y - y_mean) / y_std

    split = np.full(len(X), "", dtype="<U8")
    for name, m in masks.items():
        split[m] = name

    bench_dir = Path(out_dir) / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bench_dir / "windows_scaled.npz",
        X=Xs, Y=Ys,
        sample_time=w["sample_time"].values,
        sample_imputed=w["sample_imputed"],
        split=split,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        features=np.array(w["features"]), targets=np.array(w["targets"]),
        train_end=train_end, val_end=val_end,
    )
    logger.info("windows_scaled.npz written (Colab-ready): %d samples "
                "(train %d / val %d / test %d)", len(Xs),
                masks["train"].sum(), masks["val"].sum(), masks["test"].sum())

    return {"X": X, "Y": Y, "Xs": Xs, "Ys": Ys, "masks": masks,
            "sample_time": w["sample_time"], "sample_imputed": w["sample_imputed"],
            "features": w["features"], "targets": w["targets"],
            "y_mean": y_mean, "y_std": y_std}


def flatten(X: np.ndarray) -> np.ndarray:
    """(n, 672, 11) -> (n, 7392) for the tabular models."""
    return X.reshape(X.shape[0], -1)
