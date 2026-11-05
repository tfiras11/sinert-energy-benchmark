"""Phase 2/4 - benchmark engine: metrics, result table and prediction store.

Metrics (cahier de charges Phase 4), computed per target on the requested
split, after inverse-scaling back to physical units:

* MAE / RMSE / R2  on the pooled 96-step profiles (n_samples x 96)
* MAPE             on DAILY aggregates (sum of the 96 steps for energies;
                     avoids division by near-zero instantaneous values)
* PDE              Peak Demand Error on the DAILY peak
                     mean |peak_pred - peak_true| / peak_true
* n_params / train_time_s  (frugality criteria, Phase 4)

Every model's predictions (physical units) are stored so Phase 3 can apply
the STEG tariff model to them (delta-facture) and Phase 5 can plot them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGETS = ["dEa_kWh", "dEr_kvarh", "P_max_kW", "S_max_kVA"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """y_*: (n, 96, 4) physical units. Returns the flat metric dict of one model."""
    out: dict[str, float] = {}
    for k, tgt in enumerate(TARGETS):
        t, p = y_true[:, :, k].ravel(), y_pred[:, :, k].ravel()
        err = p - t
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[f"MAE_{tgt}"] = float(np.mean(np.abs(err)))
        out[f"RMSE_{tgt}"] = float(np.sqrt(ss_res / len(t)))
        out[f"R2_{tgt}"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if tgt.startswith("d"):          # daily energy -> MAPE on the daily sum
            d_t, d_p = y_true[:, :, k].sum(1), y_pred[:, :, k].sum(1)
            out[f"MAPE_{tgt}_daily"] = _mape(d_t, d_p)
        else:                             # demand -> MAE on the daily peak
            pk_t, pk_p = y_true[:, :, k].max(1), y_pred[:, :, k].max(1)
            out[f"MAE_{tgt}_dailypeak"] = float(np.mean(np.abs(pk_p - pk_t)))

    # Peak Demand Error (cahier de charges) on the daily peaks of P & S
    for k, tgt in ((2, "P"), (3, "S")):
        pk_t = y_true[:, :, k].max(1)
        pk_p = y_pred[:, :, k].max(1)
        out[f"PDE_{tgt}max"] = float(np.mean(np.abs(pk_p - pk_t) / np.maximum(pk_t, 1e-6)) * 100.0)
    return out


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
class Benchmark:
    """Collects models, evaluates them and writes the ranked result table."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir) / "benchmark"
        self.pred_dir = self.out_dir / "predictions"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def load_existing(self) -> None:
        """Preserve rows of a previous run (models re-run here are dropped)."""
        csv = self.out_dir / "benchmark_results.csv"
        if csv.exists():
            old = pd.read_csv(csv)
            self.rows = old.to_dict("records")
            logger.info("loaded %d existing benchmark row(s)", len(self.rows))

    def register(self, name: str, family: str, y_true: dict[str, np.ndarray],
                 y_pred: dict[str, np.ndarray], n_params: int | None = None,
                 train_time_s: float | None = None, note: str = "") -> dict[str, float]:
        np.savez_compressed(self.pred_dir / f"{name}.npz",
                            val=y_pred["val"], test=y_pred["test"])
        row = {"model": name, "family": family,
               "n_params": n_params if n_params is not None else np.nan,
               "train_time_s": round(train_time_s, 1) if train_time_s else np.nan,
               "note": note}
        metrics: dict[str, float] = {}
        for split in ("val", "test"):
            m = evaluate(y_true[split], y_pred[split])
            metrics.update({f"{k}_{split}": v for k, v in m.items()})
        row.update(metrics)
        self.rows.append(row)
        logger.info("registered %-12s | test RMSE_dEa=%.1f  MAPE_dEa=%.2f%%  PDE_P=%.2f%%",
                    name, metrics["RMSE_dEa_kWh_test"], metrics["MAPE_dEa_kWh_daily_test"],
                    metrics["PDE_Pmax_test"])
        return metrics

    def save(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        df.to_csv(self.out_dir / "benchmark_results.csv", index=False)
        return df
