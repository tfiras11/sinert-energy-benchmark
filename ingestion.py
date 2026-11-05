"""Phase 1 - Step 1: ingestion of the raw SInERT / VMU-C "Daily Report" exports.

Every monthly folder of the dataset contains one export file per ~2 days (CSV,
or XLSX for a few months such as November 2023), each holding 193 rows
(96 steps/day + an empty trailing "24:00:00" row) and 1501 columns
(1 EM_VIRTUAL block + 13 sub-meter blocks). We keep only the 6 variables that
matter for the project, for each of the 14 blocks, and concatenate everything
into a single wide time series indexed by timestamp.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BLOCKS,
    COL,
    DEFAULT_DATA_DIR,
    MISSING_TOKEN,
    VARIABLES,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def discover_files(data_dir: Path) -> list[Path]:
    """Return every raw export (CSV or XLSX) under *data_dir*, sorted by name."""
    files = sorted(p for p in data_dir.rglob("*.*")
                   if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"})
    logger.info("Discovered %d export files under %s", len(files), data_dir)
    return files


# --------------------------------------------------------------------------- #
# Raw column mapping
# --------------------------------------------------------------------------- #
def build_column_map() -> dict[str, str]:
    """Map raw (stripped) column name -> pipeline column name ``{block}__{var}``."""
    mapping = {}
    for block in BLOCKS:
        for var, suffix in VARIABLES.items():
            mapping[f"{block} {suffix}"] = COL.format(block=block, var=var)
    return mapping


# --------------------------------------------------------------------------- #
# Single-file parsing
# --------------------------------------------------------------------------- #
def load_raw_file(path: Path, col_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Load one export file, returning a clean frame indexed by timestamp.

    Returns
    -------
    df : DataFrame
        One column per ``{block}__{var}`` found in the file, float values,
        ``NaN`` wherever the raw file contains ``---``.
    info : dict
        Per-file bookkeeping used for the ingestion log.
    """
    info = {"file": path.name,
            "rows_kept": 0, "rows_dropped_2400": 0, "rows_bad_ts": 0,
            "cols_missing": ""}

    # -- read (files are small; full read keeps us robust to schema drift) ---
    if path.suffix.lower() == ".xlsx":
        # Excel exports append a numeric index to every column name
        # (e.g. "EM_VIRTUAL kWh (kWh) [ABS] 1"); Date is already a datetime.
        df = pd.read_excel(path, na_values=[MISSING_TOKEN])
        df.columns = [re.sub(r"\s+\d+$", "", str(c).strip()) for c in df.columns]
    else:
        try:
            df = pd.read_csv(path, sep=";", na_values=[MISSING_TOKEN])
        except UnicodeDecodeError:  # pragma: no cover - defensive
            df = pd.read_csv(path, sep=";", na_values=[MISSING_TOKEN],
                             encoding="latin-1")
        # raw headers carry trailing spaces -> normalise once
        df.columns = [str(c).strip() for c in df.columns]

    # -- timestamp ------------------------------------------------------------
    if "Date" not in df.columns or "Time" not in df.columns:
        info["cols_missing"] = "Date/Time"
        return pd.DataFrame(), info

    n_before = len(df)
    df = df[df["Time"].astype(str).str.strip() != "24:00:00"]
    info["rows_dropped_2400"] = n_before - len(df)

    date_part = df["Date"]
    if pd.api.types.is_datetime64_any_dtype(date_part):
        ts = pd.to_datetime(date_part.dt.strftime("%Y-%m-%d") + " "
                            + df["Time"].astype(str).str.strip(),
                            format="%Y-%m-%d %H:%M:%S", errors="coerce")
    else:
        ts = pd.to_datetime(
            date_part.astype(str).str.strip() + " "
            + df["Time"].astype(str).str.strip(),
            format="%d-%m-%Y %H:%M:%S", errors="coerce",
        )
    bad = ts.isna()
    info["rows_bad_ts"] = int(bad.sum())
    df = df.loc[~bad]
    ts = ts.loc[~bad]

    # -- variable selection ----------------------------------------------------
    available = [raw for raw in col_map if raw in df.columns]
    missing = sorted(set(col_map) - set(available))
    if missing:
        info["cols_missing"] = "; ".join(missing)
        logger.warning("%s: %d expected column(s) missing: %s",
                       path.name, len(missing), ", ".join(missing))

    out = df[available].apply(pd.to_numeric, errors="coerce").astype("float64")
    out.columns = [col_map[c] for c in available]
    out.index = ts
    out.index.name = "timestamp"
    info["rows_kept"] = len(out)
    return out, info


# --------------------------------------------------------------------------- #
# Full ingestion
# --------------------------------------------------------------------------- #
def ingest_all(data_dir: Path = DEFAULT_DATA_DIR,
               output_dir: Path | None = None) -> pd.DataFrame:
    """Concatenate every export file into one wide, timestamp-indexed frame."""
    col_map = build_column_map()
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    frames: list[pd.DataFrame] = []
    logs: list[dict] = []
    t0 = time.perf_counter()

    for i, path in enumerate(files, 1):
        frame, info = load_raw_file(path, col_map)
        frames.append(frame)
        logs.append(info)
        if i % 100 == 0 or i == len(files):
            logger.info("  ingested %d / %d files (%.1fs)",
                        i, len(files), time.perf_counter() - t0)

    raw = pd.concat(frames)
    raw = raw[sorted(raw.columns)]  # stable column order: blocks, then vars

    # duplicate timestamps can only come from overlapping files -> report them
    dup_mask = raw.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        logger.warning("%d duplicate timestamps dropped (kept first occurrence)", n_dup)
    raw = raw[~dup_mask].sort_index()

    # ingestion log
    if output_dir is not None:
        log_df = pd.DataFrame(logs)
        log_df.to_csv(output_dir / "ingest_log.csv", index=False)
        logger.info("Ingestion log written to %s", output_dir / "ingest_log.csv")

    logger.info("Ingestion done: %d rows x %d columns, %s -> %s",
                len(raw), raw.shape[1], raw.index.min(), raw.index.max())
    return raw
