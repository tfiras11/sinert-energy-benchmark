"""Phase 1 pipeline entry point: ingestion + cleaning of the SInERT dataset.

Usage (from anywhere):
    python -I run_phase1.py [--data-dir PATH] [--out-dir PATH]

Outputs (in --out-dir, default: sg_project/output):
    raw_ingested.parquet     raw 15-min table, as read from the files
    cleaned_15min.parquet    cleaned / imputed table (Phase 1 - steps 1 & 2)
    cleaning_report.csv      per block/variable cleaning statistics
    ingest_log.csv           per-file ingestion bookkeeping
    summary.txt              human-readable run summary + verification
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# allow running this script from anywhere, in isolated mode (-I)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from cleaning import clean_dataset  # noqa: E402
from config import COL, MAIN_BLOCK  # noqa: E402
from ingestion import ingest_all  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_phase1")


def verify(cleaned: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    """Sanity checks with known ground truths from the exploration phase."""
    lines = []

    # 1) September 2025 total consumption of the main meter
    #    (raw exploration value: 88,116 kWh computed as counter differences)
    e = COL.format(block=MAIN_BLOCK, var="E_kWh")
    sep25 = cleaned.loc["2025-09-01":"2025-09-30 23:45", e]
    lines.append(f"[verify] Sept 2025 EM_VIRTUAL energy (cleaned): "
                 f"{sep25.sum():,.0f} kWh  (raw exploration reference: 88,116 kWh)")

    # 2) energy conservation: totals from cleaned deltas vs raw counter spans
    #    (gaps <= 2 days spread evenly, so totals must match closely)
    c = COL.format(block=MAIN_BLOCK, var="kWh_abs")
    raw_first = raw[c].dropna().iloc[0]
    raw_last = raw[c].dropna().iloc[-1]
    lines.append(f"[verify] Whole-period EM_VIRTUAL energy (cleaned sum): "
                 f"{cleaned[e].sum():,.0f} kWh")
    lines.append(f"[verify] Whole-period EM_VIRTUAL raw counter span:   "
                 f"{raw_last - raw_first:,.0f} kWh")

    # 3) per-step power consistency: mean(P_avg) * 0.25 h vs energy sum
    p = COL.format(block=MAIN_BLOCK, var="P_avg")
    both = cleaned[[e, p]].dropna()
    e_from_p = both[p].sum() * 0.25
    lines.append(f"[verify] Energy from mean power integration:      "
                 f"{e_from_p:,.0f} kWh  (same period as cleaned sum)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 - ingestion & cleaning")
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="dataset root containing the month folders")
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR,
                        help="where to write parquet/report outputs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---------------- step 1: ingestion -----------------------------------
    logger.info("=== STEP 1: INGESTION ===")
    raw = ingest_all(args.data_dir, args.out_dir)
    raw.to_parquet(args.out_dir / "raw_ingested.parquet")

    # ---------------- step 2: cleaning & imputation ------------------------
    logger.info("=== STEP 2: CLEANING & IMPUTATION ===")
    cleaned, report = clean_dataset(raw)
    cleaned.to_parquet(args.out_dir / "cleaned_15min.parquet")
    report.to_csv(args.out_dir / "cleaning_report.csv", index=False)

    # ---------------- summary ----------------------------------------------
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 1 - INGESTION & CLEANING SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Raw frame           : {len(raw):,} rows x {raw.shape[1]} cols")
    lines.append(f"Cleaned frame       : {len(cleaned):,} rows x {cleaned.shape[1]} cols")
    lines.append(f"Period              : {cleaned.index.min()}  ->  {cleaned.index.max()}")
    lines.append(f"Missing 15-min slots: {int(cleaned['flag_slot_missing'].sum()):,}")
    lines.append(f"Elapsed             : {time.perf_counter() - t0:.1f} s")
    lines.append("")
    lines.append("Cleaning report (per block/variable):")
    with pd.option_context("display.width", 120, "display.max_rows", 200):
        lines.append(report.to_string(index=False))
    lines.append("")
    lines.extend(verify(cleaned, raw))

    text = "\n".join(lines)
    (args.out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    logger.info("Outputs written to %s", args.out_dir)


if __name__ == "__main__":
    main()
