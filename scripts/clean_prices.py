"""CLI wrapper: clean a processed price file and emit a data-quality report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.prep.clean import CleaningPolicy, clean_prices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="processed parquet input")
    parser.add_argument("--out", default=None, help="cleaned parquet output")
    parser.add_argument("--report", default=None, help="quality report JSON output")
    parser.add_argument(
        "--primary",
        default="price_apx",
        help="price column to gate usability on",
    )
    parser.add_argument("--max-interp", type=int, default=2, help="max gap filled, periods")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    src = Path(args.file)
    df = pd.read_parquet(src)
    print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    policy = CleaningPolicy(
        primary_price_column=args.primary,
        max_interp_periods=args.max_interp,
    )
    clean, report = clean_prices(df, policy)

    out_path = Path(args.out or src.with_name(src.stem + "_clean.parquet"))
    report_path = Path(args.report or f"reports/quality/{src.stem}_quality.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path)
    report.to_json(report_path)

    print("\n=== data quality ===")
    print(report.summary())
    print(f"\ncleaned frame -> {out_path}")
    print(f"quality report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
