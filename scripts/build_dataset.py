"""Concatenate monthly processed price files into one cleaned two-year dataset."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from src.prep.clean import CleaningPolicy, clean_prices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="data/processed/mid_*.parquet")
    parser.add_argument("--out", default="data/processed/prices_full.parquet")
    parser.add_argument("--report", default="reports/quality/prices_full_quality.json")
    parser.add_argument("--primary", default="price_apx")
    args = parser.parse_args()

    # Exclude previously cleaned outputs so a re-run does not ingest its own
    # product, which would double the flag columns and corrupt the counts.
    files = sorted(f for f in glob.glob(args.pattern) if "_clean" not in f and "_full" not in f)
    if not files:
        raise SystemExit(f"no files matched {args.pattern}")
    print(f"found {len(files)} monthly files")

    frames = [pd.read_parquet(f) for f in files]
    combined = pd.concat(frames).sort_index()
    print(f"concatenated {len(combined)} rows, {combined.index[0]} .. {combined.index[-1]}")

    # Consecutive monthly fetches share boundary periods because the API filters
    # inclusively; the cleaner drops them but the count is worth reporting here.
    print(f"duplicate timestamps before cleaning: {int(combined.index.duplicated().sum())}")

    clean, report = clean_prices(combined, CleaningPolicy(primary_price_column=args.primary))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.out)
    report.to_json(args.report)

    print("\n=== data quality, full period ===")
    print(report.summary())
    if report.incomplete_day_detail:
        print("\nincomplete days (observed, expected):")
        for date, (obs, exp) in list(report.incomplete_day_detail.items())[:20]:
            print(f"  {date}  {obs} / {exp}")
        if len(report.incomplete_day_detail) > 20:
            print(f"  ... and {len(report.incomplete_day_detail) - 20} more")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
