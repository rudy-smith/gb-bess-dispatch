"""Audit liquidity-defaulted Market Index Prices and their reach into the benchmark.
 
The Market Index Definition Statement defaults both the Market Index Price and
the Market Index Volume to zero for any settlement period whose qualifying
traded volume falls below the Individual Liquidity Threshold. A defaulted price
is an absence of liquidity, not a traded price of zero.
 
This script answers three questions, in order of how much they matter:
 
  1. How many defaulted periods are present in the series actually optimised?
  2. Do any periods show the combination the default rule forbids, which would
     mean the rule is being read wrongly?
  3. Which settlement days does the contamination reach? The optimiser consumes
     whole days, so a day is the unit of consequence, and any affected day's
     benchmark revenue is suspect until the period is re-treated.
 
The bias introduced by a spurious zero is one-directional rather than merely
noisy. A perfect-foresight optimiser sorts each day and selects the extremes, so
a fake price of zero is close to certain to be chosen as a charging period.
Every one of them inflates the benchmark, and a benchmark inflated on the
denominator deflates any later "percentage of perfect foresight captured".
 
Usage:
 
    python -m scripts.audit_liquidity_defaults \
        --file data/processed/prices_full.parquet \
        --price-col price_apx
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
 
from src.prep.clean import (
    INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH,
    PRICE_TO_VOLUME_COLUMN,
    defaulted_price_mask,
    liquidity_anomaly_mask,
)
 
 
def resolve_volume_column(frame: pd.DataFrame, price_col: str, override: str | None) -> str:
    """Find the volume column belonging to a price column, or fail with the options.
 
    Failing loudly here rather than silently skipping the test is deliberate: an
    audit that quietly checks nothing reports a clean result, which is worse
    than no audit at all.
    """
    candidate = override or PRICE_TO_VOLUME_COLUMN.get(price_col)
    if candidate is None:
        raise KeyError(
            f"no volume column is mapped for price column {price_col!r}. "
            f"Pass --volume-col explicitly. Available: {sorted(frame.columns)}"
        )
    if candidate not in frame.columns:
        raise KeyError(
            f"volume column {candidate!r} not present in the frame. "
            f"Available: {sorted(frame.columns)}"
        )
    return candidate
 
 
def summarise(frame: pd.DataFrame, price_col: str, volume_col: str, threshold: float) -> str:
    price = frame[price_col].astype(float)
    volume = frame[volume_col].astype(float)
 
    is_zero = price.eq(0.0)
    defaulted = defaulted_price_mask(price, volume, threshold)
    anomalies = liquidity_anomaly_mask(price, volume, threshold)
    traded_zero = is_zero & ~defaulted
 
    lines = [
        f"rows                        {len(frame):,}",
        f"threshold                   {threshold:g} MWh",
        f"price is exactly zero       {int(is_zero.sum()):,}",
        f"  of which defaulted        {int(defaulted.sum()):,}",
        f"  of which traded at zero   {int(traded_zero.sum()):,}",
        f"price missing               {int(price.isna().sum()):,}",
        f"volume missing              {int(volume.isna().sum()):,}",
        f"liquidity anomalies         {int(anomalies.sum()):,}",
    ]
 
    if anomalies.any():
        lines.append(
            "  WARNING: a non-zero price alongside sub-threshold volume cannot "
            "occur under the default rule. Explain this before trusting the test."
        )
 
    # Share of volume below the threshold, which says whether the sample sits
    # close to the cliff edge or comfortably clear of it.
    below = volume.lt(threshold) & volume.notna()
    lines.append(f"periods with volume < threshold  {int(below.sum()):,}")
 
    return "\n".join(lines)
 
 
def affected_days(frame: pd.DataFrame, defaulted: pd.Series, date_col: str) -> pd.DataFrame:
    """Defaulted-period counts per settlement day, worst first."""
    if date_col not in frame.columns:
        raise KeyError(
            f"settlement date column {date_col!r} not in frame. "
            f"Available: {sorted(frame.columns)}"
        )
    counts = defaulted.groupby(frame[date_col]).sum().rename("defaulted_periods")
    counts = counts[counts > 0].sort_values(ascending=False)
    return counts.to_frame()
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/processed/prices_full.parquet")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--volume-col", default=None)
    parser.add_argument("--date-col", default="settlement_date")
    parser.add_argument(
        "--threshold",
        type=float,
        default=INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH,
        help="Individual Liquidity Threshold in MWh, per the MIDS in force",
    )
    parser.add_argument("--out", default=None, help="optional Parquet path for affected days")
    args = parser.parse_args()
 
    frame = pd.read_parquet(args.file)
    volume_col = resolve_volume_column(frame, args.price_col, args.volume_col)
 
    print(f"file      {args.file}")
    print(f"price     {args.price_col}")
    print(f"volume    {volume_col}\n")
    print(summarise(frame, args.price_col, volume_col, args.threshold))
 
    defaulted = defaulted_price_mask(
        frame[args.price_col].astype(float),
        frame[volume_col].astype(float),
        args.threshold,
    )
 
    if not defaulted.any():
        print("\nno defaulted periods: the benchmark is unaffected by this mechanism")
        return 0
 
    days = affected_days(frame, defaulted, args.date_col)
    print(f"\nsettlement days containing at least one defaulted period: {len(days)}")
    print(days.head(25).to_string())
 
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        days.to_parquet(args.out)
        print(f"\nwritten: {args.out}")
 
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())

 