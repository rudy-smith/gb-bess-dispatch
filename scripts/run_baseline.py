"""Compare the MILP against the trailing-percentile threshold rule."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model.baselines import run_threshold_baseline
from src.model.battery import BatterySpec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/processed/prices_full.parquet")
    parser.add_argument("--benchmark", default="data/processed/benchmark_daily.parquet")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--duration-h", type=float, default=2.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--out", default="data/processed/baseline_daily.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    spec = BatterySpec(duration_h=args.duration_h, round_trip_efficiency=args.efficiency)

    baseline = run_threshold_baseline(df, spec, args.price_col, args.window_days)
    optimal = pd.read_parquet(args.benchmark)

    joined = optimal[["net_revenue_gbp", "equivalent_full_cycles"]].join(
        baseline[["net_revenue_gbp", "equivalent_full_cycles"]],
        how="inner", lsuffix="_milp", rsuffix="_rule",
    )
    years = len(joined) / 365.25

    milp = joined["net_revenue_gbp_milp"].sum() / spec.power_mw / years
    rule = joined["net_revenue_gbp_rule"].sum() / spec.power_mw / years

    print(f"days compared          {len(joined)}")
    print(f"MILP                   £{milp:,.0f}/MW/yr")
    print(f"threshold rule         £{rule:,.0f}/MW/yr")
    print(f"rule captures          {100 * rule / milp:.1f}% of perfect foresight")
    print(f"MILP cycles/yr         {joined['equivalent_full_cycles_milp'].sum() / years:.1f}")
    print(f"rule cycles/yr         {joined['equivalent_full_cycles_rule'].sum() / years:.1f}")

    beat = (joined["net_revenue_gbp_milp"] > joined["net_revenue_gbp_rule"] + 1e-6).mean()
    print(f"days MILP beats rule   {100 * beat:.1f}%")
    print(f"rule days at a loss    {int((joined['net_revenue_gbp_rule'] < 0).sum())}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
