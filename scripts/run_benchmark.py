"""Perfect-foresight benchmark: solve every usable settlement day and aggregate."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.model.battery import BatterySpec
from src.model.milp import get_solver, solve_dispatch


def run(
    df: pd.DataFrame,
    spec: BatterySpec,
    price_col: str,
    degradation_cost: float,
) -> pd.DataFrame:
    """Solve each usable day independently and return one row per day.

    Days are solved independently with matched opening and closing state of
    charge, which makes daily revenues additive and each day self-contained. The
    cost is the overnight arbitrage between a cheap night and the following
    evening peak, which this formulation cannot capture.
    """
    solver = get_solver()
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    grouped = df.groupby(pd.to_datetime(df["settlement_date"]).dt.date, sort=True)
    total = len(grouped)
    started = time.time()

    for i, (date, day) in enumerate(grouped, start=1):
        if "day_usable" in day.columns and not bool(day["day_usable"].iloc[0]):
            skipped.append(str(date))
            continue
        if day[price_col].isna().any():
            skipped.append(str(date))
            continue

        prices = day.sort_index()[price_col]
        result = solve_dispatch(
            prices, spec, degradation_cost_gbp_per_mwh=degradation_cost, solver=solver
        )
        if not result.solved or result.violations:
            skipped.append(str(date))
            continue

        rows.append(
            {
                "settlement_date": pd.Timestamp(date),
                "periods": len(prices),
                "price_mean": float(prices.mean()),
                "price_min": float(prices.min()),
                "price_max": float(prices.max()),
                "naive_spread": float(prices.max() - prices.min()),
                "gross_revenue_gbp": result.gross_revenue_gbp,
                "degradation_cost_gbp": result.degradation_cost_gbp,
                "net_revenue_gbp": result.net_revenue_gbp,
                "throughput_mwh": result.throughput_mwh,
                "equivalent_full_cycles": result.equivalent_full_cycles,
            }
        )

        if i % 100 == 0:
            print(f"  {i}/{total} days, {time.time() - started:.0f}s elapsed")

    if skipped:
        print(f"skipped {len(skipped)} days: {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    return pd.DataFrame(rows).set_index("settlement_date")


def summarise(daily: pd.DataFrame, spec: BatterySpec) -> str:
    """Annualise the daily results and express revenue per MW of connection."""
    days = len(daily)
    years = days / 365.25
    net = daily["net_revenue_gbp"].sum()
    per_mw_year = net / spec.power_mw / years
    cycles_per_year = daily["equivalent_full_cycles"].sum() / years

    return "\n".join(
        [
            f"days solved            {days}",
            f"coverage               {daily.index.min().date()} .. {daily.index.max().date()}",
            f"gross revenue          GBP {daily['gross_revenue_gbp'].sum():,.0f}",
            f"degradation cost       GBP {daily['degradation_cost_gbp'].sum():,.0f}",
            f"net revenue            GBP {net:,.0f}",
            "",
            f"revenue per MW-year    GBP {per_mw_year:,.0f} /MW/yr",
            f"equivalent cycles/yr   {cycles_per_year:,.1f}",
            f"throughput/yr          {daily['throughput_mwh'].sum() / years:,.0f} MWh",
            "",
            f"daily revenue mean     GBP {daily['net_revenue_gbp'].mean():,.2f}",
            f"daily revenue median   GBP {daily['net_revenue_gbp'].median():,.2f}",
            f"daily revenue p90      GBP {daily['net_revenue_gbp'].quantile(0.90):,.2f}",
            f"best day               GBP {daily['net_revenue_gbp'].max():,.2f} "
            f"({daily['net_revenue_gbp'].idxmax().date()})",
            f"zero-revenue days      {int((daily['net_revenue_gbp'] <= 1e-6).sum())}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/processed/prices_full.parquet")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--power-mw", type=float, default=1.0)
    parser.add_argument("--duration-h", type=float, default=2.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--degradation-cost", type=float, default=0.0)
    parser.add_argument("--out", default="data/processed/benchmark_daily.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    spec = BatterySpec(
        power_mw=args.power_mw,
        duration_h=args.duration_h,
        round_trip_efficiency=args.efficiency,
    )
    print(spec.describe())
    print(f"degradation cost GBP {args.degradation_cost:g}/MWh discharged\n")

    daily = run(df, spec, args.price_col, args.degradation_cost)
    if daily.empty:
        raise SystemExit("no days solved")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.out)

    print("\n=== perfect-foresight benchmark ===")
    print(summarise(daily, spec))

    monthly = daily.resample("MS")[["net_revenue_gbp", "throughput_mwh"]].sum()
    monthly["cycles"] = daily.resample("MS")["equivalent_full_cycles"].sum()
    print("\n=== monthly ===")
    print(monthly.round(1).to_string())
    print(f"\ndaily results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
