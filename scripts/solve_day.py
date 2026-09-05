"""Solve a single settlement day and plot price, dispatch and state of charge."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.model.battery import BatterySpec
from src.model.milp import solve_dispatch

PRICE_COLOUR = "#333333"
CHARGE_COLOUR = "#2f6690"
DISCHARGE_COLOUR = "#b5442f"
SOC_COLOUR = "#3a7d44"


def load_day(path: Path, date: str, price_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    day = df[pd.to_datetime(df["settlement_date"]).dt.date == pd.Timestamp(date).date()]
    if day.empty:
        raise SystemExit(f"no rows for settlement date {date} in {path}")
    if "day_usable" in day.columns and not bool(day["day_usable"].iloc[0]):
        raise SystemExit(f"{date} is flagged unusable; refusing to optimise an incomplete day")
    if day[price_col].isna().any():
        raise SystemExit(f"{date} has missing prices in {price_col}")
    return day.sort_index()


def plot_day(day: pd.DataFrame, result, spec: BatterySpec, date: str, out_path: Path) -> None:
    trace = result.trace
    x = day["start_time_local"] if "start_time_local" in day.columns else trace.index

    fig, (ax_price, ax_power, ax_soc) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1.6], "hspace": 0.12},
    )

    ax_price.step(x, trace["price_gbp_mwh"], where="post", color=PRICE_COLOUR, linewidth=1.4)
    ax_price.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax_price.set_ylabel("Price\n£/MWh")
    ax_price.set_title(
        f"Perfect-foresight dispatch, {date}\n{spec.describe()}",
        loc="left", fontsize=11, pad=10,
    )

    ax_power.bar(x, trace["discharge_mw"], width=0.02, color=DISCHARGE_COLOUR, label="Discharge (export)")
    ax_power.bar(x, -trace["charge_mw"], width=0.02, color=CHARGE_COLOUR, label="Charge (import)")
    ax_power.axhline(0, color="#999999", linewidth=0.8)
    ax_power.set_ylabel("Power\nMW")
    ax_power.legend(frameon=False, fontsize=9, loc="upper left")

    ax_soc.fill_between(x, trace["soc_mwh"], step="post", color=SOC_COLOUR, alpha=0.25)
    ax_soc.step(x, trace["soc_mwh"], where="post", color=SOC_COLOUR, linewidth=1.4)
    ax_soc.axhline(spec.soc_max_mwh, color="#999999", linewidth=0.8, linestyle="--")
    ax_soc.axhline(spec.soc_min_mwh, color="#999999", linewidth=0.8, linestyle="--")
    ax_soc.set_ylabel("Stored energy\nMWh")
    ax_soc.set_xlabel("Local time")

    for ax in (ax_price, ax_power, ax_soc):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--date", required=True, help="settlement date, YYYY-MM-DD")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--power-mw", type=float, default=1.0)
    parser.add_argument("--duration-h", type=float, default=2.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--degradation-cost", type=float, default=0.0, help="£/MWh discharged")
    parser.add_argument("--compare-relaxation", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    day = load_day(Path(args.file), args.date, args.price_col)
    spec = BatterySpec(
        power_mw=args.power_mw,
        duration_h=args.duration_h,
        round_trip_efficiency=args.efficiency,
    )
    prices = day[args.price_col]

    print(f"{args.date}: {len(prices)} settlement periods")
    print(spec.describe())

    result = solve_dispatch(
        prices, spec, degradation_cost_gbp_per_mwh=args.degradation_cost
    )
    print("\n=== mixed-integer solution ===")
    print(result.summary())

    if args.compare_relaxation:
        relaxed = solve_dispatch(
            prices, spec,
            degradation_cost_gbp_per_mwh=args.degradation_cost,
            relax_binaries=True,
        )
        gap = relaxed.objective_gbp - result.objective_gbp
        print("\n=== linear relaxation ===")
        print(relaxed.summary())
        print(
            f"\nrelaxation exceeds integer optimum by GBP {gap:,.4f} "
            f"({gap / result.objective_gbp:.2%})"
            if result.objective_gbp
            else f"\nrelaxation gap GBP {gap:,.4f}"
        )

    out = Path(f"reports/figures/dispatch_{args.date}.png")
    plot_day(day, result, spec, args.date, out)
    print(f"\nfigure written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
