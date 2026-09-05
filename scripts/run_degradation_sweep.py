"""Sweep the degradation cost and trace the revenue-versus-throughput frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.run_benchmark import run
from src.model.battery import BatterySpec

DEFAULT_COSTS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0]


def plot_frontier(frontier: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        frontier["cycles_per_year"],
        frontier["gross_revenue_per_mw_year"],
        marker="o", color="#2f6690", linewidth=1.6, markersize=5,
    )
    for _, row in frontier.iterrows():
        ax.annotate(
            f"£{row['degradation_cost']:g}",
            (row["cycles_per_year"], row["gross_revenue_per_mw_year"]),
            textcoords="offset points", xytext=(6, -10), fontsize=8, color="#555555",
        )

    ax.set_xlabel("Equivalent full cycles per year")
    ax.set_ylabel("Gross trading revenue, £/MW/year")
    ax.set_title(
        "Revenue against cycling as degradation cost rises\n"
        "labels show the degradation cost applied, £/MWh discharged",
        loc="left", fontsize=11, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/processed/prices_full.parquet")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--duration-h", type=float, default=2.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--costs", type=float, nargs="*", default=DEFAULT_COSTS)
    parser.add_argument("--out", default="data/processed/degradation_frontier.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    rows: list[dict[str, float]] = []

    for cost in args.costs:
        spec = BatterySpec(duration_h=args.duration_h, round_trip_efficiency=args.efficiency)
        daily = run(df, spec, args.price_col, cost)
        years = len(daily) / 365.25
        rows.append(
            {
                "degradation_cost": cost,
                # Gross revenue is reported because the degradation charge is an
                # internal shadow price steering the optimiser, not a cash cost
                # the operator pays to anyone. Reporting net would confuse a
                # decision weight with an expense.
                "gross_revenue_per_mw_year": daily["gross_revenue_gbp"].sum() / spec.power_mw / years,
                "net_of_wear_per_mw_year": daily["net_revenue_gbp"].sum() / spec.power_mw / years,
                "throughput_per_year": daily["throughput_mwh"].sum() / years,
                "cycles_per_year": daily["equivalent_full_cycles"].sum() / years,
                "days": len(daily),
            }
        )
        print(
            f"£{cost:>5.1f}/MWh  ->  £{rows[-1]['gross_revenue_per_mw_year']:>8,.0f}/MW/yr  "
            f"{rows[-1]['cycles_per_year']:>6.1f} cycles/yr"
        )

    frontier = pd.DataFrame(rows)
    base = frontier.iloc[0]
    frontier["revenue_retained_pct"] = (
        100 * frontier["gross_revenue_per_mw_year"] / base["gross_revenue_per_mw_year"]
    )
    frontier["cycles_retained_pct"] = 100 * frontier["cycles_per_year"] / base["cycles_per_year"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frontier.to_parquet(args.out)

    print("\n=== degradation frontier ===")
    print(frontier.round(1).to_string(index=False))

    plot_frontier(frontier, Path("reports/figures/degradation_frontier.png"))
    print("\nfigure -> reports/figures/degradation_frontier.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
