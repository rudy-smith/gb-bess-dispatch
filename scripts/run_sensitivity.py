"""Sweep storage duration against round-trip efficiency and plot the surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_benchmark import run
from src.model.battery import BatterySpec

DURATIONS = [1.0, 2.0, 4.0]
EFFICIENCIES = [0.80, 0.85, 0.90, 0.92]


def plot_heatmap(grid: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = grid.to_numpy()

    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [f"{c:.0%}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{d:g} h" for d in grid.index])
    ax.set_xlabel("Round-trip efficiency")
    ax.set_ylabel("Storage duration")
    ax.set_title(
        "Perfect-foresight revenue, £/MW/year",
        loc="left", fontsize=11, pad=10,
    )

    # Annotate in the contrasting colour so labels stay legible at both ends.
    threshold = values.min() + 0.55 * (values.max() - values.min())
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j, i, f"{values[i, j]:,.0f}",
                ha="center", va="center", fontsize=9,
                color="white" if values[i, j] < threshold else "black",
            )

    fig.colorbar(im, ax=ax, label="£/MW/year")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/processed/prices_full.parquet")
    parser.add_argument("--price-col", default="price_apx")
    parser.add_argument("--degradation-cost", type=float, default=0.0)
    parser.add_argument("--out", default="data/processed/sensitivity.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    rows: list[dict[str, float]] = []

    for duration in DURATIONS:
        for efficiency in EFFICIENCIES:
            spec = BatterySpec(duration_h=duration, round_trip_efficiency=efficiency)
            daily = run(df, spec, args.price_col, args.degradation_cost)
            years = len(daily) / 365.25
            revenue = daily["net_revenue_gbp"].sum() / spec.power_mw / years
            rows.append(
                {
                    "duration_h": duration,
                    "efficiency": efficiency,
                    "revenue_per_mw_year": revenue,
                    "cycles_per_year": daily["equivalent_full_cycles"].sum() / years,
                    # Revenue per MWh of installed energy shows whether longer
                    # duration pays for the extra cells, which revenue per MW
                    # of connection cannot answer.
                    "revenue_per_mwh_installed": revenue / duration,
                }
            )
            print(f"{duration:g}h, {efficiency:.0%}  ->  £{revenue:>8,.0f}/MW/yr")

    results = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.out)

    grid = results.pivot(index="duration_h", columns="efficiency", values="revenue_per_mw_year")
    print("\n=== revenue £/MW/year ===")
    print(grid.round(0).to_string())

    per_mwh = results.pivot(index="duration_h", columns="efficiency", values="revenue_per_mwh_installed")
    print("\n=== revenue £/MWh installed ===")
    print(per_mwh.round(0).to_string())

    plot_heatmap(grid, Path("reports/figures/sensitivity_duration_efficiency.png"))
    print("\nfigure -> reports/figures/sensitivity_duration_efficiency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
