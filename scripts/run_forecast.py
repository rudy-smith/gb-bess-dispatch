"""
Walk-forward price forecasts, scored against naive baselines.

    python -m scripts.run_forecast
    python -m scripts.run_forecast --no-anchor
    python -m scripts.run_forecast --drop-features f_last_known_price --no-save

Needs data/processed/features.parquet (scripts.build_features). Writes:

    data/processed/forecasts.parquet          out-of-sample forecasts per half hour
    reports/figures/forecast_mae_by_hour.png
    reports/figures/forecast_reliability.png

Test months run from July 2023 to December 2024, so the first model has six
months of history. Every number printed is on out-of-sample months only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.forecast.evaluate import (
    BASELINES,
    add_baselines,
    block_bootstrap_skill,
    interval_coverage,
    mae_by_hour,
    reliability,
    score_table,
)
from src.forecast.walkforward import monthly_folds, walk_forward
from src.prep.features import FEATURES

FEATURES_FILE = Path("data/processed/features.parquet")
OUT = Path("data/processed/forecasts.parquet")
FIG_DIR = Path("reports/figures")
QUANTILES = {0.1: "pred_q10", 0.5: "pred_q50", 0.9: "pred_q90"}


def plot_by_hour(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = {"pred_point": "LightGBM", "baseline_d2": "D-2 same time",
              "baseline_d7": "D-7 same time", "baseline_7d_mean": "7-day same-time mean"}
    for col in table.columns:
        ax.plot(table.index, table[col], label=labels.get(col, col),
                lw=2.2 if col == "pred_point" else 1.3)
    ax.set_xlabel("Hour of day (London)")
    ax.set_ylabel("MAE, £/MWh")
    ax.set_title("Day-ahead price forecast error by hour, out-of-sample")
    ax.set_xticks(range(0, 24, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
    ax.plot(table.index, table["observed"], marker="o", lw=2, label="LightGBM quantiles")
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Observed share of outcomes below")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Quantile reliability, out-of-sample")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-test", default="2023-07-01")
    parser.add_argument("--last-test", default="2024-12-01")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--drop-features", nargs="+", default=[],
                        help="ablation: train without these features")
    parser.add_argument("--no-save", action="store_true",
                        help="print scores only; leave forecasts and figures untouched")
    args = parser.parse_args()

    frame = add_baselines(pd.read_parquet(FEATURES_FILE))
    unknown = set(args.drop_features) - set(FEATURES)
    if unknown:
        raise SystemExit(f"not registered features: {sorted(unknown)}")
    features = [f for f in FEATURES if f not in args.drop_features]
    if args.drop_features:
        print(f"ablation: dropped {args.drop_features}")
    folds = monthly_folds(args.first_test, args.last_test)
    print(f"{len(folds)} monthly folds, {folds[0].label} .. {folds[-1].label}; "
          f"{len(features)} features; anchored={not args.no_anchor}")

    result = walk_forward(frame, features, folds, tuple(QUANTILES), anchored=not args.no_anchor)
    compare = ["pred_point", *BASELINES]

    print("\nout-of-sample, matched rows:")
    print(score_table(result, compare, "baseline_d7").round(3).to_string())
    for year in (2023, 2024):
        part = result[pd.to_datetime(result["settlement_date"]).dt.year == year]
        print(f"\n{year}:")
        print(score_table(part, compare, "baseline_d7").round(3).to_string())

    best = min(BASELINES, key=lambda b: score_table(result, compare, b).loc[b, "mae"])
    s, lo, hi = block_bootstrap_skill(result, "pred_point", best)
    print(f"\nMAE skill over the best baseline ({best}): {s:.1%}  95% CI [{lo:.1%}, {hi:.1%}]"
          "  (7-day block bootstrap over days)")

    # The 2023 wind block. If its forecasts had been rebuilt with hindsight, the
    # model would be conspicuously better on those days than on the same calendar
    # window of 2024. The comparison is against the best baseline, so a change in
    # how hard the two summers were to forecast cancels out.
    day = pd.to_datetime(result["settlement_date"])
    block = result[result["wind_suspect"].fillna(False).astype(bool)]
    same_window_2024 = result[(day >= "2024-06-01") & (day <= "2024-10-03")]
    for label, part in (("2023 wind block", block), ("same window 2024", same_window_2024)):
        if len(part):
            k, klo, khi = block_bootstrap_skill(part, "pred_point", best)
            print(f"  {label:<18} skill {k:.1%}  [{klo:.1%}, {khi:.1%}]  "
                  f"({part['settlement_date'].nunique()} days)")

    rel = reliability(result, QUANTILES)
    print("\nquantiles:")
    print(rel.round(3).to_string())
    print(f"10-90 interval coverage: {interval_coverage(result, 'pred_q10', 'pred_q90'):.1%} "
          "(nominal 80%)")

    if args.no_save:
        return 0

    by_hour = mae_by_hour(result, compare)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_by_hour(by_hour, FIG_DIR / "forecast_mae_by_hour.png")
    plot_reliability(rel, FIG_DIR / "forecast_reliability.png")

    keep = ["target_time", "settlement_date", "settlement_period", "decision_time", "fold",
            "target_price", "day_usable", "wind_suspect", *BASELINES, "pred_point",
            *QUANTILES.values()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result[keep].to_parquet(OUT, index=False)
    print(f"\nwrote {OUT} ({len(result):,} rows) and two figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
