"""
Forecast-driven backtest: dispatch on day-ahead forecasts, settle at actual prices.

    python -m scripts.run_forecast_backtest
    python -m scripts.run_forecast_backtest --degradation-cost 10

Needs data/processed/forecasts.parquet (scripts.run_forecast). For every
out-of-sample delivery day, solves perfect foresight and one schedule per
forecast, settles every schedule at actual prices, and reports the share of the
perfect-foresight revenue each captures, on one matched set of days.

Writes data/processed/backtest_daily.parquet and
reports/figures/backtest_cumulative.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.forecast_dispatch import (
    bootstrap_capture,
    bootstrap_capture_difference,
    run_backtest,
)
from src.model.battery import BatterySpec
from src.model.milp import get_solver

FORECASTS = Path("data/processed/forecasts.parquet")
OUT = Path("data/processed/backtest_daily.parquet")
FIG = Path("reports/figures/backtest_cumulative.png")

# name -> forecast column. The naive forecasts are included so the value of the
# model is measured in pounds, not only in forecast error.
STRATEGIES = {
    "lightgbm": "pred_point",
    "naive_7d_mean": "baseline_7d_mean",
    "naive_d2": "baseline_d2",
    "naive_d7": "baseline_d7",
}
LABELS = {
    "perfect": "Perfect foresight",
    "lightgbm": "LightGBM forecast",
    "naive_7d_mean": "7-day same-time mean",
    "naive_d2": "D-2 same time",
    "naive_d7": "D-7 same time",
}


def table(daily: pd.DataFrame, spec: BatterySpec) -> pd.DataFrame:
    years = len(daily) / 365.25
    rows = []
    for name in ["perfect", *STRATEGIES]:
        net = daily[f"{name}_net"]
        row = {
            "strategy": LABELS[name],
            "gbp_per_mw_yr": net.sum() / spec.power_mw / years,
            "cycles_per_yr": daily[f"{name}_mwh"].sum() / spec.usable_energy_mwh / years,
            "loss_days": int((net < -1e-6).sum()),
        }
        if name == "perfect":
            row.update(capture=1.0, ci_low=float("nan"), ci_high=float("nan"))
        else:
            c, lo, hi = bootstrap_capture(daily, name)
            row.update(capture=c, ci_low=lo, ci_high=hi)
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def show(t: pd.DataFrame) -> str:
    out = t.copy()
    out["gbp_per_mw_yr"] = out["gbp_per_mw_yr"].map("{:,.0f}".format)
    out["cycles_per_yr"] = out["cycles_per_yr"].map("{:,.0f}".format)
    for c in ("capture", "ci_low", "ci_high"):
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    return out.to_string()


def plot(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in ["perfect", *STRATEGIES]:
        ax.plot(daily.index, daily[f"{name}_net"].cumsum() / 1000, label=LABELS[name],
                lw=2.2 if name in ("perfect", "lightgbm") else 1.2)
    ax.set_ylabel("Cumulative net revenue, £k per MW")
    ax.set_title("Forecast-driven dispatch settled at actual prices, out-of-sample")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degradation-cost", type=float, default=0.0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    spec = BatterySpec()
    frame = pd.read_parquet(FORECASTS)
    missing = set(STRATEGIES.values()) - set(frame.columns)
    if missing:
        raise SystemExit(f"{FORECASTS} lacks {sorted(missing)}: rerun python -m scripts.run_forecast")
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    print(spec.describe())
    print(f"degradation cost £{args.degradation_cost:g}/MWh; decision 11:00 on D-1; "
          "settled at actual prices\n")

    started = time.time()
    daily, dropped = run_backtest(frame, STRATEGIES, spec, args.degradation_cost, get_solver())
    print(f"{len(daily)} matched days, {daily.index.min():%Y-%m-%d} .. {daily.index.max():%Y-%m-%d}; "
          f"{len(dropped)} dropped; {time.time() - started:.0f}s")
    if dropped:
        print(f"dropped: {[f'{d:%Y-%m-%d}' for d in dropped][:12]}")

    print("\nall matched days:")
    print(show(table(daily, spec)))
    for year in sorted(daily.index.year.unique()):
        print(f"\n{year}:")
        print(show(table(daily[daily.index.year == year], spec)))

    print("\nLightGBM minus each naive forecast, capture points, paired 95% CI:")
    for name in [n for n in STRATEGIES if n != "lightgbm"]:
        for label, part in (("all days", daily), ("no suspect wind", daily[~daily["wind_suspect"]])):
            d, lo, hi = bootstrap_capture_difference(part, "lightgbm", name)
            print(f"  vs {LABELS[name]:<22} {label:<16} {d:+.1%}  [{lo:+.1%}, {hi:+.1%}]")

    clean = daily[~daily["wind_suspect"]]
    print(f"\nexcluding the {int(daily['wind_suspect'].sum())} days with suspect wind "
          "publication times:")
    print(show(table(clean, spec)))

    if not args.no_save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(OUT)
        plot(daily, FIG)
        print(f"\nwrote {OUT} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
