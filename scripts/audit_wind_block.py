"""
Test whether the 2023 block of late-stamped NESO wind forecasts was regenerated.

    python -m scripts.audit_wind_block
    python -m scripts.audit_wind_block --control-years 2022 2024

Needs the cached NESO wind archive (data/raw/neso/wind_da.parquet) and the FUELHH
outturn (data/processed/fuelhh.parquet, from scripts.fetch_fuelhh).

The block is derived from the data - rows in the study year whose recorded
publication instant does not precede their settlement day - rather than typed in,
so the test examines whatever the archive actually contains. The controls are the
same calendar window in each control year.

The method and the thresholds are in src/prep/forecast_skill.py. Output: a table
of skill by window, the bootstrap ratio and verdict, a figure, and the joined
forecast/outturn frame at data/processed/wind_forecast_outturn.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.fetch.neso import (
    RESOURCES,
    cache_path,
    day_start_utc,
    normalise_wind,
    resolve_published_at,
)
from src.prep.forecast_skill import (
    add_errors,
    bootstrap_ratio,
    contiguous_runs,
    daily_mae,
    in_calendar_window,
    join_forecast_outturn,
    summarise,
    verdict,
)

FUELHH = Path("data/processed/fuelhh.parquet")
OUT = Path("data/processed/wind_forecast_outturn.parquet")
FIG = Path("reports/figures/wind_block_skill.png")


def load_forecast(raw_root: Path, years: list[int]) -> pd.DataFrame:
    """Normalised wind forecasts for the given years, with the suspect flag."""
    resource = RESOURCES["wind_da"]
    raw = pd.read_parquet(cache_path(raw_root, resource.name))
    raw = raw[pd.to_datetime(raw["Date"]).dt.year.isin(years)].copy()

    resolved = resolve_published_at(raw, resource, "Date", day_start_utc(raw["Date"]))
    flags = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(resolved["Date"]).dt.normalize(),
            "settlement_period": resolved["Settlement_period"].astype(int),
            "published_at_suspect": resolved["published_at_suspect"].to_numpy(),
        }
    )
    norm = normalise_wind(raw).reset_index()
    return norm.merge(flags, on=["settlement_date", "settlement_period"], validate="one_to_one")


def load_outturn() -> pd.DataFrame:
    if not FUELHH.exists():
        raise SystemExit(f"{FUELHH} not found: run python -m scripts.fetch_fuelhh first")
    df = pd.read_parquet(FUELHH)
    return pd.DataFrame(
        {
            "settlement_date": df["settlement_date"],
            "settlement_period": df["settlement_period"],
            "wind_outturn_mw": df["gen_wind_mw"],
            "outturn_start_utc": df.index,
        }
    )


def plot(joined, windows, block, out_path):
    """Daily MAE through the calendar window, one line per year, block year bold."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for label, frame in windows.items():
        d = daily_mae(frame)
        x = d.index.dayofyear
        is_block = label.startswith("block")
        ax1.plot(x, 100 * d.rolling(7, min_periods=3).mean().to_numpy(), label=label,
                 lw=2.2 if is_block else 1.2, color="C3" if is_block else None)
        by_sp = frame.groupby("settlement_period")["abs_error_share"].mean() * 100
        ax2.plot(by_sp.index, by_sp.to_numpy(), label=label,
                 lw=2.2 if is_block else 1.2, color="C3" if is_block else None)

    ax1.set_title(f"Day-ahead wind forecast error, {block[0]:%d %b} to {block[1]:%d %b}")
    ax1.set_xlabel("Day of year")
    ax1.set_ylabel("MAE, % of capacity\n(7-day rolling)")
    ax1.legend(frameon=False)
    ax2.set_title("Error by settlement period (later periods are further from publication)")
    ax2.set_xlabel("Settlement period")
    ax2.set_ylabel("MAE, % of capacity")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--block-year", type=int, default=2023)
    parser.add_argument("--control-years", type=int, nargs="+", default=[2022, 2024])
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    years = sorted({args.block_year, *args.control_years})
    forecast = load_forecast(args.raw_root, years)
    joined = add_errors(join_forecast_outturn(forecast, load_outturn()))
    print(f"joined rows: {len(joined):,} (forecast {len(forecast):,})")

    # Two independent sources claim the UTC start of each period. They must agree,
    # or the join is aligned on labels that mean different things.
    ts_neso = pd.to_datetime(joined["timestamp_utc"], utc=True)
    ts_elx = pd.to_datetime(joined["outturn_start_utc"], utc=True)
    print(f"NESO Datetime_GMT == project calendar start: {(ts_neso == ts_elx).sum():,} "
          f"of {len(joined):,}")

    missing = joined["wind_outturn_mw"].isna().sum() + joined["wind_forecast_mw"].isna().sum()
    joined = joined.dropna(subset=["wind_outturn_mw", "wind_forecast_mw", "abs_error_share"])
    print(f"rows dropped for a missing value: {missing}")

    # --- locate the block ----------------------------------------------------
    in_year = joined["settlement_date"].dt.year == args.block_year
    suspect_days = joined.loc[in_year & joined["published_at_suspect"], "settlement_date"]
    runs = contiguous_runs(suspect_days)
    print(f"\nsuspect runs in {args.block_year}:")
    for a, b in runs:
        print(f"  {a:%Y-%m-%d} .. {b:%Y-%m-%d}  ({(b - a).days + 1} days)")
    if not runs:
        print("no suspect rows; nothing to test")
        return 0
    block = max(runs, key=lambda r: r[1] - r[0])

    # --- windows ----------------------------------------------------------------
    same_season = in_calendar_window(joined["settlement_date"], *block)
    windows = {
        f"block {args.block_year}": joined[same_season & in_year],
    }
    for y in args.control_years:
        windows[f"control {y}"] = joined[same_season & (joined["settlement_date"].dt.year == y)]
    windows[f"{args.block_year} outside block (different season)"] = joined[
        in_year & ~same_season & ~joined["published_at_suspect"]
    ]

    table = pd.DataFrame({k: summarise(v) for k, v in windows.items()}).T
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print("\n" + table.to_string())

    # --- the test ----------------------------------------------------------------
    block_daily = daily_mae(windows[f"block {args.block_year}"])
    controls = pd.concat([daily_mae(windows[f"control {y}"]) for y in args.control_years])
    ratio, lo, hi = bootstrap_ratio(block_daily, controls, n_boot=args.n_boot)
    print(f"\nMAE ratio, block / controls: {ratio:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    if len(args.control_years) >= 2:
        a, b = args.control_years[:2]
        cr, clo, chi = bootstrap_ratio(
            daily_mae(windows[f"control {a}"]), daily_mae(windows[f"control {b}"]),
            n_boot=args.n_boot,
        )
        print(f"reference, control {a} / control {b}: {cr:.3f}  95% CI [{clo:.3f}, {chi:.3f}]"
              "   (genuine year-to-year variation)")

    v = verdict(lo, hi)
    print(f"\nVERDICT: {v.label}. {v.reason}.")

    joined.to_parquet(OUT)
    plot(joined, {k: w for k, w in windows.items() if "outside" not in k}, block, FIG)
    print(f"\nwrote {OUT} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
