"""
Fetch GB half-hourly Market Index Price data, report on its quality, and plot it.
 
    python -m scripts.fetch_prices --start 2024-10-01 --end 2024-10-31
    python -m scripts.fetch_prices --start 2024-10-01 --end 2024-10-31 --force
 
Dates are settlement dates, both inclusive. Results are cached, so re-running is
cheap; --force bypasses the cache and re-hits the API.
 
Parameterised via argparse rather than hardcoded so that the command which
produced any given figure is recoverable from shell history. Every figure in the
final report must be reproducible from a single command.
"""
 
from __future__ import annotations
 
import argparse
import logging
from pathlib import Path
 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
 
from src.fetch.elexon import fetch_mid, mid_quality_report
 
FIG_DIR = Path("reports/figures")
PROCESSED_DIR = Path("data/processed")
 
 
def plot_prices(df, start: str, end: str, out_path: Path):
    """Plot the half-hourly price series and the daily spread beneath it.
 
    Two panels rather than one because they answer different questions. The top
    panel shows the raw signal; the bottom shows how much intraday spread exists
    to arbitrage, which is the quantity that ultimately determines whether a
    battery earns anything at all.
 
    The x-axis is local (Europe/London) time. Price shape is driven by human
    activity, which follows the wall clock. A UTC axis would smear the evening
    peak by an hour across the BST boundary and make seasonal comparison
    misleading.
 
    Returns the daily spread series so the caller can summarise it.
    """
    d = df.dropna(subset=["price_mid_gbp_mwh"]).copy()
    d = d.set_index("start_time_local")
 
    daily = d["price_mid_gbp_mwh"].resample("1D")
    lo, hi = daily.min(), daily.max()
    spread = hi - lo
 
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1]
    )
 
    ax1.plot(d.index, d["price_mid_gbp_mwh"], lw=0.8, color="#1b3a5c")
    ax1.fill_between(
        lo.index, lo, hi, alpha=0.15, color="#1b3a5c", step="post",
        label="daily min-max range",
    )
    ax1.axhline(0, color="#c1440e", lw=0.8, ls="--", label="zero")
    ax1.set_ylabel("Price (£/MWh)")
    ax1.set_title(
        f"GB half-hourly Market Index Price (volume-weighted N2EX + APX), "
        f"{start} to {end}",
        loc="left", fontsize=11, fontweight="bold",
    )
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
 
    ax2.bar(spread.index, spread.to_numpy(), width=0.8, color="#3d7ea6")
    ax2.set_ylabel("Daily spread\n(£/MWh)")
    ax2.set_xlabel("Local time (Europe/London)")
 
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
 
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return spread
 
 
def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch, validate and plot GB half-hourly Market Index Prices"
    )
    p.add_argument("--start", required=True, help="first settlement date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last settlement date, YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="ignore the Parquet cache")
    args = p.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
 
    df = fetch_mid(args.start, args.end, force=args.force)
 
    print("\n=== MID data quality report ===")
    print(mid_quality_report(df).to_string(index=False))
 
    fig_path = FIG_DIR / f"mid_prices_{args.start}_{args.end}.png"
    spread = plot_prices(df, args.start, args.end, fig_path)
    print(f"\nfigure written to {fig_path}")
 
    print("\n=== daily spread (the arbitrage signal) ===")
    print(f"  mean   £{spread.mean():.2f}/MWh")
    print(f"  median £{spread.median():.2f}/MWh")
    print(f"  max    £{spread.max():.2f}/MWh on {spread.idxmax():%Y-%m-%d}")
    print(f"  min    £{spread.min():.2f}/MWh on {spread.idxmin():%Y-%m-%d}")
 
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DIR / f"mid_{args.start}_{args.end}.parquet"
    df.to_parquet(out_file)
    print(f"\nprocessed data written to {out_file}")
 
 
if __name__ == "__main__":
    main()
 