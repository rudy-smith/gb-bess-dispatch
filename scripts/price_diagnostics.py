"""
Interrogate a fetched MID price series before trusting it.
 
    python -m scripts.price_diagnostics --file data/processed/mid_2024-10-01_2024-10-31.parquet
 
Answers two questions that determine whether the headline numbers mean anything:
 
1. Are the two Market Index Data Providers reporting comparable quantities? A
   volume share of 99.99/0.01 is more consistent with a definitional difference
   than with a liquidity difference, and the two have very different
   implications for which price series the optimiser should use.
 
2. How much of the naive daily high-minus-low spread is actually reachable by a
   battery of a given duration and round-trip efficiency? The high-low spread is
   an upper bound that no asset can capture, and quoting it as though it were
   revenue is the most common way a storage model overstates itself.
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
FIG_DIR = Path("reports/figures")
 
 
def compare_providers(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of the two MIDP series.
 
    The volume statistics are reported as medians as well as sums. A provider
    that reports plausible volumes most of the time but occasionally reports
    zero looks identical in a sum to one that reports a constant near-zero, and
    those are different problems with different fixes.
    """
    rows = {}
    for prov in ("n2ex", "apx"):
        p, v = f"price_{prov}", f"volume_{prov}"
        if p not in df.columns:
            continue
        rows[prov] = {
            "periods_with_price": int(df[p].notna().sum()),
            "periods_with_volume>0": int((df[v] > 0).sum()),
            "median_volume_MWh": round(float(df[v].median()), 2),
            "mean_volume_MWh": round(float(df[v].mean()), 2),
            "max_volume_MWh": round(float(df[v].max()), 2),
            "total_volume_MWh": round(float(df[v].sum()), 1),
            "median_price": round(float(df[p].median()), 2),
            "mean_price": round(float(df[p].mean()), 2),
        }
    out = pd.DataFrame(rows).T
 
    # Price agreement matters more than volume agreement. If the two venues
    # price the same half-hour almost identically, the choice of provider barely
    # affects the dispatch result and the volume anomaly is a footnote. If they
    # diverge, the choice is load-bearing and has to be justified.
    both = df.dropna(subset=["price_n2ex", "price_apx"])
    if len(both):
        diff = both["price_n2ex"] - both["price_apx"]
        print(f"\nperiods where both providers quote a price: {len(both)}")
        print(f"  correlation                : {both['price_n2ex'].corr(both['price_apx']):.4f}")
        print(f"  mean   (N2EX - APX)        : £{diff.mean():+.2f}/MWh")
        print(f"  median (N2EX - APX)        : £{diff.median():+.2f}/MWh")
        print(f"  mean absolute difference   : £{diff.abs().mean():.2f}/MWh")
        print(f"  95th pct absolute diff     : £{diff.abs().quantile(0.95):.2f}/MWh")
        print(f"  periods differing by >£5   : {int((diff.abs() > 5).sum())}")
 
    return out
 
 
def realisable_spread(
    df: pd.DataFrame, durations_h=(1.0, 2.0, 4.0), eta: float = 0.90
) -> pd.DataFrame:
    """Daily spread a battery of each duration could actually reach.
 
    The naive metric is max(price) - min(price) over the day. That is only
    achievable by an asset which can fill and empty within a single half-hour,
    i.e. one with zero duration. A real battery must charge across several
    consecutive-or-not cheap periods and discharge across several dear ones, so
    the relevant quantity is the mean of the k dearest periods minus the mean of
    the k cheapest, where k = duration in half-hours.
 
    Efficiency is charged on the buy side: to deliver 1 MWh out, 1/eta MWh must
    be bought in, so the margin per MWh discharged is
 
        mean(top k) - mean(bottom k) / eta
 
    Two simplifications, both of which make this an OPTIMISTIC bound:
      - it assumes the k cheapest and k dearest periods can be used freely,
        ignoring the ordering constraint that charging must precede discharging;
      - it ignores that charging needs k/eta periods of power, not k, so a
        power-limited battery needs more charge periods than discharge periods.
    The MILP handles both properly. This is a sanity check, not a result.
    """
    out = []
    for dur in durations_h:
        k = round(dur * 2)  # half-hour periods
        recs = []
        for _, day in df.dropna(subset=["price_mid_gbp_mwh"]).groupby("settlement_date"):
            p = np.sort(day["price_mid_gbp_mwh"].to_numpy())
            if len(p) < 2 * k:
                continue
            lo, hi = p[:k].mean(), p[-k:].mean()
            recs.append(
                {
                    "naive": p[-1] - p[0],
                    "topk_minus_botk": hi - lo,
                    "after_efficiency": hi - lo / eta,
                }
            )
        r = pd.DataFrame(recs)
        naive_mean = r["naive"].mean()
        out.append(
            {
                "duration_h": dur,
                "periods_k": k,
                "naive_high_low": round(naive_mean, 2),
                "top_k_minus_bottom_k": round(r["topk_minus_botk"].mean(), 2),
                f"after_{int(eta * 100)}pct_efficiency": round(r["after_efficiency"].mean(), 2),
                # A perfectly flat market has a naive spread of zero, so guard
                # the ratio rather than emitting inf.
                "pct_of_naive": (
                    round(100 * r["after_efficiency"].mean() / naive_mean, 1)
                    if naive_mean > 0
                    else float("nan")
                ),
                "median_realisable": round(r["after_efficiency"].median(), 2),
                "days_negative": int((r["after_efficiency"] <= 0).sum()),
            }
        )
    return pd.DataFrame(out)
 
 
def plot_provider_volumes(df: pd.DataFrame, out_path: Path) -> None:
    """Log-scale volume comparison. Log because the whole question is whether the
    two series differ by a factor of ten or a factor of ten thousand, and a linear
    axis renders the smaller series as a flat line on zero either way."""
    d = df.set_index("start_time_local")
    fig, ax = plt.subplots(figsize=(12, 4))
    for prov, colour in (("apx", "#1b3a5c"), ("n2ex", "#c1440e")):
        col = f"volume_{prov}"
        if col in d.columns:
            ax.plot(d.index, d[col].replace(0, np.nan), lw=0.7, color=colour,
                    label=f"{prov.upper()} reported volume")
    ax.set_yscale("log")
    ax.set_ylabel("Volume (MWh, log scale)")
    ax.set_title("MID reported volume by provider", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
 
 
def main() -> None:
    p = argparse.ArgumentParser(description="Interrogate a fetched MID price series")
    p.add_argument("--file", required=True, help="processed Parquet file")
    p.add_argument("--eta", type=float, default=0.90, help="round-trip efficiency")
    args = p.parse_args()
 
    df = pd.read_parquet(args.file).reset_index()
 
    print("=== provider comparison ===")
    print(compare_providers(df).to_string())
 
    print(f"\n=== realisable daily spread (round-trip efficiency {args.eta:.0%}) ===")
    print(realisable_spread(df, eta=args.eta).to_string(index=False))
 
    fig = FIG_DIR / "provider_volumes.png"
    plot_provider_volumes(df, fig)
    print(f"\nfigure written to {fig}")
 
 
if __name__ == "__main__":
    main()
 