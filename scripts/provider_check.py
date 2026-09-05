"""Characterise the two Market Index Data Provider price series."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.file))
    apx, n2ex = df["price_apx"], df["price_n2ex"]

    print(f"periods                {len(df)}")
    print(f"apx  distinct values   {apx.nunique()}")
    print(f"n2ex distinct values   {n2ex.nunique()}")
    print(f"\nn2ex ten most frequent values:\n{n2ex.value_counts().head(10)}")
    print(f"\nn2ex describe:\n{n2ex.describe()}")

    both = df[["price_apx", "price_n2ex"]].dropna()
    diff = both["price_n2ex"] - both["price_apx"]
    print(f"\ncorrelation            {both.corr().iloc[0, 1]:.4f}")
    print(f"mean   n2ex - apx      {diff.mean():.2f}")
    print(f"median n2ex - apx      {diff.median():.2f}")
    print(f"periods differing >£5  {(diff.abs() > 5).sum()} of {len(both)}")

    # Longest constant run in each series, as a blunt staleness measure.
    for name, s in (("apx", apx), ("n2ex", n2ex)):
        run_id = s.ne(s.shift()).cumsum()
        print(f"{name:<5} longest constant run  {s.groupby(run_id).size().max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

