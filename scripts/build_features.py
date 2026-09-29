"""
Build the feature frame from the point-in-time frame.

    python -m scripts.build_features

Needs data/processed/point_in_time.parquet (scripts.build_point_in_time),
prices_full.parquet and fuelhh.parquet. Writes data/processed/features.parquet
and prints the share of each feature that is missing, so a feature that is
silently NULL for a whole season is visible before a model trains on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.db.duck import connect, query, query_file, register_calendar
from src.prep.features import FEATURES, PUBLICATION_LAG_MINUTES, check_feature_frame

OUT = Path("data/processed/features.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lag-minutes", type=int, default=PUBLICATION_LAG_MINUTES)
    args = parser.parse_args()

    con = connect(required=("point_in_time", "prices", "generation"))
    span = query(con, "SELECT min(target_time) AS a, max(target_time) AS b FROM point_in_time")
    first = (span["a"].iloc[0].tz_convert("Europe/London") - pd.Timedelta(days=35)).date()
    last = span["b"].iloc[0].tz_convert("Europe/London").date()
    register_calendar(con, str(first), str(last))

    frame = query_file(con, "features", {"lag_minutes": args.lag_minutes})
    n_pit = query(con, "SELECT count(*) AS n FROM point_in_time")["n"].item()
    check_feature_frame(frame, n_pit)

    print(f"rows {len(frame):,}   features {len(FEATURES)}   publication lag {args.lag_minutes} min")
    print(f"target_price missing {int(frame['target_price'].isna().sum()):,}\n")
    missing = frame[list(FEATURES)].isna().mean().mul(100).round(2)
    print(f"{'feature':<28}{'missing %':>10}   first value")
    for c in FEATURES:
        first = frame.loc[frame[c].notna(), "settlement_date"].min()
        shown = "never" if pd.isna(first) else f"{pd.Timestamp(first):%Y-%m-%d}"
        print(f"{c:<28}{missing[c]:>10.2f}   {shown}")

    # A feature that is missing everywhere is a broken join, not a data gap.
    empty = [c for c in FEATURES if missing[c] == 100]
    if empty:
        raise SystemExit(f"\nfeatures missing in every row, not written: {empty}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
