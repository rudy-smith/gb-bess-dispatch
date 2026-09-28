"""
Build the point-in-time frame and check it obeys its own rule.

    python -m scripts.build_point_in_time
    python -m scripts.build_point_in_time --decision-time 09:20

Needs data/processed/prices_full.parquet and the tables written by
scripts.build_forecast_tables. Writes data/processed/point_in_time.parquet.

The checks do not trust the SQL. They re-verify from the output that every
attached forecast was published before its decision, and count, with a
separate plain join, how many forecasts existed for each half hour but were
published too late to use. That second number is what the as-of join is
protecting against; if it is zero, the join changed nothing on this data, and
that is worth knowing rather than assuming.
"""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

from src.db.duck import connect, query, query_file, register_decisions

OUT = Path("data/processed/point_in_time.parquet")

TOO_LATE = """
SELECT count(*) AS forecasts_published_too_late,
       count(DISTINCT f.target_time) AS half_hours_affected
FROM decisions d
JOIN {table} f ON f.target_time = d.target_time
WHERE f.published_at >= d.decision_time
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--decision-time", default="11:00", help="London clock time on D-1")
    args = parser.parse_args()
    hour, minute = (int(x) for x in args.decision_time.split(":"))

    con = connect(required=("prices", "wind_forecasts", "demand_forecasts"))
    register_decisions(con, args.start, args.end, time(hour, minute))
    frame = query_file(con, "point_in_time")

    n_expected = query(con, "SELECT count(*) AS n FROM decisions")["n"].item()
    print(f"rows {len(frame):,} (decision half hours {n_expected:,})")
    if len(frame) != n_expected:
        raise SystemExit("row count changed in the join: a key is duplicated somewhere")

    for source in ("wind", "demand"):
        published = frame[f"{source}_published_at"]
        attached = published.notna()
        violations = int((published[attached] >= frame.loc[attached, "decision_time"]).sum())
        margin = (frame.loc[attached, "decision_time"] - published[attached]).dt.total_seconds() / 3600
        late = query(con, TOO_LATE.format(table=f"{source}_forecasts")).iloc[0]
        print(
            f"\n{source}:\n"
            f"  attached                    {int(attached.sum()):,} of {len(frame):,}\n"
            f"  published after decision    {violations}   (must be 0)\n"
            f"  margin before decision      {margin.min():.2f} .. {margin.max():.2f} h "
            f"(median {margin.median():.2f})\n"
            f"  excluded as too late        {int(late.forecasts_published_too_late):,} forecasts "
            f"across {int(late.half_hours_affected):,} half hours\n"
            f"  suspect publication time    {int(frame[f'{source}_suspect'].fillna(False).sum()):,}"
        )
        if violations:
            raise SystemExit(f"{source}: forecast used before it was published")

    print(f"\nprice missing                 {int(frame['price_apx'].isna().sum()):,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
