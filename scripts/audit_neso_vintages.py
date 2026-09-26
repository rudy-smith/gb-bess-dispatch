"""Audit the publication timestamps in a cached NESO archive.
 
Three questions, in order of how much they matter to the backtest:
 
  1. How many rows record a publication instant that does not precede the
     settlement day they forecast? Such a row is not a forecast. If they are
     confined to the start of the archive they are backfill and harmless to a
     later study period; if they are scattered through it, the feature cannot
     be used as it stands.
 
  2. Is the naive timestamp column UTC or local clock time? NESO publishes on a
     clock-time schedule, so the correct reading is the one whose publication
     time is stable across the year. The wrong reading shows a one-hour jump at
     each clock change, because it is carrying the daylight-saving offset that
     the other reading absorbs. This is decidable from the data rather than by
     convention.
 
  3. Does the archive cover the study period at the expected resolution, with
     one vintage per settlement period?
 
Usage:
 
    python -m scripts.audit_neso_vintages --resource wind_da
    python -m scripts.audit_neso_vintages --resource wind_da --start 2023-01-01 --end 2024-12-31
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
 
from src.fetch.neso import RESOURCES, cache_path, day_start_utc
from src.prep.calendar_gb import LONDON, UTC
 
# Columns holding the settlement date and the naive publication instant, by
# resource. Written out rather than derived, because the two archives use
# different names for the same thing and a wrong guess would audit nothing.
DATE_COLUMN = {"wind_da": "Date", "demand_da_hh": "Date"}
 
 
def load(resource_name: str, raw_root: Path) -> pd.DataFrame:
    path = cache_path(raw_root, resource_name)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached archive at {path}; fetch it with fetch_and_cache first"
        )
    return pd.read_parquet(path)
 
 
def timeliness(frame: pd.DataFrame, resource_name: str) -> pd.DataFrame:
    """Count rows per year whose publication does not precede their day."""
    resource = RESOURCES[resource_name]
    dates = pd.to_datetime(frame[DATE_COLUMN[resource_name]])
    starts = day_start_utc(dates)
    recorded = pd.to_datetime(
        frame[resource.timestamp_column], errors="coerce"
    ).dt.tz_localize(resource.timestamp_timezone)
    if resource.timestamp_timezone != "UTC":
        recorded = recorded.dt.tz_convert(UTC)
 
    late = recorded.isna() | (recorded >= starts)
    lead_hours = (starts - recorded).dt.total_seconds() / 3600.0
 
    return pd.DataFrame(
        {
            "rows": 1,
            "late_or_missing": late.astype(int),
            "lead_hours": lead_hours,
        }
    ).groupby(dates.dt.year).agg(
        rows=("rows", "sum"),
        late_or_missing=("late_or_missing", "sum"),
        median_lead_hours=("lead_hours", "median"),
        min_lead_hours=("lead_hours", "min"),
    )
 
 
def timezone_evidence(frame: pd.DataFrame, resource_name: str) -> str:
    """Compare the two readings of the naive timestamp across the seasons.
 
    NESO publishes at a fixed clock time. Under the correct reading the
    publication clock time is the same in winter and summer; under the wrong
    one it shifts by an hour, because the daylight-saving offset has been
    applied to the wrong side of the conversion.
    """
    resource = RESOURCES[resource_name]
    naive = pd.to_datetime(frame[resource.timestamp_column], errors="coerce")
    naive = naive.dropna()
    if naive.empty:
        return "no parseable timestamps"
 
    as_utc_local = naive.dt.tz_localize(UTC).dt.tz_convert(LONDON)
    as_local = naive.dt.tz_localize(LONDON, ambiguous="NaT", nonexistent="NaT")
 
    lines = ["", "timezone evidence (publication clock time, London):"]
    for label, series in (("read as UTC", as_utc_local), ("read as local", as_local)):
        valid = series.dropna()
        if valid.empty:
            lines.append(f"  {label:<16} unparseable")
            continue
        clock = valid.dt.hour + valid.dt.minute / 60.0
        winter = clock[valid.dt.month.isin([12, 1, 2])]
        summer = clock[valid.dt.month.isin([6, 7, 8])]
        shift = summer.median() - winter.median() if len(winter) and len(summer) else float("nan")
        lines.append(
            f"  {label:<16} winter median {winter.median():.2f}h  "
            f"summer median {summer.median():.2f}h  shift {shift:+.2f}h"
        )
 
    lines.append(
        "  The reading with a shift near zero is the correct one. A shift near "
        "one hour means that reading is carrying the daylight-saving offset."
    )
    return "\n".join(lines)
 
 
def coverage(frame: pd.DataFrame, resource_name: str, start: str, end: str) -> str:
    dates = pd.to_datetime(frame[DATE_COLUMN[resource_name]])
    window = frame[(dates >= start) & (dates <= end)]
    window_dates = pd.to_datetime(window[DATE_COLUMN[resource_name]])
    per_day = window.groupby(window_dates).size()
 
    expected_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    odd = per_day[~per_day.isin([46, 48, 50])]
 
    lines = [
        "",
        f"coverage {start} to {end}:",
        f"  rows                    {len(window):,}",
        f"  days present            {len(per_day)} of {expected_days}",
        f"  periods/day min..max    {per_day.min()}..{per_day.max()}",
        f"  days not 46/48/50       {len(odd)}",
    ]
    if len(odd):
        lines.append(f"  {odd.head(10).to_dict()}")
    return "\n".join(lines)
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", choices=sorted(RESOURCES), required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
 
    frame = load(args.resource, args.raw_root)
    resource = RESOURCES[args.resource]
 
    print(f"=== {args.resource}: {resource.description} ===")
    print(f"rows                      {len(frame):,}")
    print(f"timestamp column          {resource.timestamp_column}")
    print(f"read as                   {resource.timestamp_timezone}")
 
    print("\npublication timeliness by target year:")
    print(timeliness(frame, args.resource).to_string())
 
    print(timezone_evidence(frame, args.resource))
    print(coverage(frame, args.resource, args.start, args.end))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())

 