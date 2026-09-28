"""
Build the canonical forecast tables the SQL layer reads.

    python -m scripts.build_forecast_tables
    python -m scripts.build_forecast_tables --start 2023-01-01 --end 2024-12-31

Reads the cached NESO archives in data/raw/neso/ and writes:

    data/processed/wind_forecasts.parquet
    data/processed/demand_forecasts.parquet
    data/processed/demand_outturn.parquet    (kept apart from the forecast on purpose)

The default range matches the price data, 2023-2024. Earlier years are not
needed for features, and the demand archive is not clean before then: on the
2022-10-30 clock change it repeats settlement periods 2 and 3 with the same
publication instant, which the uniqueness check rejects. The wind audit reads
its own files and keeps its 2022 control year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fetch.neso import RESOURCES, cache_path
from src.prep.forecast_tables import build_demand_tables, build_wind_table

OUT = Path("data/processed")


def summarise(name: str, frame: pd.DataFrame) -> None:
    lead = (
        frame.groupby("settlement_date")["target_time"].transform("min") - frame["published_at"]
    ).dt.total_seconds() / 3600
    print(
        f"{name:<18} rows {len(frame):>7,}  "
        f"{frame['target_time'].min():%Y-%m-%d} .. {frame['target_time'].max():%Y-%m-%d}  "
        f"suspect {int(frame['published_at_suspect'].sum()):>5,}  "
        f"lead to day start {lead.min():.2f}..{lead.max():.2f} h"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    wind_raw = pd.read_parquet(cache_path(args.raw_root, RESOURCES["wind_da"].name))
    demand_raw = pd.read_parquet(cache_path(args.raw_root, RESOURCES["demand_da_hh"].name))

    wind = build_wind_table(wind_raw, args.start, args.end)
    demand, outturn = build_demand_tables(demand_raw, args.start, args.end)

    OUT.mkdir(parents=True, exist_ok=True)
    wind.to_parquet(OUT / "wind_forecasts.parquet", index=False)
    demand.to_parquet(OUT / "demand_forecasts.parquet", index=False)
    outturn.to_parquet(OUT / "demand_outturn.parquet", index=False)

    summarise("wind_forecasts", wind)
    summarise("demand_forecasts", demand)
    print(f"demand_outturn     rows {len(outturn):>7,}  columns {list(outturn.columns[3:])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
