"""
Fetch half-hourly generation outturn by fuel type (FUELHH) and report its quality.

    python -m scripts.fetch_fuelhh --start 2022-01-01 --end 2024-12-31

2022 is included because the wind contamination test needs the June to October
2022 window as a seasonal control. Results are cached per month under
data/raw/fuelhh/; --force refetches. The wide frame is written to
data/processed/fuelhh.parquet for the audit and for later feature work.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.fetch.elexon_fuelhh import fetch_fuelhh, fuelhh_quality_report

PROCESSED = Path("data/processed/fuelhh.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = fetch_fuelhh(args.start, args.end, force=args.force)
    print(fuelhh_quality_report(df).to_string(index=False))

    wind = df["gen_wind_mw"]
    print(
        f"\nwind outturn: mean {wind.mean():,.0f} MW, max {wind.max():,.0f} MW, "
        f"min {wind.min():,.0f} MW"
    )

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED)
    print(f"wrote {len(df):,} rows to {PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
