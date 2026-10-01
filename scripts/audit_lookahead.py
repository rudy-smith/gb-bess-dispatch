"""
Look-ahead audit of the feature frame.

    python -m scripts.audit_lookahead

Recomputes every feature independently of the SQL, compares the two, and
reports for each feature the smallest margin between the moment its inputs
became public and the decision it informs. Exits with status 1 if any feature
disagrees with its recomputation or has an input that was not public in time.

Run after scripts.build_features and before any backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.prep.feature_audit import audit
from src.prep.features import PUBLICATION_LAG_MINUTES

PROCESSED = Path("data/processed")


def main() -> int:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    prices = pd.read_parquet(PROCESSED / "prices_full.parquet")
    pit = pd.read_parquet(PROCESSED / "point_in_time.parquet")
    generation = pd.read_parquet(PROCESSED / "fuelhh.parquet")

    result = audit(features, prices, pit, generation, PUBLICATION_LAG_MINUTES)

    print(f"look-ahead audit: {len(features):,} half hours, "
          f"publication lag {PUBLICATION_LAG_MINUTES} min, decision 11:00 London on D-1\n")
    shown = result.assign(min_margin_h=result["min_margin_h"].round(2))
    with pd.option_context("display.width", 140, "display.max_colwidth", 42):
        print(shown.to_string())

    failed = result[result["verdict"] != "PASS"]
    if len(failed):
        print(f"\nFAIL: {list(failed.index)}")
        return 1
    print(f"\nPASS: all {len(result)} features agree with an independent recomputation, "
          "and every input was public before its decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
