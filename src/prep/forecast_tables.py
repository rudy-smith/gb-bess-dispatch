"""
Canonical forecast tables: one row per (target half hour, publication instant).

The raw NESO archives use different column names, different timestamp
conventions and, for demand, ship the outturn in the same file as the forecast.
This module turns each into the same long shape, so the SQL layer can treat
every forecast source identically:

    target_time          UTC start of the half hour being forecast
    settlement_date      settlement-day label of that half hour
    settlement_period    settlement-period label of that half hour
    published_at         UTC instant the forecast is treated as available
    published_at_source  "archive" or "schedule" (see neso.resolve_published_at)
    published_at_suspect True where the archive's own timestamp failed validation
    <value columns>

target_time is derived from the settlement labels through this project's
calendar, not taken from either archive's own timestamp column. The two archives
disagree about what their timestamps mean (period start in UTC for wind, period
end in local time for demand), and the labels are the one convention they share.

Demand outturn is written to a separate table. Keeping it out of the forecast
table physically, rather than excluding it by column name at query time, means a
feature query cannot pick it up by accident: it is not in the table it reads.
"""

from __future__ import annotations

import pandas as pd

from src.fetch.neso import (
    DEMAND_OUTTURN_COLUMNS,
    RESOURCES,
    day_start_utc,
    normalise_demand,
    normalise_wind,
    resolve_published_at,
)
from src.prep.calendar_gb import attach_utc_index, expected_periods_in_day

KEYS = ["settlement_date", "settlement_period"]
PUBLICATION = ["published_at", "published_at_source", "published_at_suspect"]


def _restrict(raw: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(raw[date_col])
    return raw.loc[(dates >= start) & (dates <= end)].copy()


def _add_target_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the labels, then derive the UTC start of each half hour.

    Validation comes first for the reason given in calendar_gb: once converted,
    an impossible label such as SP50 on a 48-period day becomes a real half hour
    of the next day and can no longer be recognised as wrong.
    """
    max_sp = frame["settlement_date"].map(
        {d: expected_periods_in_day(d) for d in frame["settlement_date"].unique()}
    )
    bad = (frame["settlement_period"] < 1) | (frame["settlement_period"] > max_sp)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} forecast rows have an impossible settlement period:\n"
            f"{frame.loc[bad, KEYS].head()}"
        )
    out = attach_utc_index(frame).reset_index()
    return out.rename(columns={"start_time_utc": "target_time"}).drop(
        columns="start_time_local"
    )


def _check_unique(frame: pd.DataFrame, name: str) -> None:
    """One value per (target, publication instant), or an as-of join is ambiguous.

    With two rows sharing a target and a publication instant, "the latest forecast
    published before the decision" has two answers, and the join would return
    whichever the engine met first.
    """
    dupes = frame.duplicated(["target_time", "published_at"])
    if dupes.any():
        raise ValueError(
            f"{name}: {int(dupes.sum())} rows share a target half hour and a publication "
            f"instant:\n{frame.loc[dupes, KEYS + ['published_at']].head()}"
        )


def build_wind_table(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Canonical wind forecast table from the raw NESO wind archive."""
    resource = RESOURCES["wind_da"]
    raw = _restrict(raw, "Date", start, end)
    resolved = resolve_published_at(raw, resource, "Date", day_start_utc(raw["Date"]))

    # normalise_wind re-indexes by its own timestamp column, so the publication
    # columns are carried across on the settlement labels, strictly one to one.
    publication = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(resolved["Date"]).dt.normalize(),
            "settlement_period": resolved["Settlement_period"].astype(int),
            **{c: resolved[c].to_numpy() for c in PUBLICATION},
        }
    )
    values = normalise_wind(raw).reset_index(drop=True)
    merged = values.merge(publication, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(values):
        raise ValueError("wind publication columns did not align one to one with the values")

    out = _add_target_time(merged)
    cols = ["target_time", *KEYS, *PUBLICATION, "wind_forecast_mw", "wind_capacity_mw",
            "wind_forecast_share"]
    out = out[[c for c in cols if c in out.columns]].sort_values("target_time")
    _check_unique(out, "wind")
    return out.reset_index(drop=True)


def build_demand_tables(
    raw: pd.DataFrame, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecast table, outturn table) from the raw NESO demand performance archive."""
    resource = RESOURCES["demand_da_hh"]
    raw = _restrict(raw, "Date", start, end)
    resolved = resolve_published_at(raw, resource, "Date", day_start_utc(raw["Date"]))

    # normalise_demand keeps the raw index, so publication columns align by index.
    values = normalise_demand(raw)
    for c in PUBLICATION:
        values[c] = resolved[c]
    out = _add_target_time(values)

    forecast = out[["target_time", *KEYS, *PUBLICATION, "demand_forecast_mw"]]
    forecast = forecast.sort_values("target_time").reset_index(drop=True)
    _check_unique(forecast, "demand")

    outturn_cols = [c for c in DEMAND_OUTTURN_COLUMNS if c in out.columns]
    outturn = out[["target_time", *KEYS, *outturn_cols]].sort_values("target_time")
    return forecast, outturn.reset_index(drop=True)
