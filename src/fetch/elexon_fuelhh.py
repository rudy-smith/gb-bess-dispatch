"""
Elexon Insights Solution API client - half-hourly generation outturn by fuel type (FUELHH).

FUELHH is the metered average output of each fuel category over each settlement
period, in MW. Its WIND category is transmission-connected, BM-metered wind. That
is the same population the NESO day-ahead wind archive forecasts (its `Capacity`
field is defined as wind "connected to the Transmission Network"), which is what
makes the two comparable. Embedded wind on the distribution network appears in
neither and depresses national demand instead.

API notes (checked against the Insights API specification, not inferred from the
MID client, which uses a different route with different filter semantics):
  - Base URL: https://data.elexon.co.uk/bmrs/api/v1
  - Endpoint: /datasets/FUELHH
  - Filters: settlementDateFrom / settlementDateTo, as plain yyyy-MM-dd
    settlement dates, inclusive. These are settlement-day labels, so a request
    covers whole settlement days in local time with no UTC boundary arithmetic.
    That is a real difference from the MID route, whose `from`/`to` filter on
    UTC instants and clip an hour off each end of a month in BST if built from
    UTC midnights.
  - Optional fuelType (repeatable). Omitted here: every fuel type is fetched,
    because the other categories are needed later for residual-demand features
    and the request count is the same either way.
  - Response rows: dataset, publishTime, startTime, settlementDate,
    settlementPeriod, fuelType, generation.
  - No API key.

Outturn is a target, never a feature at the same horizon. Where it is later used
as a lagged feature, note that this archive serves the latest revision of each
value, not the value as first published, so a lagged outturn feature carries a
small revision look-ahead. Recorded in the documentation as a limitation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.fetch.elexon import BASE_URL, _session
from src.prep.calendar_gb import expected_periods_in_day, settlement_period_grid

log = logging.getLogger(__name__)

FUELHH_ENDPOINT = "/datasets/FUELHH"
DEFAULT_RAW_DIR = Path("data/raw/fuelhh")

# Settlement days per request. The MID route rejects windows wider than seven
# days; whether FUELHH has the same limit is unverified, so the same conservative
# width is used. The cost of being wrong in this direction is a few extra requests.
MAX_QUERY_DAYS = 7

REQUIRED_FIELDS = ("publishTime", "settlementDate", "settlementPeriod", "fuelType", "generation")


def date_windows(month_start: pd.Timestamp, days: int = MAX_QUERY_DAYS) -> list[tuple[str, str]]:
    """Non-overlapping inclusive (from, to) settlement-date pairs covering one month.

    Because both ends of a settlement-date filter are inclusive and the unit is a
    whole day, windows are [d, d+days-1], [d+days, ...]: adjacent windows share no
    day, so no boundary duplicates arise. The MID client needs a de-duplication
    step for its shared boundary instant; this one does not, although the caller
    still checks for duplicates rather than relying on the arithmetic.
    """
    first = pd.Timestamp(month_start).normalize().replace(day=1)
    last = first + pd.offsets.MonthEnd(0)
    windows = []
    cursor = first
    while cursor <= last:
        stop = min(cursor + pd.Timedelta(days=days - 1), last)
        windows.append((f"{cursor:%Y-%m-%d}", f"{stop:%Y-%m-%d}"))
        cursor = stop + pd.Timedelta(days=1)
    return windows


def _fetch_window(session, date_from: str, date_to: str) -> pd.DataFrame:
    """One request. The response body is surfaced on failure, as in the MID client."""
    params = {"settlementDateFrom": date_from, "settlementDateTo": date_to, "format": "json"}
    resp = session.get(BASE_URL + FUELHH_ENDPOINT, params=params, timeout=60)
    if not resp.ok:
        raise RuntimeError(
            f"Elexon returned HTTP {resp.status_code} for FUELHH {date_from} .. {date_to}\n"
            f"URL:  {resp.url}\n"
            f"Body: {resp.text[:1000]}"
        )
    payload = resp.json()
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    return pd.DataFrame(rows)


def fetch_month_raw(session, month_start: pd.Timestamp) -> pd.DataFrame:
    """All FUELHH rows for one calendar month of settlement days, long format."""
    frames = []
    for date_from, date_to in date_windows(month_start):
        log.info("GET %s  %s .. %s", FUELHH_ENDPOINT, date_from, date_to)
        frames.append(_fetch_window(session, date_from, date_to))

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"Elexon returned no FUELHH rows for {month_start:%Y-%m}")

    missing = set(REQUIRED_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(
            f"FUELHH response missing expected fields {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}. The API schema may have changed."
        )
    keep = list(REQUIRED_FIELDS) + (["startTime"] if "startTime" in df.columns else [])
    return df[keep].reset_index(drop=True)


def to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """Long (one row per fuel per period) -> one row per settlement period.

    Revisions: if the same (date, period, fuel) appears more than once, the row
    with the latest publishTime is kept. For outturn, the latest revision is the
    best estimate of what physically happened, which is what a target should be.
    The count is logged so a revision-heavy month is visible rather than absorbed.

    Columns come out as gen_<fuel>_mw, e.g. gen_wind_mw, gen_ccgt_mw.
    """
    df = raw.copy()
    df["settlement_date"] = pd.to_datetime(df["settlementDate"]).dt.normalize()
    df["settlement_period"] = df["settlementPeriod"].astype(int)
    df["fuel"] = df["fuelType"].astype(str).str.strip().str.lower()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["publish_time"] = pd.to_datetime(df["publishTime"], utc=True, errors="coerce")

    key = ["settlement_date", "settlement_period", "fuel"]
    df = df.sort_values(key + ["publish_time"])
    revised = int(df.duplicated(key).sum())
    if revised:
        log.info("FUELHH: %s superseded revisions dropped, latest publishTime kept", revised)
    df = df.drop_duplicates(key, keep="last")

    wide = df.pivot(index=["settlement_date", "settlement_period"], columns="fuel",
                    values="generation")
    wide.columns = [f"gen_{c}_mw" for c in wide.columns]

    # startTime is Elexon's own UTC start of the period. Kept only as a cross-check
    # on this project's calendar, never as a key: it is wrong by 24 hours on every
    # SP48 from 2022-01-01 to 2022-07-14 (see quarantine_out_of_range).
    if "startTime" in df.columns:
        start = df.drop_duplicates(["settlement_date", "settlement_period"]).set_index(
            ["settlement_date", "settlement_period"]
        )["startTime"]
        wide["elexon_start_utc"] = pd.to_datetime(start, utc=True)

    return wide.reset_index()


# Most out-of-range rows tolerated before the fetch is refused outright. A handful
# of stray labels is a source quirk; more than this is a systematic fault, such as
# a changed period convention, and must stop the pipeline rather than be patched.
MAX_OUT_OF_RANGE = 10


def quarantine_out_of_range(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows whose settlement period cannot exist on their settlement day.

    A dropped row leaves its period NaN on the grid, where the quality report
    counts it and the cleaning policy handles it, so the loss is visible. The
    alternative, reassigning the value to the period it "probably" is, plants a
    value that no later check can detect.

    Elexon's `startTime` is deliberately not used to reassign rows, because in
    this archive it is the less reliable field. From 2022-01-01 to 2022-07-14,
    every SP48 row carries a startTime exactly 24 hours early (the previous
    day's SP48), while its label matches the calendar. The one out-of-range row,
    2022-03-27 "SP48" on a 46-period day, carries that same early startTime, so
    it is most likely that day's final period numbered as if the day had 48.
    Neither field identifies it with confidence, so it is dropped and
    2022-03-27 SP46 stays missing: one half hour, outside the study period.

    Returns (kept, quarantined). Raises above MAX_OUT_OF_RANGE stray rows,
    which would indicate a systematic change of period convention.
    """
    max_sp = wide["settlement_date"].map(
        {d: expected_periods_in_day(d) for d in wide["settlement_date"].unique()}
    )
    bad = (wide["settlement_period"] > max_sp) | (wide["settlement_period"] < 1)
    quarantined = wide.loc[bad]
    if quarantined.empty:
        return wide, quarantined

    if len(quarantined) > MAX_OUT_OF_RANGE:
        raise ValueError(
            f"{len(quarantined)} FUELHH rows have a settlement period outside the valid "
            f"range for their day, above the tolerance of {MAX_OUT_OF_RANGE}. First rows:\n"
            f"{quarantined[['settlement_date', 'settlement_period']].head()}"
        )

    log.warning(
        "FUELHH: quarantined %s row(s) with a settlement period that cannot exist on "
        "their day; any period they displaced stays NaN:\n%s",
        len(quarantined),
        quarantined.dropna(axis=1, how="all").to_string(),
    )
    return wide.loc[~bad], quarantined


def validate_and_grid(wide: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Check labels, then left-join onto the authoritative settlement grid.

    Same order of checks as the MID client, for the same reason: an out-of-range
    period must be caught while it is still a label, because once converted to a
    timestamp SP50 on a 48-period day becomes a valid half hour of the next day.
    Unlike the MID client, a few stray labels are quarantined rather than fatal,
    because FUELHH does serve them; see quarantine_out_of_range.
    """
    wide, _ = quarantine_out_of_range(wide)
    if wide.duplicated(["settlement_date", "settlement_period"]).any():
        raise ValueError("duplicate settlement periods in FUELHH after pivoting")

    grid = settlement_period_grid(start_date, end_date).reset_index()
    grid["settlement_date"] = pd.to_datetime(grid["settlement_date"]).dt.normalize()

    out = grid.merge(
        wide, on=["settlement_date", "settlement_period"], how="left", validate="one_to_one"
    )
    if len(out) != len(grid):
        raise ValueError(f"join changed row count: grid={len(grid)}, result={len(out)}")
    return out.set_index("start_time_utc")


def fetch_fuelhh(
    start_date: str,
    end_date: str,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    force: bool = False,
    session=None,
) -> pd.DataFrame:
    """FUELHH for a settlement-date range, cached one raw Parquet file per month.

    The cache is never invalidated automatically, for the same reason as the MID
    cache: a backtest has to run against a frozen snapshot, and a restatement by
    Elexon should change results only when a refetch is requested deliberately.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = session or _session()

    months = pd.date_range(
        pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS"
    )
    frames = []
    for m in months:
        path = raw_dir / f"fuelhh_{m:%Y-%m}.parquet"
        if path.exists() and not force:
            frames.append(pd.read_parquet(path))
            continue
        raw = fetch_month_raw(session, m)
        raw.to_parquet(path, index=False)
        frames.append(raw)

    wide = to_wide(pd.concat(frames, ignore_index=True))
    in_range = (wide["settlement_date"] >= pd.Timestamp(start_date)) & (
        wide["settlement_date"] <= pd.Timestamp(end_date)
    )
    return validate_and_grid(wide.loc[in_range], start_date, end_date)


def fuelhh_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing periods per fuel column, and the Elexon/project calendar agreement."""
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    rows = [{"check": f"{c}_missing", "value": int(df[c].isna().sum())} for c in gen_cols]
    rows.insert(0, {"check": "periods_expected", "value": len(df)})
    if "elexon_start_utc" in df.columns:
        present = df["elexon_start_utc"].notna()
        agree = (df.loc[present, "elexon_start_utc"] == df.index[present]).sum()
        rows.append({"check": "elexon_start_matches_calendar", "value": f"{agree} of {present.sum()}"})
    return pd.DataFrame(rows)
