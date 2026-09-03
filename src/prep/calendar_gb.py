"""
GB electricity settlement-period calendar.
 
A GB Settlement Day runs from 00:00 to 00:00 *local* (Europe/London) time and is
divided into half-hour Settlement Periods numbered from 1. Because the UK observes
British Summer Time, a settlement day contains:
 
    48 periods on a normal day
    46 periods on the spring-forward day  (last Sunday in March, 23 hours long)
    50 periods on the autumn-back day     (last Sunday in October, 25 hours long)
 
Every function here treats a *UTC* timestamp as the canonical identity of a
half-hour, and (settlement_date, settlement_period) as a derived label. UTC is
canonical because it is monotonic, unique and gap-free by construction: there is
no ambiguous or nonexistent UTC time. Local time is derived for feature
engineering and plotting only, and is never used as an index.
 
The mapping is a bijection, so nothing is lost by preferring one over the other.
"""
 
from __future__ import annotations
 
import pandas as pd
 
LONDON = "Europe/London"
UTC = "UTC"
HALF_HOUR = "30min"
 
 
def settlement_period_grid(start_date: str, end_date: str) -> pd.DataFrame:
    """Build the complete, correct grid of settlement periods for a date range.
 
    This is the reference index. Any dataset we fetch gets reindexed onto this,
    which turns "did the API give me everything?" into a one-line NaN check.
 
    Parameters
    ----------
    start_date, end_date : str
        Settlement dates in 'YYYY-MM-DD' form. Both inclusive. These are LOCAL
        settlement dates, not UTC dates.
 
    Returns
    -------
    DataFrame indexed by tz-aware UTC timestamp (the period START), with columns:
        settlement_date    : date  - the GB settlement day (local date)
        settlement_period  : int   - 1..46/48/50
        start_time_local   : tz-aware Europe/London timestamp
    """
    # Anchor on LOCAL midnight at both ends, then convert to UTC. Doing it this
    # way round is what makes clock changes come out right: local midnight to
    # local midnight is 23, 24 or 25 real hours depending on the day, and
    # converting those two anchors to UTC captures that automatically.
    start_local = pd.Timestamp(start_date, tz=LONDON)
    end_local = _next_local_midnight(end_date)
 
    # Generate in UTC. A UTC range can never produce a duplicate or a nonexistent
    # timestamp, so no DST handling is needed here at all.
    idx_utc = pd.date_range(
        start=start_local.tz_convert(UTC),
        end=end_local.tz_convert(UTC),
        freq=HALF_HOUR,
        inclusive="left",  # period start times; the final midnight belongs to the next day
    )
 
    local = idx_utc.tz_convert(LONDON)
 
    grid = pd.DataFrame(
        {
            "settlement_date": local.date,
            "start_time_local": local,
        },
        index=idx_utc,
    )
    grid.index.name = "start_time_utc"
 
    # Settlement period = 1-based rank of the half-hour within its local day.
    # Derived by counting rather than by arithmetic on the hour, because
    # arithmetic breaks on the two clock-change days and counting does not.
    grid["settlement_period"] = grid.groupby("settlement_date").cumcount() + 1
 
    return grid[["settlement_date", "settlement_period", "start_time_local"]]
 
 
def _next_local_midnight(settlement_date) -> pd.Timestamp:
    """Local midnight at the START of the following settlement day.
 
    Deliberately NOT `pd.Timestamp(d, tz=LONDON) + pd.Timedelta(days=1)`.
    Adding a Timedelta to a tz-aware timestamp advances 24 hours of *absolute*
    time, which on a clock-change day lands on 23:00 or 01:00 local, not
    midnight. Incrementing the naive calendar date and localising afterwards is
    the only version that is right on all three day lengths.
    """
    d = pd.Timestamp(settlement_date).normalize()  # tz-naive calendar date
    return pd.Timestamp(d + pd.Timedelta(days=1)).tz_localize(LONDON)
 
 
def expected_periods_in_day(settlement_date) -> int:
    """Number of settlement periods in a given GB settlement day (46, 48 or 50).
 
    Computed from the real elapsed time between consecutive local midnights,
    not from a hardcoded list of clock-change dates, so it stays correct if the
    UK ever changes its DST rules.
    """
    midnight = pd.Timestamp(pd.Timestamp(settlement_date).date(), tz=LONDON)
    hours = (_next_local_midnight(settlement_date) - midnight).total_seconds() / 3600.0
    return round(hours * 2)
 
 
def to_utc(settlement_date, settlement_period: int) -> pd.Timestamp:
    """(settlement_date, settlement_period) -> UTC start time of that period.
 
    Walks forward from local midnight in real half-hour steps. Adding a
    Timedelta to a tz-aware timestamp advances *absolute* time, which is what we
    want: period 3 is always 60 real minutes after period 1, even on a day where
    the wall clock jumps.
    """
    n = expected_periods_in_day(settlement_date)
    if not 1 <= settlement_period <= n:
        raise ValueError(
            f"settlement period {settlement_period} out of range for "
            f"{settlement_date}, which has {n} periods"
        )
    midnight_utc = pd.Timestamp(settlement_date, tz=LONDON).tz_convert(UTC)
    return midnight_utc + pd.Timedelta(minutes=30 * (settlement_period - 1))
 
 
def from_utc(ts: pd.Timestamp) -> tuple:
    """UTC timestamp -> (settlement_date, settlement_period). Inverse of to_utc."""
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        raise ValueError("timestamp must be tz-aware; naive timestamps are ambiguous")
    local = ts.tz_convert(LONDON)
    d = local.date()
    midnight_utc = pd.Timestamp(d, tz=LONDON).tz_convert(UTC)
    elapsed_minutes = (ts.tz_convert(UTC) - midnight_utc).total_seconds() / 60.0
    if elapsed_minutes % 30 != 0:
        raise ValueError(f"{ts} is not aligned to a half-hour boundary")
    return d, int(elapsed_minutes // 30) + 1
 
 
def attach_utc_index(
    df: pd.DataFrame,
    date_col: str = "settlement_date",
    period_col: str = "settlement_period",
) -> pd.DataFrame:
    """Vectorised (date, period) -> UTC index for a whole DataFrame.
 
    Used on API responses, which arrive labelled by settlement date and period.
    Vectorised rather than df.apply(to_utc, axis=1) because a two-year fetch is
    ~35,000 rows and a row-wise Python loop over tz-aware Timestamps is roughly
    two orders of magnitude slower.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
 
    # Localise local-midnight for every row at once, then step forward.
    midnights_utc = (
        dates.dt.tz_localize(LONDON, ambiguous=False, nonexistent="raise")
        .dt.tz_convert(UTC)
    )
    offsets = pd.to_timedelta((out[period_col].astype(int) - 1) * 30, unit="m")
 
    out["start_time_utc"] = midnights_utc + offsets
    out["start_time_local"] = out["start_time_utc"].dt.tz_convert(LONDON)
    return out.set_index("start_time_utc").sort_index()
 