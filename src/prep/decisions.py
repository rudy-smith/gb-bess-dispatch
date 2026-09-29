"""
Decision times: the instant at which each delivery half hour's schedule is fixed.

The backtest uses a day-ahead framing. Every half hour of delivery day D is
scheduled once, at a fixed clock time on D-1, and only information published
strictly before that instant may inform the schedule. That instant is the
single reference point for every look-ahead check in the project.

The default, 11:00 London time on D-1, follows NESO's publication times rather
than an auction timetable, because the prices traded against are a prompt index,
not an auction clear. It is after the wind forecast (archive stamps cluster near
07:23; the documented window ends at 09:15) and after the demand forecast (13.25
hours before the delivery day starts, i.e. 10:45 on the conservative UTC reading
of its timestamp). A decision earlier than 10:45 would lose the demand forecast
on that reading; scripts.build_point_in_time reports the margin for each source
so the effect of moving it is visible rather than silent.

Built in local clock time and converted, never by a fixed UTC offset: 11:00 in
London is 10:00 UTC in summer and 11:00 UTC in winter.
"""

from __future__ import annotations

from datetime import time

import pandas as pd

from src.prep.calendar_gb import LONDON, UTC, settlement_period_grid

DEFAULT_DECISION_TIME = time(11, 0)
DEFAULT_LEAD_DAYS = 1


def decision_table(
    start_date: str,
    end_date: str,
    decision_local: time = DEFAULT_DECISION_TIME,
    lead_days: int = DEFAULT_LEAD_DAYS,
) -> pd.DataFrame:
    """One row per delivery half hour, with the UTC instant its decision is taken.

    Columns: target_time (UTC), settlement_date, settlement_period, decision_time (UTC).
    """
    grid = settlement_period_grid(start_date, end_date).reset_index()
    delivery = pd.to_datetime(grid["settlement_date"]).dt.normalize()
    decision_day = delivery - pd.Timedelta(days=lead_days)

    # Naive date plus clock time, localised as a whole: the same construction as
    # neso.impute_published_at. A clock time of 11:00 never falls in a British
    # clock-change gap (those are at 01:00), so nonexistent/ambiguous cannot occur,
    # and "raise" makes that assumption fail loudly if the time is ever changed to one.
    local = decision_day + pd.Timedelta(hours=decision_local.hour, minutes=decision_local.minute)
    decision = local.dt.tz_localize(LONDON, nonexistent="raise", ambiguous="raise").dt.tz_convert(UTC)

    out = pd.DataFrame(
        {
            "target_time": grid["start_time_utc"],
            "settlement_date": delivery,
            "settlement_period": grid["settlement_period"].astype(int),
            "decision_time": decision,
        }
    )
    if not (out["decision_time"] < out.groupby("settlement_date")["target_time"].transform("min")).all():
        raise ValueError("a decision time falls inside its own delivery day")
    return out


def calendar_table(start_date: str, end_date: str) -> pd.DataFrame:
    """Local clock keys for every half hour, for joins that follow the wall clock.

    Price shape follows human activity, which follows the local clock, so "the same
    half hour two days earlier" means the same local clock time, not 48 hours
    earlier in UTC. Across a clock change the two differ by an hour.

    Columns: target_time (UTC), local_date (naive midnight), minute_of_day (local
    minutes since midnight), day_of_week (0 = Monday), month, first_occurrence.

    first_occurrence is False for the second pass through 01:00-02:00 on the
    autumn clock-change day. As a SOURCE of a clock-time lag, only the first pass
    is used, so each (local_date, minute_of_day) names exactly one half hour and a
    lag join cannot fan out. As a TARGET, both passes keep their own rows.
    """
    grid = settlement_period_grid(start_date, end_date).reset_index()
    local = grid["start_time_local"]
    local_date = local.dt.tz_localize(None).dt.normalize()
    minute = local.dt.hour * 60 + local.dt.minute
    out = pd.DataFrame(
        {
            "target_time": grid["start_time_utc"],
            "local_date": local_date,
            "minute_of_day": minute.astype(int),
            "day_of_week": local.dt.dayofweek.astype(int),
            "month": local.dt.month.astype(int),
        }
    )
    out["first_occurrence"] = ~out.duplicated(["local_date", "minute_of_day"])
    return out
