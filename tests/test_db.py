"""Tests for decision times and the point-in-time join.

The join is tested on vintages placed either side of the decision instant, and
exactly on it, because the boundary is where a look-ahead bug would live.
"""

from datetime import time

import pandas as pd
import pytest

from src.db.duck import connect, query_file, register_decisions
from src.prep.decisions import decision_table

# --- decision times ----------------------------------------------------------


@pytest.mark.parametrize(
    "delivery, expected_utc",
    [
        ("2024-01-15", "2024-01-14 11:00"),  # GMT
        ("2024-07-15", "2024-07-14 10:00"),  # BST: 11:00 London is 10:00 UTC
        ("2024-03-31", "2024-03-30 11:00"),  # spring-forward day; decision still in GMT
        ("2024-04-01", "2024-03-31 10:00"),  # decision on the spring-forward day, after 01:00
        ("2024-10-27", "2024-10-26 10:00"),  # autumn-back day; decision still in BST
    ],
)
def test_decision_time_is_1100_london_on_the_previous_day(delivery, expected_utc):
    t = decision_table(delivery, delivery)
    assert t["decision_time"].nunique() == 1
    assert t["decision_time"].iloc[0] == pd.Timestamp(expected_utc, tz="UTC")


@pytest.mark.parametrize("delivery, n", [("2024-03-31", 46), ("2024-10-27", 50)])
def test_decision_table_covers_clock_change_days(delivery, n):
    assert len(decision_table(delivery, delivery)) == n


def test_decision_precedes_delivery_day():
    t = decision_table("2023-01-01", "2024-12-31")
    first = t.groupby("settlement_date")["target_time"].transform("min")
    assert (t["decision_time"] < first).all()


# --- point-in-time join --------------------------------------------------------


DAY = "2024-01-15"  # GMT, so decision is 2024-01-14 11:00 UTC
DECISION = pd.Timestamp("2024-01-14 11:00", tz="UTC")


def _write(tmp_path, wind_vintages, demand_vintages=None):
    """Write synthetic processed files. Each vintage is (published_at, value) and
    applies to every half hour of DAY."""
    proc = tmp_path / "processed"
    proc.mkdir()
    grid = decision_table(DAY, DAY)

    prices = pd.DataFrame(
        {
            "start_time_utc": grid["target_time"],
            "price_apx": 50.0,
            "period_usable": True,
            "day_usable": True,
        }
    )
    prices.to_parquet(proc / "prices_full.parquet", index=False)

    def forecasts(vintages, value_col):
        frames = []
        for published, value in vintages:
            f = grid[["target_time", "settlement_date", "settlement_period"]].copy()
            f["published_at"] = pd.Timestamp(published, tz="UTC")
            f["published_at_source"] = "archive"
            f["published_at_suspect"] = False
            f[value_col] = float(value)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    w = forecasts(wind_vintages, "wind_forecast_mw")
    w["wind_capacity_mw"] = 20_000.0
    w["wind_forecast_share"] = w["wind_forecast_mw"] / 20_000.0
    w.to_parquet(proc / "wind_forecasts.parquet", index=False)
    forecasts(demand_vintages or [("2024-01-14 10:45", 25_000)], "demand_forecast_mw").to_parquet(
        proc / "demand_forecasts.parquet", index=False
    )
    return tmp_path


def _frame(data_dir):
    con = connect(data_dir, required=("prices", "wind_forecasts", "demand_forecasts"))
    register_decisions(con, DAY, DAY)
    return query_file(con, "point_in_time")


def test_latest_vintage_before_decision_is_chosen(tmp_path):
    data = _write(
        tmp_path,
        [("2024-01-14 07:00", 1), ("2024-01-14 10:59", 2), ("2024-01-14 12:00", 3)],
    )
    out = _frame(data)
    assert (out["wind_forecast_mw"] == 2).all()
    assert (out["wind_published_at"] < out["decision_time"]).all()


def test_vintage_published_exactly_at_decision_is_excluded(tmp_path):
    """Strictly before: a forecast published at the decision instant is too late."""
    data = _write(tmp_path, [("2024-01-14 07:00", 1), ("2024-01-14 11:00", 2)])
    assert (_frame(data)["wind_forecast_mw"] == 1).all()


def test_no_qualifying_vintage_keeps_the_row_with_nulls(tmp_path):
    """Only a late vintage exists: the row survives, the forecast is NULL."""
    data = _write(tmp_path, [("2024-01-14 12:00", 3)])
    out = _frame(data)
    assert len(out) == 48
    assert out["wind_forecast_mw"].isna().all()
    assert out["demand_forecast_mw"].notna().all()


def test_join_preserves_one_row_per_half_hour(tmp_path):
    data = _write(tmp_path, [("2024-01-14 07:00", 1), ("2024-01-14 09:00", 2)],
                  [("2024-01-14 08:00", 1), ("2024-01-14 10:45", 2)])
    out = _frame(data)
    assert len(out) == 48
    assert out["target_time"].is_unique
    assert (out["demand_forecast_mw"] == 2).all()


def test_earlier_decision_loses_the_demand_forecast(tmp_path):
    """At 09:20 the 10:45 demand forecast does not yet exist."""
    data = _write(tmp_path, [("2024-01-14 07:00", 1)])
    con = connect(data)
    register_decisions(con, DAY, DAY, time(9, 20))
    out = query_file(con, "point_in_time")
    assert out["demand_forecast_mw"].isna().all()
    assert out["wind_forecast_mw"].notna().all()


def test_missing_required_view_raises(tmp_path):
    (tmp_path / "processed").mkdir()
    with pytest.raises(FileNotFoundError, match="wind_forecasts"):
        connect(tmp_path, required=("wind_forecasts",))


def test_session_runs_in_utc(tmp_path):
    (tmp_path / "processed").mkdir()
    con = connect(tmp_path)
    assert con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"
