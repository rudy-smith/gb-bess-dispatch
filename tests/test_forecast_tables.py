"""Tests for the canonical forecast tables, on synthetic raw archives."""

import pandas as pd
import pytest

from src.prep.forecast_tables import build_demand_tables, build_wind_table


def _wind_raw(dates, periods=48, published="07:23"):
    rows = []
    for d in pd.to_datetime(dates):
        for sp in range(1, periods + 1):
            rows.append(
                {
                    "Datetime_GMT": "2000-01-01T00:00:00",  # deliberately ignored
                    "Date": f"{d:%Y-%m-%d}",
                    "Settlement_period": sp,
                    "Capacity": 20_000,
                    "Incentive_forecast": 1000 + sp,
                    "Forecast_Timestamp": f"{d - pd.Timedelta(days=1):%Y-%m-%d} {published}:00",
                }
            )
    return pd.DataFrame(rows)


def _demand_raw(dates, periods=48):
    rows = []
    for d in pd.to_datetime(dates):
        for sp in range(1, periods + 1):
            rows.append(
                {
                    "Date": f"{d:%Y-%m-%d}",
                    "Settlement_Period": sp,
                    "Demand_Forecast": 25_000 + sp,
                    "Demand_Outturn": 26_000 + sp,
                    "Publish_Datetime": f"{d - pd.Timedelta(days=1):%Y-%m-%d} 10:45:00",
                }
            )
    return pd.DataFrame(rows)


def test_wind_target_time_comes_from_labels_not_archive_timestamp():
    out = build_wind_table(_wind_raw(["2023-06-01"]), "2023-06-01", "2023-06-01")
    # SP1 on a BST day starts at 23:00 UTC the previous evening.
    assert out.loc[0, "target_time"] == pd.Timestamp("2023-05-31 23:00", tz="UTC")
    assert out.loc[0, "wind_forecast_mw"] == 1001


def test_wind_publication_read_as_london_time():
    out = build_wind_table(_wind_raw(["2023-06-01"]), "2023-06-01", "2023-06-01")
    # 07:23 London on 31 May (BST) is 06:23 UTC.
    assert out.loc[0, "published_at"] == pd.Timestamp("2023-05-31 06:23", tz="UTC")
    assert out["published_at_source"].eq("archive").all()


def test_wind_clock_change_day_has_46_rows():
    out = build_wind_table(_wind_raw(["2024-03-31"], periods=46), "2024-03-31", "2024-03-31")
    assert len(out) == 46
    assert out["target_time"].is_unique


def test_impossible_period_raises():
    raw = _wind_raw(["2023-06-01"], periods=50)
    with pytest.raises(ValueError, match="impossible settlement period"):
        build_wind_table(raw, "2023-06-01", "2023-06-01")


def test_duplicate_vintage_raises():
    """Two values for one target and one publication instant make the as-of
    join's answer depend on row order."""
    raw = _wind_raw(["2023-06-01"])
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_wind_table(raw, "2023-06-01", "2023-06-01")


def test_date_range_is_applied():
    out = build_wind_table(_wind_raw(["2023-06-01", "2023-06-02"]), "2023-06-02", "2023-06-02")
    assert out["settlement_date"].nunique() == 1


def test_demand_outturn_is_not_in_the_forecast_table():
    forecast, outturn = build_demand_tables(
        _demand_raw(["2024-01-15"]), "2024-01-15", "2024-01-15"
    )
    assert "demand_outturn_mw" not in forecast.columns
    assert "demand_outturn_mw" in outturn.columns
    assert len(forecast) == len(outturn) == 48


def test_demand_publication_read_as_utc():
    forecast, _ = build_demand_tables(_demand_raw(["2024-07-15"]), "2024-07-15", "2024-07-15")
    assert forecast.loc[0, "published_at"] == pd.Timestamp("2024-07-14 10:45", tz="UTC")
    