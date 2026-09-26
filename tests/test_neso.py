"""Tests for the NESO fetcher, covering the vintage imputation.
 
The imputation is the part that decides whether the backtest is honest, so it
is tested against the two dates where local time and UTC diverge and against a
clock-change day, rather than only on an arbitrary midweek date.
"""
 
import pandas as pd
import pytest
 
from src.fetch.neso import RESOURCES, impute_published_at
 
WIND = RESOURCES["wind_da"]
 
 
def _dates(*values):
    return pd.Series(pd.to_datetime(list(values)))
 
 
def test_publication_is_the_day_before_the_target():
    published = impute_published_at(_dates("2024-01-15"), WIND)
    assert published.iloc[0] == pd.Timestamp("2024-01-14 09:15", tz="Europe/London")
 
 
def test_winter_publication_is_0915_utc():
    """In GMT, 09:15 clock time is 09:15 UTC."""
    published = impute_published_at(_dates("2024-01-15"), WIND)
    assert published.iloc[0].tz_convert("UTC") == pd.Timestamp("2024-01-14 09:15", tz="UTC")
 
 
def test_summer_publication_is_0815_utc():
    """In BST, 09:15 clock time is 08:15 UTC.
 
    Subtracting a fixed offset from UTC would put this an hour late, granting
    the strategy information an hour before it existed for half the year.
    """
    published = impute_published_at(_dates("2024-07-15"), WIND)
    assert published.iloc[0].tz_convert("UTC") == pd.Timestamp("2024-07-14 08:15", tz="UTC")
 
 
def test_publication_precedes_every_target_period():
    """The decision instant must precede the first settlement period it informs."""
    targets = _dates("2024-03-31", "2024-10-27", "2024-07-15", "2024-01-15")
    published = impute_published_at(targets, WIND)
    day_start = targets.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    assert (published.dt.tz_convert("UTC") < day_start).all()
 
 
def test_clock_change_days_are_handled():
    """Targets on both 2024 transition days impute without raising."""
    published = impute_published_at(_dates("2024-03-31", "2024-10-27"), WIND)
    assert published.notna().all()
    assert len(published) == 2
 
 
def test_demand_uses_its_own_schedule():
    """Each resource carries its own publication assumption."""
    demand = RESOURCES["demand_da_hh"]
    assert demand.resource_id != WIND.resource_id
    assert impute_published_at(_dates("2024-01-15"), demand).iloc[0] == pd.Timestamp(
        "2024-01-14 09:15", tz="Europe/London"
    )
 
 
def test_target_times_are_dropped_to_date_granularity():
    """A target carrying a time of day does not shift the publication instant."""
    with_time = pd.Series(pd.to_datetime(["2024-01-15 17:30"]))
    without = _dates("2024-01-15")
    assert impute_published_at(with_time, WIND).iloc[0] == impute_published_at(
        without, WIND
    ).iloc[0]
 
 
@pytest.mark.parametrize("name", sorted(RESOURCES))
def test_every_resource_publishes_before_its_target(name):
    resource = RESOURCES[name]
    assert resource.lead_days >= 1
    assert resource.publication_local_time.hour < 24
 
 
# --------------------------------------------------------------------------- #
# Publication timestamps from the archive
#
# The wind archive carries its own Forecast_Timestamp, and some of its rows
# record an instant after the settlement day they forecast. These tests pin the
# rule that decides which recorded timestamps may be used.
# --------------------------------------------------------------------------- #
 
from src.fetch.neso import day_start_utc, normalise_wind, resolve_published_at
 
 
def _wind_rows():
    """Two rows for 2024-01-15, one credible timestamp and one published late."""
    return pd.DataFrame(
        {
            "Datetime_GMT": ["2024-01-15 00:00:00", "2024-01-15 00:30:00"],
            "Date": ["2024-01-15", "2024-01-15"],
            "Settlement_period": [1, 2],
            "Capacity": [28000.0, 28000.0],
            "Incentive_forecast": [9100.0, 9400.0],
            # First published the morning before; second stamped two days after
            # delivery, the pattern seen at the start of the real archive.
            "Forecast_Timestamp": ["2024-01-14 09:05:00", "2024-01-17 08:05:00"],
        }
    )
 
 
def test_archive_timestamp_is_used_when_it_precedes_the_settlement_day():
    frame = _wind_rows()
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
    assert out["published_at_source"].iloc[0] == "archive"
    assert not out["published_at_suspect"].iloc[0]
    assert out["published_at"].iloc[0] == pd.Timestamp("2024-01-14 09:05", tz="UTC")
 
 
def test_timestamp_after_the_settlement_day_is_rejected():
    """A forecast published after delivery is the outturn wearing a disguise."""
    frame = _wind_rows()
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
    assert out["published_at_suspect"].iloc[1]
    assert out["published_at_source"].iloc[1] == "schedule"
    assert out["published_at"].iloc[1] < starts.iloc[1]
 
 
def test_every_resolved_instant_precedes_its_settlement_day():
    frame = _wind_rows()
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
    assert (out["published_at"] < starts).all()
 
 
def test_missing_timestamp_falls_back_rather_than_passing():
    frame = _wind_rows()
    frame.loc[0, "Forecast_Timestamp"] = None
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
    assert out["published_at_suspect"].iloc[0]
    assert out["published_at_source"].iloc[0] == "schedule"
 
 
def test_day_start_is_local_midnight_not_utc_midnight():
    """In BST the settlement day opens at 23:00 UTC on the previous date."""
    assert day_start_utc(_dates("2024-07-15")).iloc[0] == pd.Timestamp(
        "2024-07-14 23:00", tz="UTC"
    )
    assert day_start_utc(_dates("2024-01-15")).iloc[0] == pd.Timestamp(
        "2024-01-15 00:00", tz="UTC"
    )
 
 
def test_wind_normalisation_produces_canonical_columns():
    out = normalise_wind(_wind_rows())
    assert list(out["settlement_period"]) == [1, 2]
    assert out.index.tz is not None
    assert out["wind_forecast_mw"].iloc[0] == pytest.approx(9100.0)
    assert out["wind_forecast_share"].iloc[0] == pytest.approx(9100.0 / 28000.0)
 
 
def test_zero_capacity_does_not_produce_an_infinite_share():
    frame = _wind_rows()
    frame["Capacity"] = 0.0
    out = normalise_wind(frame)
    assert out["wind_forecast_share"].isna().all()
 
 
def test_wind_normalisation_names_missing_columns():
    frame = _wind_rows().drop(columns=["Incentive_forecast"])
    with pytest.raises(KeyError, match="Incentive_forecast"):
        normalise_wind(frame)
 
 
# --------------------------------------------------------------------------- #
# Demand archive
#
# The demand archive's own Datetime column uses a different convention from the
# wind archive's. These tests pin the decision not to rely on either.
# --------------------------------------------------------------------------- #
 
from src.fetch.neso import DEMAND_OUTTURN_COLUMNS, normalise_demand
 
 
def _demand_rows():
    return pd.DataFrame(
        {
            "Month": ["Apr", "Apr"],
            "Date": ["2021-04-01", "2021-04-01"],
            # Local time, period end. Wind records UTC period start for the same
            # label, which is why neither column is used as a join key.
            "Datetime": ["2021-04-01 00:30:00", "2021-04-01 01:00:00"],
            "Settlement_Period": [1, 2],
            "Demand_Forecast": [21570, 21130],
            "Demand_Outturn": [19970, 20030],
            "TRIAD_Avoidance_Estimate": [0, 0],
            "TRIAD_Avoidance_Corrected_Demand_Outturn": [19970, 20030],
            "APE": [8.0, 5.5],
            "Absolute_Error": [1600, 1100],
            "Publish_Datetime": ["2021-03-31 09:45:00", "2021-03-31 09:45:00"],
        }
    )
 
 
def test_demand_normalisation_produces_canonical_columns():
    out = normalise_demand(_demand_rows())
    assert list(out["settlement_period"]) == [1, 2]
    assert out["demand_forecast_mw"].iloc[0] == 21570
    assert out["settlement_date"].iloc[0] == pd.Timestamp("2021-04-01")
 
 
def test_demand_outturn_columns_are_named_for_exclusion():
    out = normalise_demand(_demand_rows())
    for column in DEMAND_OUTTURN_COLUMNS:
        assert column in out.columns
 
 
def test_demand_forecast_is_not_an_outturn_column():
    """The forecast must never appear in the exclusion list."""
    assert "demand_forecast_mw" not in DEMAND_OUTTURN_COLUMNS
 
 
def test_demand_normalisation_does_not_index_on_the_archive_datetime():
    """Indexing on Datetime would misalign demand against wind by up to 90 min."""
    out = normalise_demand(_demand_rows())
    assert not isinstance(out.index, pd.DatetimeIndex)
 
 
def test_demand_normalisation_names_missing_columns():
    frame = _demand_rows().drop(columns=["Demand_Forecast"])
    with pytest.raises(KeyError, match="Demand_Forecast"):
        normalise_demand(frame)
 
 
def test_demand_timestamp_is_read_as_configured():
    frame = _demand_rows()
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, RESOURCES["demand_da_hh"], "Date", starts)
    assert out["published_at_source"].iloc[0] == "archive"
    assert out["published_at"].iloc[0] == pd.Timestamp("2021-03-31 09:45", tz="UTC")
 
 
def test_wind_archive_timestamp_is_read_as_local_clock_time():
    """A July timestamp is BST, so it resolves one hour earlier in UTC.
 
    Established from the archive rather than assumed: read as local the
    publication hour is stable across the year, read as UTC it shifts by one
    hour at each clock change.
    """
    frame = _wind_rows()
    frame["Date"] = ["2024-07-15", "2024-07-15"]
    frame["Datetime_GMT"] = ["2024-07-14 23:00:00", "2024-07-14 23:30:00"]
    frame["Forecast_Timestamp"] = ["2024-07-14 07:23:00", "2024-07-14 07:23:00"]
 
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
 
    assert (out["published_at_source"] == "archive").all()
    assert out["published_at"].iloc[0] == pd.Timestamp("2024-07-14 06:23", tz="UTC")
 
 
def test_wind_winter_timestamp_is_unchanged_by_the_local_reading():
    """In GMT the two readings coincide, which is why one sample cannot decide."""
    frame = _wind_rows()
    frame["Forecast_Timestamp"] = ["2024-01-14 07:23:00", "2024-01-14 07:23:00"]
    starts = day_start_utc(frame["Date"])
    out = resolve_published_at(frame, WIND, "Date", starts)
    assert out["published_at"].iloc[0] == pd.Timestamp("2024-01-14 07:23", tz="UTC")

 