"""Tests for the GB settlement-period calendar.
 
Every test here exists because it is cheap to get this wrong and expensive to
notice later: a one-period offset in the calendar shifts every price by half an
hour, which silently inflates or destroys arbitrage revenue in a way that looks
plausible on a chart.
"""
 
import pandas as pd
import pytest
 
from src.prep.calendar_gb import (
    expected_periods_in_day,
    from_utc,
    settlement_period_grid,
    to_utc,
)
 
 
@pytest.mark.parametrize(
    "date,expected",
    [
        ("2024-06-15", 48),  # ordinary BST day
        ("2024-01-15", 48),  # ordinary GMT day
        ("2024-03-31", 46),  # spring forward: 23 hours
        ("2024-10-27", 50),  # autumn back: 25 hours
        ("2025-03-30", 46),
        ("2025-10-26", 50),
    ],
)
def test_period_counts(date, expected):
    assert expected_periods_in_day(date) == expected
    assert len(settlement_period_grid(date, date)) == expected
 
 
def test_grid_is_an_unbroken_utc_half_hourly_range():
    """The point of a UTC index: completeness is checkable in one line."""
    g = settlement_period_grid("2023-01-01", "2025-12-31")
    assert (g.index == pd.date_range(g.index[0], g.index[-1], freq="30min")).all()
    assert not g.index.duplicated().any()
    assert g.index.is_monotonic_increasing
    assert str(g.index.tz) == "UTC"
 
 
def test_three_years_have_exactly_three_short_and_three_long_days():
    g = settlement_period_grid("2023-01-01", "2025-12-31")
    counts = g.groupby("settlement_date").size().value_counts().to_dict()
    assert counts[46] == 3
    assert counts[50] == 3
 
 
def test_spring_forward_skips_local_one_am():
    """On 2024-03-31 the local clock goes 00:59 -> 02:00, so SP3 is 02:00 local."""
    g = settlement_period_grid("2024-03-31", "2024-03-31")
    sp3 = g[g["settlement_period"] == 3].iloc[0]
    assert sp3["start_time_local"].hour == 2
    assert g.index[2] == pd.Timestamp("2024-03-31 01:00", tz="UTC")
 
 
def test_autumn_back_repeats_local_one_am_at_distinct_periods():
    """Local 01:00 occurs twice on 2024-10-27 - the case a local index cannot hold."""
    g = settlement_period_grid("2024-10-27", "2024-10-27")
    ones = g[g["start_time_local"].dt.strftime("%H:%M") == "01:00"]
    assert len(ones) == 2
    assert sorted(ones["settlement_period"]) == [3, 5]
    assert ones.index[1] - ones.index[0] == pd.Timedelta(hours=1)
 
 
def test_to_utc_and_from_utc_are_exact_inverses():
    g = settlement_period_grid("2024-01-01", "2024-12-31")
    fwd = pd.DatetimeIndex([to_utc(r.settlement_date, r.settlement_period)
                            for r in g.itertuples()])
    assert (fwd == g.index).all()
    for ts, row in zip(g.index, g.itertuples()):
        assert from_utc(ts) == (row.settlement_date, row.settlement_period)
 
 
def test_out_of_range_period_raises():
    with pytest.raises(ValueError, match="out of range"):
        to_utc("2024-03-31", 48)  # only 46 periods exist that day
    with pytest.raises(ValueError, match="out of range"):
        to_utc("2024-06-15", 50)
 
 
def test_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        from_utc(pd.Timestamp("2024-06-15 12:00"))
 