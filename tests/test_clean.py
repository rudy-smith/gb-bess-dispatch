"""Unit tests for the cleaning policy."""

import numpy as np
import pandas as pd
import pytest

from src.prep.clean import (
    CleaningPolicy,
    clean_prices,
    expected_periods,
    gap_lengths,
    interpolate_short_gaps,
    out_of_band_mask,
    stuck_mask,
)

PRICE_COL = "price_apx"


def _frame(prices, start="2024-10-01 00:00", freq="30min"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame({PRICE_COL: prices}, index=idx)


def test_expected_periods_normal_day():
    assert expected_periods("2024-06-15") == 48


def test_expected_periods_spring_forward():
    assert expected_periods("2024-03-31") == 46


def test_expected_periods_autumn_back():
    assert expected_periods("2024-10-27") == 50


def test_gap_lengths_counts_runs():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    assert list(gap_lengths(s)) == [0, 2, 2, 0, 1, 0]


def test_short_gap_is_filled_long_gap_is_not():
    prices = [10.0, np.nan, 30.0, 40.0, np.nan, np.nan, np.nan, 80.0]
    s = _frame(prices)[PRICE_COL    ]
    filled, mask = interpolate_short_gaps(s, max_gap=2)
    assert mask.sum() == 1
    assert filled.iloc[1] == pytest.approx(20.0)
    assert filled.iloc[4:7].isna().all()


def test_leading_gap_is_never_extrapolated():
    s = _frame([np.nan, 20.0, 30.0])[PRICE_COL]
    filled, mask = interpolate_short_gaps(s, max_gap=2)
    assert not mask.any()
    assert np.isnan(filled.iloc[0])


def test_out_of_band_flags_only_extremes():
    s = pd.Series([-9.74, 605.17, 9999.0, -5000.0])
    mask = out_of_band_mask(s, (-1000.0, 6000.0))
    assert list(mask) == [False, False, True, True]


def test_stuck_mask_needs_a_long_run():
    s = pd.Series([50.0] * 6 + [51.0, 52.0])
    mask = stuck_mask(s, min_run=6)
    assert mask.iloc[:6].all()
    assert not mask.iloc[6:].any()


def test_negative_prices_survive_cleaning():
    """Negative prices are market behaviour and must not be clipped away."""
    df = _frame([-20.0, 50.0, 120.0, -5.0] * 12)
    clean, _ = clean_prices(df)
    assert (clean[PRICE_COL] < 0).sum() == 24


def test_reindex_inserts_missing_periods():
    idx = pd.to_datetime(
        ["2024-10-01 00:00", "2024-10-01 00:30", "2024-10-01 02:00"], utc=True
    )
    df = pd.DataFrame({PRICE_COL: [10.0, 20.0, 30.0]}, index=idx)
    clean, report = clean_prices(df)
    assert report.grid_rows_inserted == 2
    assert len(clean) == 5


def test_inserted_rows_get_local_time_and_labels():
    """Rows created by the reindex must be labelled, not left as NaT."""
    idx = pd.to_datetime(
        ["2024-10-01 00:00", "2024-10-01 00:30", "2024-10-01 02:00"], utc=True
    )
    df = pd.DataFrame(
        {
            PRICE_COL: [10.0, 20.0, 30.0],
            "start_time_local": idx.tz_convert("Europe/London"),
        },
        index=idx,
    )
    clean, _ = clean_prices(df)
    assert clean["start_time_local"].notna().all()
    assert clean["settlement_period"].tolist() == [3, 4, 5, 6, 7]


def test_duplicate_timestamps_dropped():
    idx = pd.to_datetime(
        ["2024-10-01 00:00", "2024-10-01 00:00", "2024-10-01 00:30"], utc=True
    )
    df = pd.DataFrame({PRICE_COL: [10.0, 10.0, 20.0]}, index=idx)
    _, report = clean_prices(df)
    assert report.duplicate_rows_dropped == 1


def test_clock_change_day_is_complete_at_50_periods():
    idx = pd.date_range("2024-10-26 23:00", periods=50, freq="30min", tz="UTC")
    df = pd.DataFrame({PRICE_COL: np.linspace(20, 120, 50)}, index=idx)
    clean, report = clean_prices(df)
    day = clean[clean["settlement_date"] == pd.Timestamp("2024-10-27")]
    assert len(day) == 50
    assert day["day_complete"].all()


def test_incomplete_day_is_flagged_not_dropped():
    idx = pd.date_range("2024-06-15 00:00", periods=40, freq="30min", tz="UTC")
    df = pd.DataFrame({PRICE_COL: np.linspace(20, 120, 40)}, index=idx)
    clean, report = clean_prices(df)
    assert report.days_incomplete >= 1
    assert not clean["day_complete"].iloc[0]
    assert len(clean) == 40  # nothing deleted
