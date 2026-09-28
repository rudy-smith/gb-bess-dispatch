"""Tests for the forecast-skill measures and the regeneration test.

The verdict is tested on synthetic years whose answer is known by construction:
a block with the same error distribution as its controls must not be called
regenerated, and a block with half the error must be.
"""

import numpy as np
import pandas as pd
import pytest

from src.prep.forecast_skill import (
    add_errors,
    bootstrap_ratio,
    contiguous_runs,
    daily_mae,
    in_calendar_window,
    join_forecast_outturn,
    summarise,
    verdict,
)


def _year(year, error_scale, seed, capacity=20_000.0):
    """120 days x 48 periods of synthetic forecast/outturn with a given error size.

    Errors are autocorrelated across days, like weather, so the block bootstrap
    has something to do.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{year}-06-01", periods=120)
    day_level = np.convolve(rng.normal(0, 1, 126), np.ones(7) / np.sqrt(7), mode="valid")
    rows = []
    for d, level in zip(days, day_level):
        sp = np.arange(1, 49)
        outturn = capacity * 0.3 + rng.normal(0, 2000, 48)
        err = error_scale * capacity * (0.5 * level + rng.normal(0, 1, 48)) * (1 + sp / 96)
        rows.append(
            pd.DataFrame(
                {
                    "settlement_date": d,
                    "settlement_period": sp,
                    "wind_forecast_mw": outturn + err,
                    "wind_outturn_mw": outturn,
                    "wind_capacity_mw": capacity,
                }
            )
        )
    return add_errors(pd.concat(rows, ignore_index=True))


def test_error_sign_and_normalisation():
    f = pd.DataFrame(
        {"wind_forecast_mw": [100.0], "wind_outturn_mw": [80.0], "wind_capacity_mw": [200.0]}
    )
    out = add_errors(f)
    assert out.loc[0, "error_share"] == pytest.approx(0.1)  # over-forecast is positive


def test_zero_capacity_gives_nan_not_infinity():
    f = pd.DataFrame({"wind_forecast_mw": [1.0], "wind_outturn_mw": [0.0], "wind_capacity_mw": [0]})
    assert np.isnan(add_errors(f).loc[0, "error_share"])


def test_join_rejects_duplicate_keys():
    forecast = pd.DataFrame(
        {
            "settlement_date": ["2023-06-01", "2023-06-01"],
            "settlement_period": [1, 1],
            "wind_forecast_mw": [1.0, 2.0],
            "wind_capacity_mw": [10.0, 10.0],
        }
    )
    outturn = pd.DataFrame(
        {"settlement_date": ["2023-06-01"], "settlement_period": [1], "wind_outturn_mw": [1.0]}
    )
    with pytest.raises(pd.errors.MergeError):
        join_forecast_outturn(forecast, outturn)


def test_contiguous_runs_finds_gaps():
    dates = pd.Series(pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-03", "2023-06-10"]))
    runs = contiguous_runs(dates)
    assert runs == [
        (pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-03")),
        (pd.Timestamp("2023-06-10"), pd.Timestamp("2023-06-10")),
    ]


def test_calendar_window_ignores_year():
    dates = pd.Series(pd.to_datetime(["2022-06-01", "2024-10-03", "2024-10-04", "2023-05-31"]))
    mask = in_calendar_window(dates, pd.Timestamp("2023-06-01"), pd.Timestamp("2023-10-03"))
    assert mask.tolist() == [True, True, False, False]


def test_error_growth_is_detected():
    """The synthetic generator widens error with settlement period; summarise sees it."""
    assert summarise(_year(2022, 0.07, seed=1))["growth_late_over_early"] > 1.1


def test_same_distribution_is_not_called_regenerated():
    block = daily_mae(_year(2023, 0.07, seed=2))
    controls = pd.concat([daily_mae(_year(2022, 0.07, seed=3)), daily_mae(_year(2024, 0.07, seed=4))])
    _, lo, hi = bootstrap_ratio(block, controls, n_boot=2000)
    assert lo < 1 < hi
    assert verdict(lo, hi).label != "REGENERATED"


def test_halved_error_is_called_regenerated():
    block = daily_mae(_year(2023, 0.035, seed=2))
    controls = pd.concat([daily_mae(_year(2022, 0.07, seed=3)), daily_mae(_year(2024, 0.07, seed=4))])
    ratio, lo, hi = bootstrap_ratio(block, controls, n_boot=2000)
    assert ratio == pytest.approx(0.5, abs=0.08)
    assert verdict(lo, hi).label == "REGENERATED"


def test_bootstrap_is_reproducible():
    a = daily_mae(_year(2023, 0.07, seed=5))
    b = daily_mae(_year(2022, 0.07, seed=6))
    assert bootstrap_ratio(a, b, n_boot=500, seed=7) == bootstrap_ratio(a, b, n_boot=500, seed=7)


def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_days():
    """Resampling correlated days one at a time understates uncertainty."""
    a = daily_mae(_year(2023, 0.07, seed=8))
    b = daily_mae(_year(2022, 0.07, seed=9))
    _, lo7, hi7 = bootstrap_ratio(a, b, n_boot=4000, block=7)
    _, lo1, hi1 = bootstrap_ratio(a, b, n_boot=4000, block=1)
    assert (hi7 - lo7) > (hi1 - lo1)


def test_verdict_thresholds():
    assert verdict(0.85, 1.2).label == "GENUINE"
    assert verdict(0.4, 0.6).label == "REGENERATED"
    assert verdict(0.6, 0.95).label == "INCONCLUSIVE"
