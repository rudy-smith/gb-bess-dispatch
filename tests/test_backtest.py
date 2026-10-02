"""Tests for forecast-driven dispatch and settlement.

The tests are built around properties that hold whatever the data: a perfect
forecast captures everything, no schedule settled at actual prices beats perfect
foresight, and a forecast that only rescales prices changes nothing. Each one
fails if revenue is settled at the forecast instead of the actual price, which is
the bug this phase is most exposed to.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.forecast_dispatch import (
    bootstrap_capture,
    bootstrap_capture_difference,
    capture,
    matched_days,
    run_backtest,
    run_day,
)
from src.model.battery import BatterySpec

SPEC = BatterySpec()


def _day(seed, n=48):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 60 + 40 * np.sin(2 * np.pi * (t - 14) / n) + rng.normal(0, 10, n)


def test_perfect_forecast_captures_everything():
    actual = _day(0)
    r = run_day(actual, {"fc": actual.copy()}, SPEC)
    assert r["fc_net"] == pytest.approx(r["perfect_net"], abs=1e-6)


def test_rescaled_forecast_gives_the_same_schedule():
    """Doubling every price leaves the optimal schedule unchanged, so settled at
    actual prices it earns exactly perfect foresight. Settled at the forecast it
    would earn twice as much, which the invariant would reject."""
    actual = _day(1)
    r = run_day(actual, {"double": 2 * actual}, SPEC)
    assert r["double_net"] == pytest.approx(r["perfect_net"], abs=1e-6)


def test_flat_forecast_does_not_trade():
    """With losses on every cycle, a flat price offers nothing to arbitrage."""
    r = run_day(_day(2), {"flat": np.full(48, 70.0)}, SPEC)
    assert r["flat_net"] == pytest.approx(0.0, abs=1e-6)
    assert r["flat_mwh"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_no_strategy_beats_perfect_foresight(seed):
    actual = _day(seed)
    rng = np.random.default_rng(100 + seed)
    forecasts = {
        "noisy": actual + rng.normal(0, 25, 48),
        "shifted": np.roll(actual, 6),
        "inverted": -actual,
    }
    r = run_day(actual, forecasts, SPEC)
    for name in forecasts:
        assert r[f"{name}_net"] <= r["perfect_net"] + 1e-6


def test_inverted_forecast_loses_money():
    actual = _day(3)
    r = run_day(actual, {"inverted": -actual}, SPEC)
    assert r["inverted_net"] < 0


def test_degradation_applies_to_both_sides():
    actual = _day(4)
    r0 = run_day(actual, {"fc": actual}, SPEC, degradation_cost=0.0)
    r10 = run_day(actual, {"fc": actual}, SPEC, degradation_cost=10.0)
    assert r10["perfect_net"] < r0["perfect_net"]
    assert r10["fc_net"] == pytest.approx(r10["perfect_net"], abs=1e-6)


def _frame(days=20):
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=days)):
        actual = _day(i)
        noisy = actual + np.random.default_rng(50 + i).normal(0, 15, 48)
        for sp in range(48):
            rows.append(
                {
                    "settlement_date": d,
                    "target_time": d.tz_localize("UTC") + pd.Timedelta(minutes=30 * sp),
                    "target_price": actual[sp],
                    "day_usable": True,
                    "wind_suspect": False,
                    "good": noisy[sp],
                    "perfect_copy": actual[sp],
                }
            )
    return pd.DataFrame(rows)


def test_days_missing_any_strategy_are_dropped_for_all():
    f = _frame()
    f.loc[f.index[5], "good"] = np.nan
    f.loc[f.index[48 * 3], "day_usable"] = False
    kept, dropped = matched_days(f, ["good", "perfect_copy"])
    assert len(kept) == 18 and len(dropped) == 2


def test_backtest_capture_and_interval():
    f = _frame()
    daily, dropped = run_backtest(f, {"good": "good", "copy": "perfect_copy"}, SPEC)
    assert not dropped
    assert capture(daily, "copy") == pytest.approx(1.0, abs=1e-9)
    c, lo, hi = bootstrap_capture(daily, "good", n_boot=300)
    assert 0 < c < 1
    assert lo <= c <= hi


def test_capture_difference_is_paired():
    f = _frame()
    daily, _ = run_backtest(f, {"good": "good", "copy": "perfect_copy"}, SPEC)
    d, lo, hi = bootstrap_capture_difference(daily, "copy", "good", n_boot=300)
    assert d == pytest.approx(1 - capture(daily, "good"), abs=1e-9)
    assert lo <= d <= hi
    z, zlo, zhi = bootstrap_capture_difference(daily, "good", "good", n_boot=300)
    assert z == zlo == zhi == 0
    