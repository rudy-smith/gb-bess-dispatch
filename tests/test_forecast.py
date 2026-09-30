"""Tests for walk-forward folds, the scoring functions and the model wrapper."""

import numpy as np
import pandas as pd
import pytest

from src.forecast.evaluate import (
    block_bootstrap_skill,
    interval_coverage,
    mae,
    matched,
    pinball,
    reliability,
    rmse,
)
from src.forecast.walkforward import GAP_DAYS, monthly_folds, walk_forward

# --- folds ------------------------------------------------------------------------


def test_folds_cover_each_month_once():
    folds = monthly_folds("2023-07-01", "2024-12-01")
    assert len(folds) == 18
    days = pd.DatetimeIndex([])
    for f in folds:
        days = days.append(pd.date_range(f.test_start, f.test_end))
    assert days.is_unique
    assert days.equals(pd.date_range("2023-07-01", "2024-12-31"))


def test_training_stops_two_days_before_the_test_month():
    """At 11:00 on the day before the month, the last complete day of prices is
    two days before the month starts."""
    for f in monthly_folds("2023-07-01", "2024-12-01"):
        assert f.train_end == f.test_start - pd.Timedelta(days=GAP_DAYS)
        assert GAP_DAYS == 2


# --- metrics ----------------------------------------------------------------------


def test_mae_rmse():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_pinball_is_asymmetric():
    """Under-forecasting a high quantile costs q per unit; over-forecasting costs 1-q."""
    assert pinball([10], [0], 0.9) == pytest.approx(9.0)
    assert pinball([0], [10], 0.9) == pytest.approx(1.0)
    assert pinball([10], [0], 0.5) == pytest.approx(5.0)


def test_pinball_is_minimised_near_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 20_000)
    grid = np.linspace(-2, 2, 81)
    best = grid[np.argmin([pinball(y, np.full_like(y, g), 0.9) for g in grid])]
    assert best == pytest.approx(np.quantile(y, 0.9), abs=0.06)


def test_reliability_of_true_quantiles_is_nominal():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 50_000)
    f = pd.DataFrame(
        {"target_price": y, "q10": -1.2816, "q50": 0.0, "q90": 1.2816}
    )
    rel = reliability(f, {0.1: "q10", 0.5: "q50", 0.9: "q90"})
    assert rel["observed"].to_numpy() == pytest.approx([0.1, 0.5, 0.9], abs=0.01)
    assert interval_coverage(f, "q10", "q90") == pytest.approx(0.8, abs=0.01)


def test_matched_drops_rows_missing_any_forecast():
    f = pd.DataFrame({"target_price": [1, 2, np.nan], "a": [1, np.nan, 1], "b": [1, 1, 1]})
    assert len(matched(f, ["a", "b"])) == 1


def test_bootstrap_skill_sign_and_interval():
    days = pd.date_range("2024-01-01", periods=120).repeat(48)
    rng = np.random.default_rng(2)
    y = rng.normal(50, 20, len(days))
    f = pd.DataFrame(
        {
            "settlement_date": days,
            "target_price": y,
            "good": y + rng.normal(0, 5, len(y)),
            "bad": y + rng.normal(0, 10, len(y)),
        }
    )
    s, lo, hi = block_bootstrap_skill(f, "good", "bad", n_boot=500)
    assert 0.4 < s < 0.6
    assert lo < s < hi


# --- model ------------------------------------------------------------------------


def _synthetic_frame():
    """Price = daily shape + level that shifts halfway through + noise, with the
    features that make it learnable."""
    days = pd.date_range("2023-01-01", "2023-12-31")
    rows = []
    rng = np.random.default_rng(3)
    for i, d in enumerate(days):
        level = 100 if i < 180 else 60
        for sp in range(1, 49):
            shape = 30 * np.sin(2 * np.pi * (sp - 12) / 48)
            rows.append(
                {
                    "settlement_date": d,
                    "settlement_period": sp,
                    "target_price": level + shape + rng.normal(0, 5),
                    "f_minute_of_day": (sp - 1) * 30,
                    "f_price_7d_mean": level + rng.normal(0, 2),
                    "f_price_d7_same_time": level + shape + rng.normal(0, 8),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_is_out_of_sample_and_beats_naive():
    frame = _synthetic_frame()
    feats = ["f_minute_of_day", "f_price_7d_mean", "f_price_d7_same_time"]
    folds = monthly_folds("2023-04-01", "2023-12-01")
    res = walk_forward(frame, feats, folds)
    assert res["fold"].nunique() == 9
    assert pd.to_datetime(res["settlement_date"]).min() == pd.Timestamp("2023-04-01")
    assert mae(res["target_price"], res["pred_point"]) < mae(
        res["target_price"], res["f_price_d7_same_time"]
    )
    q = res[["pred_q10", "pred_q50", "pred_q90"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()  # sorted, no crossing


def test_anchoring_follows_a_level_shift():
    """The level drops from 100 to 60 at the end of June. In July, a model trained
    on raw prices, almost all from the old level, predicts the old level; the
    anchored model takes the level from the trailing mean and follows at once.
    Measured: about 6 against about 39 £/MWh MAE."""
    frame = _synthetic_frame()
    feats = ["f_minute_of_day", "f_price_7d_mean"]
    folds = monthly_folds("2023-07-01", "2023-07-01")
    anchored = walk_forward(frame, feats, folds, anchored=True)
    raw = walk_forward(frame, feats, folds, anchored=False)
    a = mae(anchored["target_price"], anchored["pred_point"])
    r = mae(raw["target_price"], raw["pred_point"])
    assert a < 10
    assert r > 3 * a

