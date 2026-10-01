"""Tests for the look-ahead audit: it must agree with a correct feature frame and
catch an incorrect one."""

from pathlib import Path

import pandas as pd
import pytest

from src.prep.feature_audit import audit
from src.prep.features import FEATURES
from tests.test_features import LAG, _build


def _inputs(tmp_path: Path, start: str, end: str, interpolated=()):
    features = _build(tmp_path, start, end, interpolated=interpolated)
    proc = tmp_path / "processed"
    return (
        features,
        pd.read_parquet(proc / "prices_full.parquet"),
        pd.read_parquet(proc / "point_in_time.parquet"),
        pd.read_parquet(proc / "fuelhh.parquet"),
    )


@pytest.fixture(scope="module")
def spring(tmp_path_factory):
    return _inputs(tmp_path_factory.mktemp("a_spring"), "2024-03-01", "2024-04-06",
                   interpolated=["2024-03-19 09:00"])


@pytest.fixture(scope="module")
def autumn(tmp_path_factory):
    return _inputs(tmp_path_factory.mktemp("a_autumn"), "2023-10-15", "2023-11-02")


@pytest.mark.parametrize("fixture", ["spring", "autumn"])
def test_independent_recomputation_agrees_with_sql(fixture, request):
    result = audit(*request.getfixturevalue(fixture), lag_minutes=LAG)
    assert set(result.index) == set(FEATURES)
    assert (result["mismatches"] == 0).all(), result[result["mismatches"] > 0]
    assert (result["verdict"] == "PASS").all()


def test_every_timed_feature_has_a_positive_margin(spring):
    result = audit(*spring, lag_minutes=LAG)
    timed = result[result["source"] != "clock"]
    assert (timed["min_margin_h"] > 0).all()


def test_audit_catches_a_leaked_value(spring):
    """Replace one D-2 value with the price of the delivery half hour itself."""
    features, prices, pit, gen = spring
    bad = features.copy()
    i = bad.index[bad["f_price_d2_same_time"].notna()][100]
    bad.loc[i, "f_price_d2_same_time"] = bad.loc[i, "target_price"]
    result = audit(bad, prices, pit, gen, lag_minutes=LAG)
    assert result.loc["f_price_d2_same_time", "mismatches"] == 1
    assert result.loc["f_price_d2_same_time", "verdict"] == "FAIL"


def test_audit_catches_a_late_forecast(spring):
    """A forecast whose publication instant is after the decision fails on margin."""
    features, prices, pit, gen = spring
    late = pit.copy()
    late.loc[late.index[500], "demand_published_at"] = late.loc[late.index[500], "decision_time"]
    result = audit(features, prices, late, gen, lag_minutes=LAG)
    assert result.loc["f_demand_forecast_mw", "verdict"] == "FAIL"


def test_nullable_dtypes_are_handled(spring):
    """Real parquet files can carry pandas' nullable types or object columns
    holding pd.NA. The audit must give the same answer as with plain floats."""
    features, prices, pit, gen = spring
    f2 = features.copy()
    for c in FEATURES:
        if f2[c].dtype.kind == "f":
            f2[c] = f2[c].astype(object).where(f2[c].notna(), pd.NA)
    p2 = prices.copy()
    p2["price_apx"] = p2["price_apx"].astype("Float64")
    pit2 = pit.copy()
    pit2["wind_forecast_mw"] = pit2["wind_forecast_mw"].astype("Float64")
    g2 = gen.copy()
    g2["gen_wind_mw"] = g2["gen_wind_mw"].astype(object)
    result = audit(f2, p2, pit2, g2, lag_minutes=LAG)
    assert (result["verdict"] == "PASS").all(), result[result["verdict"] != "PASS"]
    