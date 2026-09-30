"""Tests for the SQL feature frame.

The synthetic price of each half hour is its own start time, in hours since an
epoch. A feature's value then says exactly which half hour it came from, which
turns "no look-ahead" into an arithmetic check rather than a plausibility argument.
"""

import pandas as pd
import pytest

from src.db.duck import connect, query_file, register_calendar, register_decisions
from src.prep.calendar_gb import settlement_period_grid
from src.prep.features import FEATURES, check_feature_frame

EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
LAG = 60
CAP = 20_000.0


def _hours(ts):
    return (pd.to_datetime(ts, utc=True) - EPOCH) / pd.Timedelta(hours=1)


def _decode(hours):
    return EPOCH + pd.to_timedelta(hours, unit="h")


def _build(tmp_path, start, end, lag=LAG, interpolated=()):
    """Write synthetic processed files for [start, end] and return the feature frame.

    `interpolated` lists UTC start times whose price is flagged as filled."""
    proc = tmp_path / "processed"
    proc.mkdir()
    grid = settlement_period_grid(start, end).reset_index()
    t = grid["start_time_utc"]
    day = pd.to_datetime(grid["settlement_date"])

    prices = pd.DataFrame(
        {"price_apx": _hours(t).to_numpy(), "period_usable": True, "day_usable": True},
        index=pd.DatetimeIndex(t, name="start_time_utc"),
    )
    prices["price_apx_interpolated"] = prices.index.isin(
        pd.DatetimeIndex([pd.Timestamp(x, tz="UTC") for x in interpolated])
    )
    prices.to_parquet(proc / "prices_full.parquet")

    def forecast(value_col, value, hour_utc):
        return pd.DataFrame(
            {
                "target_time": t,
                "settlement_date": day,
                "settlement_period": grid["settlement_period"],
                "published_at": (day - pd.Timedelta(days=1) + pd.Timedelta(hours=hour_utc))
                .dt.tz_localize("UTC"),
                "published_at_source": "archive",
                "published_at_suspect": False,
                value_col: value,
            }
        )

    wind = forecast("wind_forecast_mw", 5_000.0, 7)
    wind["wind_capacity_mw"] = CAP
    wind["wind_forecast_share"] = 5_000.0 / CAP
    wind.to_parquet(proc / "wind_forecasts.parquet", index=False)
    forecast("demand_forecast_mw", 30_000.0, 9).to_parquet(
        proc / "demand_forecasts.parquet", index=False
    )

    # Outturn 0.1 of capacity below forecast: every daily error is +0.1.
    gen = pd.DataFrame(
        {"gen_wind_mw": 5_000.0 - 0.1 * CAP}, index=pd.DatetimeIndex(t, name="start_time_utc")
    )
    gen.to_parquet(proc / "fuelhh.parquet")

    con = connect(tmp_path)
    register_decisions(con, start, end)
    query_file(con, "point_in_time").to_parquet(proc / "point_in_time.parquet", index=False)

    con = connect(tmp_path, required=("point_in_time", "prices", "generation"))
    register_calendar(con, start, end)
    return query_file(con, "features", {"lag_minutes": lag})


@pytest.fixture(scope="module")
def spring(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("spring"), "2024-03-01", "2024-04-06")


@pytest.fixture(scope="module")
def autumn(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("autumn"), "2023-10-20", "2023-11-02")


def _bound(frame):
    """Latest start time a source half hour may have: it must end, plus the lag,
    strictly before the decision."""
    return frame["decision_time"] - pd.Timedelta(minutes=30 + LAG)


@pytest.mark.parametrize("col", ["f_price_d2_same_time", "f_price_d7_same_time", "f_last_known_price"])
def test_every_price_lag_was_published_before_the_decision(spring, col):
    f = spring[spring[col].notna()]
    assert len(f) > 0
    assert (_decode(f[col]) < _bound(f)).all()


@pytest.mark.parametrize(
    "col", ["f_price_d2_mean", "f_price_7d_mean", "f_price_7d_same_time_mean"]
)
def test_price_averages_use_only_published_prices(spring, col):
    """An average of timestamps below the bound can only come from sources below it
    if the latest source is below it; the mean is a necessary check, the window
    tests below make it sufficient."""
    f = spring[spring[col].notna()]
    assert (_decode(f[col]) < _bound(f)).all()


def test_last_known_price_is_the_latest_published(spring):
    """The next half hour after the chosen one must not yet have been public."""
    f = spring[spring["f_last_known_price"].notna()]
    nxt = _decode(f["f_last_known_price"]) + pd.Timedelta(minutes=30)
    assert (nxt >= _bound(f)).all()


def test_d2_lag_matches_local_clock_across_spring_forward(spring):
    """2024-04-01 (BST) looks back to 2024-03-30 (GMT). 48 hours in UTC would land
    an hour off; the lag must land on the same local clock time."""
    f = spring[spring["settlement_date"] == pd.Timestamp("2024-04-01")]
    src = _decode(f["f_price_d2_same_time"]).dt.tz_convert("Europe/London")
    tgt = f["target_time"].dt.tz_convert("Europe/London")
    assert (src.dt.strftime("%H:%M") == tgt.dt.strftime("%H:%M")).all()
    assert (src.dt.date == pd.Timestamp("2024-03-30").date()).all()


def test_d2_lag_is_null_for_clock_times_that_did_not_exist(spring):
    """D-2 = 2024-03-31 has no 01:00-02:00 local, so two target half hours on
    2024-04-02 have no same-time source."""
    f = spring[spring["settlement_date"] == pd.Timestamp("2024-04-02")]
    assert f["f_price_d2_same_time"].isna().sum() == 2


def test_d2_spread_covers_one_whole_day(spring):
    """With price = hours, a 48-period day's max minus min is 23.5."""
    f = spring[spring["settlement_date"] == pd.Timestamp("2024-03-20")]
    assert f["f_price_d2_spread"].round(6).eq(23.5).all()


def test_repeated_autumn_hour_does_not_fan_out(autumn):
    assert autumn["target_time"].is_unique
    assert (autumn.groupby("settlement_date").size() == 48).sum() >= 12
    assert (autumn.groupby("settlement_date").size().loc[pd.Timestamp("2023-10-29")]) == 50


def test_repeated_autumn_hour_uses_first_pass_as_source(autumn):
    """D-2 = 2023-10-29 has 01:00 twice; the lag takes the first (BST) one."""
    f = autumn[autumn["settlement_date"] == pd.Timestamp("2023-10-31")]
    src = _decode(f["f_price_d2_same_time"])
    one_am = src[src.dt.tz_convert("Europe/London").dt.strftime("%H:%M") == "01:00"]
    assert len(one_am) == 1
    assert one_am.iloc[0] == pd.Timestamp("2023-10-29 00:00", tz="UTC")


def test_too_long_a_lag_gives_nulls_not_leaks(tmp_path):
    """With a 40-hour publication lag, D-2 is not yet public at the decision."""
    f = _build(tmp_path, "2024-03-01", "2024-03-20", lag=40 * 60)
    tail = f[f["settlement_date"] >= pd.Timestamp("2024-03-12")]
    assert tail["f_price_d2_same_time"].isna().all()
    assert tail["f_price_d7_same_time"].notna().all()


def test_interpolated_price_is_never_a_source(tmp_path):
    """2024-03-19 is in GMT, so the decision for 2024-03-20 is 11:00 UTC and the
    latest published price is the one starting 09:00 (it ends 09:30, public 10:30).
    Flag it as interpolated: the feature must fall back to 08:30, and the D-2 lag
    for that clock time must go missing."""
    filled = "2024-03-19 09:00"
    f = _build(tmp_path, "2024-03-01", "2024-03-22", interpolated=[filled])
    day = f[f["settlement_date"] == pd.Timestamp("2024-03-20")]
    latest = pd.Timestamp("2024-03-19 08:30", tz="UTC")
    assert (_decode(day["f_last_known_price"]) == latest).all()
    d2 = f[f["settlement_date"] == pd.Timestamp("2024-03-21")]
    at_0900 = d2[d2["target_time"] == pd.Timestamp("2024-03-21 09:00", tz="UTC")]
    assert len(at_0900) == 1
    assert at_0900["f_price_d2_same_time"].isna().all()


def test_wind_error_and_bias(spring):
    f = spring[spring["settlement_date"] >= pd.Timestamp("2024-03-10")]
    assert f["f_wind_err_d2"].round(9).eq(0.1).all()
    assert f["f_wind_bias_28d"].round(9).eq(0.1).all()
    assert f["f_wind_err_d2_demeaned"].abs().max() < 1e-9


def test_wind_error_absent_without_history(spring):
    first_days = spring[spring["settlement_date"] <= pd.Timestamp("2024-03-02")]
    assert first_days["f_wind_err_d2"].isna().all()


def test_residual_demand(spring):
    assert spring["f_residual_demand_mw"].eq(25_000.0).all()


def test_sql_output_matches_the_register(spring):
    check_feature_frame(spring, len(spring))
    assert {c for c in spring.columns if c.startswith("f_")} == set(FEATURES)


def test_register_rejects_an_unregistered_feature(spring):
    with pytest.raises(ValueError, match="unregistered"):
        check_feature_frame(spring.assign(f_secret=1.0), len(spring))


def test_register_rejects_outturn_columns(spring):
    with pytest.raises(ValueError, match="outturn"):
        check_feature_frame(spring.assign(demand_outturn_mw=1.0), len(spring))


def test_register_rejects_wrong_row_count(spring):
    with pytest.raises(ValueError, match="rows"):
        check_feature_frame(spring, len(spring) + 1)
        