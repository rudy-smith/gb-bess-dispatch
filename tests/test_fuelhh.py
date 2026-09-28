"""Tests for the FUELHH client. No network: a fake session stands in for the API."""

import pandas as pd
import pytest

from src.fetch.elexon_fuelhh import (
    date_windows,
    fetch_month_raw,
    labels_from_start_time,
    to_wide,
    validate_and_grid,
)


def _row(date, sp, fuel, gen, publish="2023-06-02T00:00:00Z", start=None):
    return {
        "dataset": "FUELHH",
        "publishTime": publish,
        "startTime": start,
        "settlementDate": date,
        "settlementPeriod": sp,
        "fuelType": fuel,
        "generation": gen,
    }


class _FakeResponse:
    ok = True
    status_code = 200
    url = "fake"
    text = ""

    def __init__(self, rows):
        self._rows = rows

    def json(self):
        return {"data": self._rows}


class _FakeSession:
    """Records the parameters of every request and returns one row per day."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        days = pd.date_range(params["settlementDateFrom"], params["settlementDateTo"])
        return _FakeResponse([_row(f"{d:%Y-%m-%d}", 1, "WIND", 100) for d in days])


@pytest.mark.parametrize("month", ["2023-02-01", "2024-02-01", "2023-10-01", "2023-06-01"])
def test_windows_cover_the_month_exactly_once(month):
    """Every settlement day in the month appears in exactly one window, no more."""
    windows = date_windows(pd.Timestamp(month))
    covered = pd.DatetimeIndex([])
    for a, b in windows:
        assert (pd.Timestamp(b) - pd.Timestamp(a)).days <= 6
        covered = covered.append(pd.date_range(a, b))
    expected = pd.date_range(month, pd.Timestamp(month) + pd.offsets.MonthEnd(0))
    assert covered.equals(expected)


def test_fetch_uses_settlement_date_parameters():
    """Filters are settlement-date labels, not UTC instants."""
    session = _FakeSession()
    raw = fetch_month_raw(session, pd.Timestamp("2023-03-01"))
    assert set(session.calls[0]) == {"settlementDateFrom", "settlementDateTo", "format"}
    assert raw["settlementDate"].nunique() == 31
    assert not raw.duplicated(["settlementDate", "settlementPeriod", "fuelType"]).any()


def test_pivot_names_columns_by_fuel():
    raw = pd.DataFrame([_row("2023-06-01", 1, "WIND", 5000), _row("2023-06-01", 1, "CCGT", 9000)])
    wide = to_wide(raw)
    assert wide.loc[0, "gen_wind_mw"] == 5000
    assert wide.loc[0, "gen_ccgt_mw"] == 9000


def test_revision_keeps_latest_publication():
    """A restated value supersedes the original, whatever order the rows arrive in."""
    raw = pd.DataFrame(
        [
            _row("2023-06-01", 1, "WIND", 7000, publish="2023-06-05T00:00:00Z"),
            _row("2023-06-01", 1, "WIND", 5000, publish="2023-06-01T00:00:00Z"),
        ]
    )
    assert to_wide(raw).loc[0, "gen_wind_mw"] == 7000


def _full_day(date, n, fuel="WIND"):
    return [_row(date, sp, fuel, 100) for sp in range(1, n + 1)]


def test_stray_period_on_a_short_day_is_quarantined_not_merged():
    """The case seen in the real archive: SP48 on the 46-period 2022-03-27.

    Every valid period is present, so the stray row is surplus. It must be
    dropped, and in particular must not become a half hour of the next day.
    """
    rows = _full_day("2022-03-27", 46) + [_row("2022-03-27", 48, "WIND", 999)]
    out = validate_and_grid(to_wide(pd.DataFrame(rows)), "2022-03-27", "2022-03-28")
    assert len(out) == 46 + 48
    assert (out["gen_wind_mw"] == 999).sum() == 0


def test_stray_period_without_start_time_leaves_a_visible_gap():
    """No start time: the row cannot be identified, so it is dropped and the
    missing period stays NaN rather than receiving a guessed value."""
    rows = [r for r in _full_day("2022-03-27", 46) if r["settlementPeriod"] != 30]
    rows.append(_row("2022-03-27", 48, "WIND", 999))
    out = validate_and_grid(to_wide(pd.DataFrame(rows)), "2022-03-27", "2022-03-27")
    assert len(out) == 46
    assert out.loc[out["settlement_period"] == 30, "gen_wind_mw"].isna().item()
    assert (out["gen_wind_mw"] == 999).sum() == 0


@pytest.mark.parametrize(
    "start, date, sp",
    [
        ("2023-06-01T22:30:00Z", "2023-06-01", 48),  # BST: last period of the day
        ("2023-06-01T23:00:00Z", "2023-06-02", 1),  # BST: local midnight
        ("2024-01-15T23:30:00Z", "2024-01-15", 48),  # GMT
        ("2024-03-31T01:00:00Z", "2024-03-31", 3),  # just after spring forward
        ("2023-10-29T01:00:00Z", "2023-10-29", 5),  # second 01:00 local, autumn
        ("2023-10-29T23:30:00Z", "2023-10-29", 50),  # last period of a 50-period day
    ],
)
def test_labels_from_start_time(start, date, sp):
    d, p = labels_from_start_time(pd.Series([start]))
    assert d.iloc[0] == pd.Timestamp(date)
    assert p.iloc[0] == sp


def _day_without_sp48(date):
    return [r for r in _full_day(date, 48) if r["settlementPeriod"] != 48]


def test_day_early_sp48_is_moved_to_its_start_time():
    """The 2022 fault: a row labelled D SP48 holds D-1 SP48, per its startTime."""
    rows = _day_without_sp48("2022-01-02") + _day_without_sp48("2022-01-03")
    rows.append(_row("2022-01-02", 48, "WIND", 11, start="2022-01-01T23:30:00Z"))
    rows.append(_row("2022-01-03", 48, "WIND", 22, start="2022-01-02T23:30:00Z"))
    wide = to_wide(pd.DataFrame(rows))
    wide = wide[wide["settlement_date"] >= pd.Timestamp("2022-01-02")]
    out = validate_and_grid(wide, "2022-01-02", "2022-01-03")
    at = out.set_index(["settlement_date", "settlement_period"])["gen_wind_mw"]
    assert at[(pd.Timestamp("2022-01-02"), 48)] == 22
    assert pd.isna(at[(pd.Timestamp("2022-01-03"), 48)])
    assert (out["gen_wind_mw"] == 11).sum() == 0  # moved to 2022-01-01, outside range


def test_real_archive_row_moves_to_the_previous_day():
    """2022-03-27 "SP48" with startTime 2022-03-26 23:30 UTC is 2022-03-26 SP48.

    2022-03-27 SP46 has no source row, so it stays NaN rather than being filled.
    """
    rows = _day_without_sp48("2022-03-26")
    rows += [r for r in _full_day("2022-03-27", 46) if r["settlementPeriod"] != 46]
    rows.append(_row("2022-03-27", 48, "WIND", 999, start="2022-03-26T23:30:00Z"))
    out = validate_and_grid(to_wide(pd.DataFrame(rows)), "2022-03-26", "2022-03-27")
    at = out.set_index(["settlement_date", "settlement_period"])["gen_wind_mw"]
    assert at[(pd.Timestamp("2022-03-26"), 48)] == 999
    assert pd.isna(at[(pd.Timestamp("2022-03-27"), 46)])


def test_rekey_collision_raises():
    """Two rows claiming one half hour: choosing between them would be a guess."""
    rows = _day_without_sp48("2022-01-02")
    rows.append(_row("2022-01-02", 48, "WIND", 1, start="2022-01-02T23:30:00Z"))
    rows.append(_row("2022-01-03", 48, "WIND", 2, start="2022-01-02T23:30:00Z"))
    with pytest.raises(ValueError, match="same half hour"):
        to_wide(pd.DataFrame(rows))


def test_widespread_disagreement_raises():
    """A one-hour shift on every row is a convention change, not the known fault."""
    rows = [
        _row("2023-06-01", sp, "WIND", 1,
             start=f"{pd.Timestamp('2023-05-31T23:00Z') + pd.Timedelta(minutes=30 * sp):%Y-%m-%dT%H:%M:%SZ}")
        for sp in range(1, 49)
    ]
    with pytest.raises(ValueError, match="tolerance"):
        to_wide(pd.DataFrame(rows))


def test_many_stray_periods_raise():
    """Beyond the tolerance, stray labels indicate a systematic fault."""
    rows = []
    for d in pd.date_range("2023-06-01", periods=11):
        rows += _full_day(f"{d:%Y-%m-%d}", 48) + [_row(f"{d:%Y-%m-%d}", 50, "WIND", 1)]
    with pytest.raises(ValueError, match="tolerance"):
        validate_and_grid(to_wide(pd.DataFrame(rows)), "2023-06-01", "2023-06-11")


def test_sp50_is_valid_on_the_autumn_clock_change():
    raw = pd.DataFrame([_row("2023-10-29", 50, "WIND", 1)])
    out = validate_and_grid(to_wide(raw), "2023-10-29", "2023-10-29")
    assert len(out) == 50
    assert out["gen_wind_mw"].notna().sum() == 1


def test_missing_periods_become_nan_rows():
    raw = pd.DataFrame([_row("2024-03-31", 1, "WIND", 1)])
    out = validate_and_grid(to_wide(raw), "2024-03-31", "2024-03-31")
    assert len(out) == 46
    assert out["gen_wind_mw"].isna().sum() == 45
