"""Tests for the FUELHH client. No network: a fake session stands in for the API."""

import pandas as pd
import pytest

from src.fetch.elexon_fuelhh import date_windows, fetch_month_raw, to_wide, validate_and_grid


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


def test_start_time_is_never_used_to_reassign_a_row():
    """Even a startTime that names the missing period does not move the value.

    Elexon's startTime is the less reliable field in this archive (24 hours
    early on every SP48 in the first half of 2022), so it is not trusted to
    override a label.
    """
    rows = [r for r in _full_day("2022-03-27", 46) if r["settlementPeriod"] != 46]
    rows.append(_row("2022-03-27", 48, "WIND", 777, start="2022-03-27T22:30:00Z"))
    out = validate_and_grid(to_wide(pd.DataFrame(rows)), "2022-03-27", "2022-03-27")
    assert out.loc[out["settlement_period"] == 46, "gen_wind_mw"].isna().item()
    assert (out["gen_wind_mw"] == 777).sum() == 0


def test_real_archive_row_is_quarantined():
    """The real 2022-03-27 row: labelled SP48, startTime 2022-03-26 23:30 UTC.

    The row is dropped, SP46 stays NaN, and the previous day's SP48, which the
    startTime happens to name, is untouched.
    """
    rows = _full_day("2022-03-26", 48)
    rows += [r for r in _full_day("2022-03-27", 46) if r["settlementPeriod"] != 46]
    rows.append(_row("2022-03-27", 48, "WIND", 999, start="2022-03-26T23:30:00Z"))
    out = validate_and_grid(to_wide(pd.DataFrame(rows)), "2022-03-26", "2022-03-27")
    assert len(out) == 48 + 46
    assert out["gen_wind_mw"].isna().sum() == 1
    day27 = out[out["settlement_date"] == pd.Timestamp("2022-03-27")]
    assert day27.loc[day27["settlement_period"] == 46, "gen_wind_mw"].isna().item()
    assert (out["gen_wind_mw"] == 999).sum() == 0


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
