"""
Elexon Insights Solution API client - Market Index Data (MID).
 
MID is what each appointed Market Index Data Provider (MIDP) reports to Elexon for
every settlement period: a Market Index Price and a Market Index Volume describing
trading in that provider's short-term GB market. The two providers are N2EX
("N2EXMIDP") and APX/EPEX ("APXMIDP"). The volume-weighted average across
providers is the "Market Price" that feeds the imbalance price calculation.
 
Naming note: this is NOT the day-ahead auction clearing price. It is a
short-term traded reference price for a settlement period, and is referred to
as such throughout this project.
 
API notes (current as of the developer portal at https://developer.data.elexon.co.uk/):
  - Base URL: https://data.elexon.co.uk/bmrs/api/v1
  - Endpoint: /balancing/pricing/market-index
  - No API key and no authorisation required.
  - `from`/`to` are RFC 3339 datetimes filtering on period start time, inclusive.
"""
 
from __future__ import annotations
 
import logging
from pathlib import Path
 
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from src.prep.calendar_gb import (
    LONDON,
    UTC,
    _next_local_midnight,
    attach_utc_index,
    expected_periods_in_day,
    settlement_period_grid,
)
 
log = logging.getLogger(__name__)
 
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
MID_ENDPOINT = "/balancing/pricing/market-index"
PROVIDERS = ("N2EXMIDP", "APXMIDP")
 
DEFAULT_RAW_DIR = Path("data/raw/mid")
 
# Maximum width of a single API query, in days. The Insights endpoints reject
# over-wide windows with HTTP 400. Conservative by design: the cost of a
# smaller window is a few extra HTTP requests behind a cache that is written
# once, which is negligible against the cost of a backfill failing part-way.
MAX_QUERY_DAYS = 7
 
 
def _session() -> requests.Session:
    """HTTP session with bounded retries on transient failures.
 
    Retries only on 429 and 5xx - i.e. "the server is busy or broken", which is
    worth retrying. A 400 or 404 means the request itself is wrong and retrying
    it just makes the same mistake five times more slowly, so those are excluded.
 
    backoff_factor=1.0 gives waits of 1s, 2s, 4s, 8s. A two-year backfill is
    ~24 requests; being polite costs seconds and avoids getting rate-limited.
    """
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Accept": "application/json", "User-Agent": "gb-bess-dispatch/0.1"})
    return s
 
 
def _month_starts(start_date: str, end_date: str) -> list[pd.Timestamp]:
    """Month-start timestamps covering the requested range."""
    s = pd.Timestamp(start_date).normalize().replace(day=1)
    e = pd.Timestamp(end_date).normalize()
    return list(pd.date_range(s, e, freq="MS"))
 
 
def _fetch_window(
    session: requests.Session, win_start: pd.Timestamp, win_end: pd.Timestamp
) -> pd.DataFrame:
    """One API call over a single UTC window.
 
    On a non-2xx response the server's own explanation is surfaced. The obvious
    alternative, `resp.raise_for_status()`, discards the response body - which is
    exactly where the API states what it objected to. A bare "400 Client Error"
    forces a guess; the body usually names the offending parameter outright.
    """
    params = {
        # %SZ keeps this RFC 3339 compliant with an explicit UTC offset.
        "from": win_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": win_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json",
    }
    resp = session.get(BASE_URL + MID_ENDPOINT, params=params, timeout=60)
 
    if not resp.ok:
        raise RuntimeError(
            f"Elexon returned HTTP {resp.status_code} for "
            f"{params['from']} .. {params['to']}\n"
            f"URL:  {resp.url}\n"
            f"Body: {resp.text[:1000]}"
        )
 
    payload = resp.json()
    # The Insights API wraps rows in {"data": [...]} when format=json. A bare
    # list is handled too, defensively: response envelopes change without notice.
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    return pd.DataFrame(rows)
 
 
def _fetch_month_raw(
    session: requests.Session,
    month_start: pd.Timestamp,
    chunk_days: int = MAX_QUERY_DAYS,
) -> pd.DataFrame:
    """All MID rows for one calendar month, assembled from windowed requests.
 
    Request window and cache granularity are deliberately decoupled. The API
    limits how wide a single query may be, but caching per calendar month keeps
    cache keys readable and the file count manageable. Coupling the two would
    mean either a cache file per week, or invalidating the whole cache whenever
    the server-side limit changed.
 
    Month boundaries are built in LOCAL time and then converted to UTC so the
    request covers exactly the settlement days in that month. A UTC-midnight
    window would clip an hour off each end during BST and silently lose two
    settlement periods per month.
    """
    month_end = month_start + pd.offsets.MonthEnd(0)
    cursor = pd.Timestamp(month_start.date(), tz=LONDON).tz_convert(UTC)
    month_stop = _next_local_midnight(month_end).tz_convert(UTC)
 
    frames = []
    while cursor < month_stop:
        cursor_end = min(cursor + pd.Timedelta(days=chunk_days), month_stop)
        log.info("GET %s  %s -> %s", MID_ENDPOINT, cursor, cursor_end)
        frames.append(_fetch_window(session, cursor, cursor_end))
        cursor = cursor_end
 
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"Elexon returned no MID rows for {month_start:%Y-%m}")
 
    required = {"dataProvider", "settlementDate", "settlementPeriod", "price", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Elexon MID response missing expected fields {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}. The API schema may have changed."
        )
 
    df = df[
        ["dataProvider", "settlementDate", "settlementPeriod", "price", "volume"]
    ].rename(
        columns={
            "dataProvider": "provider",
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
        }
    )
 
    # The API filters inclusively at both ends. That has two consequences.
    # First, consecutive windows overlap on their shared boundary, producing
    # duplicate rows. Second, the final window's `to` boundary returns one
    # period belonging to the NEXT month. Both are handled here rather than by
    # shrinking the request windows, which keeps the request arithmetic simple
    # and stays correct even if the endpoint's inclusivity changes.
    # _to_wide raises on duplicates, so this must precede it.
    in_month = (df["settlement_date"] >= str(month_start.date())) & (
        df["settlement_date"] <= str(month_end.date())
    )
    df = df.loc[in_month]
 
    return df.drop_duplicates(
        subset=["provider", "settlement_date", "settlement_period"], keep="first"
    ).reset_index(drop=True)
 
 
def _to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """Long (one row per provider per period) -> wide (one row per period).
 
    Pivoted rather than left long because the optimiser and every downstream
    join wants one row per settlement period. Keeping it long would mean every
    later merge silently doubles the row count.
    """
    raw = raw.copy()
    raw["settlement_date"] = pd.to_datetime(raw["settlement_date"]).dt.date
    raw["settlement_period"] = raw["settlement_period"].astype(int)
 
    # Duplicate (provider, date, period) rows would silently corrupt the pivot,
    # so fail loudly rather than let pivot_table quietly average them.
    dupes = raw.duplicated(["provider", "settlement_date", "settlement_period"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate provider/period rows in MID response")
 
    wide = raw.pivot(
        index=["settlement_date", "settlement_period"],
        columns="provider",
        values=["price", "volume"],
    )
    # Flatten the MultiIndex columns to e.g. price_n2ex, volume_apx
    wide.columns = [
        f"{val}_{prov.replace('MIDP', '').lower()}" for val, prov in wide.columns
    ]
    return wide.reset_index()
 
 
def _add_volume_weighted_price(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the volume-weighted Market Index Price across providers.
 
    This reconstructs the reference price Elexon itself uses. Two things to be
    explicit about, because both are modelling choices, not facts:
 
    1. If a provider is missing or reports zero volume for a period, that
       provider is simply excluded from the weighting - the VWAP falls back to
       whichever provider did report. If neither reported, the result is NaN and
       is left as NaN. Imputation is a separate, later, documented step; it does
       not get smuggled in here.
    2. A volume-weighted average of two exchanges is not a price anyone can
       transact at. It is a reference price, and using it encodes the assumption
       that the battery is a price taker achieving the market reference.
    """
    price_cols = [c for c in df.columns if c.startswith("price_")]
    vol_cols = [f"volume_{c.removeprefix('price_')}" for c in price_cols]
 
    prices = df[price_cols].to_numpy(dtype="float64")
    volumes = df[vol_cols].to_numpy(dtype="float64")
 
    # A provider with no price contributes no weight, and a NaN volume is zero.
    # Zeroing the weights (rather than dropping rows) keeps this a single
    # vectorised expression over the whole array - no per-row branching.
    valid = np.isfinite(prices) & np.isfinite(volumes)
    w = np.where(valid, volumes, 0.0)
    p = np.where(valid, prices, 0.0)
 
    total_vol = w.sum(axis=1)
    weighted = (p * w).sum(axis=1)
 
    # np.divide with `where` avoids a divide-by-zero warning and leaves NaN
    # wherever no provider reported volume. NaN is the honest answer there;
    # imputation is a separate, documented step, not something smuggled in here.
    vwap = np.divide(
        weighted, total_vol, out=np.full_like(weighted, np.nan), where=total_vol > 0
    )
 
    out = df.copy()
    out["price_mid_gbp_mwh"] = vwap
    out["volume_mid_mwh"] = total_vol
    return out
 
 
def fetch_mid(
    start_date: str,
    end_date: str,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch MID for a settlement-date range, cached one Parquet file per month.
 
    Caching is at the RAW layer, keyed by month, and is never invalidated
    automatically. Rationale: the API is free but slow-ish and rate-limited, and
    a reproducible backtest must run against a frozen snapshot of the data. If
    Elexon restates a settlement period, that must be a deliberate act
    (`force=True`), not something which silently changes the headline result
    between runs.
 
    Returns a DataFrame indexed by tz-aware UTC period start, reindexed onto the
    complete settlement-period grid so missing periods appear as NaN rows.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
 
    frames = []
    for m in _month_starts(start_date, end_date):
        cache_file = raw_dir / f"mid_{m:%Y-%m}.parquet"
        if cache_file.exists() and not force:
            log.info("cache hit  %s", cache_file.name)
            frames.append(pd.read_parquet(cache_file))
            continue
 
        log.info("cache miss %s - fetching", cache_file.name)
        raw = _fetch_month_raw(session, m)
        raw.to_parquet(cache_file, index=False)
        frames.append(raw)
 
    long_df = pd.concat(frames, ignore_index=True)
    wide = _add_volume_weighted_price(_to_wide(long_df))
    wide = attach_utc_index(wide)
 
    # --- validation, before the join ---------------------------------------
    # Check 1: every (date, period) label must be one that can actually exist.
    # This is the check that has to come FIRST, because an out-of-range period
    # does not stay out of range once it becomes a timestamp: SP50 on a normal
    # 48-period day maps to a real half-hour belonging to the NEXT settlement
    # day, where it silently collides with that day's SP2. Catching it at the
    # label stage is the only place the error is still visible as itself.
    max_periods = {
        d: expected_periods_in_day(d) for d in wide["settlement_date"].unique()
    }
    out_of_range = wide["settlement_period"] > wide["settlement_date"].map(max_periods)
    if out_of_range.any():
        bad = wide.loc[out_of_range, ["settlement_date", "settlement_period"]]
        raise ValueError(
            f"{len(bad)} settlement periods outside the valid range for their "
            f"settlement day. First examples:\n{bad.head().to_string()}"
        )
 
    # Check 2: no duplicate half-hours. Catches genuine duplicate rows from the
    # API, and acts as a backstop for any aliasing check 1 failed to anticipate.
    if wide.index.duplicated().any():
        dupes = wide.index[wide.index.duplicated()]
        raise ValueError(
            f"{len(dupes)} duplicate settlement periods after indexing. "
            f"First examples: {dupes[:5].tolist()}"
        )
 
    # Check 3: nothing outside the span we actually fetched. Note this is the
    # FETCHED span (whole calendar months), not the requested one - the fetcher
    # chunks monthly, so `wide` legitimately overshoots the requested window at
    # both ends. Validating against the requested grid instead would raise on
    # every correct fetch whose dates are not month boundaries.
    fetched_start = pd.Timestamp(start_date).replace(day=1)
    fetched_end = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    fetched_grid = settlement_period_grid(
        str(fetched_start.date()), str(fetched_end.date())
    )
    unexpected = wide.index.difference(fetched_grid.index)
    if len(unexpected):
        raise ValueError(
            f"{len(unexpected)} periods fall outside the fetched settlement "
            f"grid. First examples: {unexpected[:5].tolist()}"
        )
 
    # --- join onto the requested grid ---------------------------------------
    # Left join FROM the authoritative grid. Any period the API failed to
    # return becomes an all-NaN row rather than a silent gap in the series.
    grid = settlement_period_grid(start_date, end_date)
    data_cols = [
        c for c in wide.columns
        if c not in ("settlement_date", "settlement_period", "start_time_local")
    ]
    out = grid.join(wide[data_cols], how="left")
 
    # Check 4: a left join from a unique index onto a unique index cannot change
    # the row count. If it did, one of the checks above has a hole in it.
    if len(out) != len(grid):
        raise ValueError(
            f"join changed row count: grid={len(grid)}, result={len(out)}. "
            "This means duplicate timestamps survived validation."
        )
 
    return out
 
 
def mid_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-check summary of MID data quality. Print this every fetch.
 
    Answers the questions the data-quality policy has to address: how many
    periods are missing, how often each provider is absent, and how lopsided the
    volume split between the two providers actually is.
    """
    n = len(df)
    checks = {
        "periods_expected": n,
        "price_mid_missing": int(df["price_mid_gbp_mwh"].isna().sum()),
    }
    for col in [c for c in df.columns if c.startswith("price_") and c != "price_mid_gbp_mwh"]:
        checks[f"{col}_missing"] = int(df[col].isna().sum())
 
    vol_cols = [c for c in df.columns if c.startswith("volume_") and c != "volume_mid_mwh"]
    total = df[vol_cols].sum().sum()
    for c in vol_cols:
        checks[f"{c}_share_pct"] = round(100 * df[c].sum() / total, 2) if total else float("nan")
 
    checks["price_min"] = round(float(df["price_mid_gbp_mwh"].min()), 2)
    checks["price_max"] = round(float(df["price_mid_gbp_mwh"].max()), 2)
    checks["price_mean"] = round(float(df["price_mid_gbp_mwh"].mean()), 2)
    checks["negative_price_periods"] = int((df["price_mid_gbp_mwh"] < 0).sum())
 
    return pd.DataFrame({"check": checks.keys(), "value": checks.values()})
 