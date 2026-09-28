"""
NESO Data Portal API client - day-ahead wind and demand forecasts.
 
NESO publishes its operational forecasts through a CKAN DataStore. Each dataset
is addressed by an opaque resource id and queried through a single endpoint,
which returns both the rows and a description of the fields those rows contain.
 
API notes (current as of the portal at https://www.neso.energy/data-portal):
  - Base URL: https://api.neso.energy/api/3/action
  - Endpoints: /datastore_search (paged), /datastore_search_sql (SQL over one
    resource), /datapackage_show (dataset metadata)
  - No API key and no authorisation required.
  - datastore_search takes `limit` and `offset`; the response carries
    `result.total`, so the number of pages is known after the first request.
 
The vintage problem
-------------------
A forecast is not one number. It is a statement made at a particular moment
about a particular future half hour, and only the statements made before a
decision may inform that decision. Two timestamps are therefore needed for
every row: the half hour the forecast is *for*, and the moment it became
*available*.
 
The historic archives carry only the first. They record a target date and
settlement period, with no publication timestamp, so availability cannot be
read from the data and must be imputed from NESO's published schedule:
 
  - Day-ahead wind forecast: daily, between 09:00 and 09:15 clock time.
  - Day-ahead national demand forecast: twice daily, between 09:00 and 09:15
    and between 12:00 and 12:15 clock time.
 
The imputation takes the later edge of the publication window. If a forecast
actually appeared at 09:03, treating it as available at 09:15 discards twelve
minutes of information that was legitimately available, which understates what
a strategy could have known. The opposite rounding would manufacture
look-ahead, and the two errors are not symmetric in their consequences: one
costs a little revenue in the backtest, the other invalidates it.
 
The imputed timestamp is written as a column rather than applied inside a join,
so that any downstream query can be audited against it and the assumption is
visible in the stored data.
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass
from datetime import time
from pathlib import Path
 
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from src.prep.calendar_gb import LONDON, UTC
 
LOGGER = logging.getLogger(__name__)
 
BASE_URL = "https://api.neso.energy/api/3/action"
 
# CKAN caps a single datastore_search page. 32,000 is below the server limit and
# keeps each response small enough to parse without excessive memory use.
PAGE_SIZE = 32_000
 
# Guard against an unbounded loop if `total` is missing or a cursor fails to
# advance. Two years of half-hourly rows is roughly 35,000, so this is far above
# any legitimate requirement for this project.
MAX_PAGES = 200
 
 
@dataclass(frozen=True)
class NesoResource:
    """One CKAN resource, with the publication schedule that dates its rows.
 
    `timestamp_timezone` is how that column's naive values are read. It is a
    per-resource property rather than a shared constant because the two
    archives were populated by different systems and cannot be assumed to
    agree. Where the reading is unverified, the conservative choice is the one
    that places publication later in UTC, since that withholds information
    rather than granting it early.
 
    `timestamp_column` names the archive's own publication timestamp where one
    exists. It is preferred over the imputed schedule, because a recorded
    instant beats an assumed one. It is not trusted blindly: see
    `resolve_published_at`.
 
    `publication_local_time` is the later edge of NESO's stated publication
    window, in clock time. `lead_days` is how far ahead of the target date that
    publication happens: 1 for a day-ahead forecast published the day before.
 
    Holding these beside the resource id keeps the vintage assumption attached
    to the dataset it belongs to. A single shared constant would silently apply
    the wind schedule to demand, which publishes twice and on a different
    cadence.
    """
 
    name: str
    resource_id: str
    publication_local_time: time
    lead_days: int
    timestamp_column: str | None
    timestamp_timezone: str
    description: str
 
 
# Both archives run from 2018 to the most recent forecast, which covers the
# 2023-24 study period without stitching together per-year resources.
RESOURCES: dict[str, NesoResource] = {
    "wind_da": NesoResource(
        name="wind_da",
        resource_id="7524ec65-f782-4258-aaf8-5b926c17b966",
        publication_local_time=time(9, 15),
        lead_days=1,
        timestamp_column="Forecast_Timestamp",
        # Verified against the archive rather than the documentation. Read as
        # local clock time the publication hour is stable across the year
        # (+0.03h winter to summer); read as UTC it shifts by a full hour,
        # which is the signature of a reading carrying the daylight-saving
        # offset. Publication clusters near 07:23 local, which is not the
        # 09:00-09:15 window NESO documents for the live day-ahead feed; the
        # historic archive is written on its own schedule.
        timestamp_timezone="Europe/London",
        description="Historic day-ahead wind forecasts, half-hourly, 2018 onwards",
    ),
    # The 1-day-ahead demand archive (9847e7bb) is published in cardinal-point
    # format: roughly 19 characteristic peaks and troughs per day rather than a
    # half-hourly profile. Interpolating it up to 48 periods would invent a
    # demand curve, so the half-hourly performance file is used instead. It
    # begins in April 2021, which still covers the 2023-24 study period.
    #
    # That file also ships demand outturn beside the forecast. The outturn is
    # not information available at the decision time and must never reach a
    # feature set; it is kept only as the target for forecast-error analysis.
    "demand_da_hh": NesoResource(
        name="demand_da_hh",
        resource_id="08e41551-80f8-4e28-a416-ea473a695db9",
        publication_local_time=time(9, 15),
        lead_days=1,
        timestamp_column="Publish_Datetime",
        # Unresolved. The stability test fails to separate the readings: the
        # publication hour shifts by two hours across the year read as UTC and
        # by one read as local, so neither is a fixed schedule and the naive
        # value carries a seasonal pattern of its own. UTC is retained because
        # it is the later of the two in UTC terms and therefore withholds
        # information rather than granting it early.
        #
        # The ambiguity is bounded and does not reach this design: lead time is
        # a constant 13.25 hours with zero variance, so one hour of uncertainty
        # cannot bring publication inside the settlement day. It would matter
        # at an intraday horizon, and is recorded as a limitation on that
        # basis rather than resolved by assumption.
        timestamp_timezone="UTC",
        description="Day-ahead half-hourly demand forecast performance, 2021 onwards",
    ),
}
 
 
def _session(retries: int = 3) -> requests.Session:
    """HTTP session retrying only on transient server-side failures.
 
    Retrying a 4xx would repeat a request the server has already rejected on
    its merits. The status list is restricted to conditions that can plausibly
    succeed on a second attempt.
    """
    session = requests.Session()
    policy = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=policy))
    return session
 
 
def _get(session: requests.Session, action: str, params: dict) -> dict:
    """Call one CKAN action and return `result`, surfacing the response body.
 
    `raise_for_status` discards the body, which is where CKAN explains what it
    objected to. The body is read first and included in the raised message, so
    a rejected request says why rather than only which status code it returned.
    """
    response = session.get(f"{BASE_URL}/{action}", params=params, timeout=60)
 
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{action} returned non-JSON with status {response.status_code}: "
            f"{response.text[:500]}"
        ) from exc
 
    if not response.ok or not payload.get("success", False):
        error = payload.get("error", payload)
        raise RuntimeError(f"{action} failed with status {response.status_code}: {error}")
 
    return payload["result"]
 
 
def describe_resource(resource_id: str, sample_rows: int = 3) -> dict:
    """Return the field definitions and a few rows for one resource.
 
    Schemas differ between NESO datasets and are not documented uniformly, so
    the column mapping is discovered from the API rather than assumed. Guessing
    column names produces a normaliser that fails on live data at the point
    where the failure is least informative.
    """
    session = _session()
    result = _get(
        session,
        "datastore_search",
        {"resource_id": resource_id, "limit": sample_rows},
    )
    return {
        "total": result.get("total"),
        "fields": result.get("fields", []),
        "records": result.get("records", []),
    }
 
 
def fetch_resource(resource_id: str, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """Page through a resource and return every row, unmodified.
 
    Nothing is renamed, parsed or coerced here. Keeping retrieval separate from
    interpretation means a schema change shows up as an unexpected column in a
    stored file rather than as a KeyError in the middle of a fetch, and the
    cached raw data stays a faithful copy of what the API served.
    """
    session = _session()
    frames: list[pd.DataFrame] = []
    offset = 0
    total: int | None = None
 
    for page in range(MAX_PAGES):
        result = _get(
            session,
            "datastore_search",
            {"resource_id": resource_id, "limit": page_size, "offset": offset},
        )
        records = result.get("records", [])
        if total is None:
            total = result.get("total")
            LOGGER.info("resource %s reports %s rows", resource_id, total)
 
        if not records:
            break
 
        frames.append(pd.DataFrame.from_records(records))
        offset += len(records)
        LOGGER.info("fetched %s of %s rows", offset, total)
 
        if total is not None and offset >= total:
            break
    else:
        raise RuntimeError(
            f"resource {resource_id} exceeded {MAX_PAGES} pages; "
            "paging is not terminating and the request should be narrowed"
        )
 
    if not frames:
        return pd.DataFrame()
 
    out = pd.concat(frames, ignore_index=True)
 
    # CKAN adds an internal row id that is meaningless outside its own store and
    # would otherwise be mistaken for data.
    return out.drop(columns=["_id"], errors="ignore")
 
 
def impute_published_at(
    target_dates: pd.Series,
    resource: NesoResource,
) -> pd.Series:
    """Return the UTC instant at which each row's forecast became available.
 
    The archive has no publication timestamp, so availability is derived from
    the target date, the documented lead time and the later edge of the
    documented publication window.
 
    The publication time is a clock time, so it is constructed in local time and
    then converted, never by subtracting a fixed number of hours from UTC. A
    09:15 publication is 08:15 UTC in summer and 09:15 UTC in winter, and an
    offset applied uniformly would be an hour wrong for half the year, in the
    direction that grants information before it existed.
 
    `nonexistent` and `ambiguous` are handled explicitly rather than left to
    default. 09:15 never falls inside a clock-change gap in Britain, where the
    transitions occur at 01:00, so neither case can arise here; naming them
    documents that this was checked rather than overlooked, and makes the code
    fail loudly if it is later reused for a publication time near midnight.
    """
    dates = pd.to_datetime(target_dates).dt.normalize()
    publication_dates = dates - pd.Timedelta(days=resource.lead_days)
 
    local = pd.to_datetime(
        publication_dates.dt.strftime("%Y-%m-%d")
        + " "
        + resource.publication_local_time.strftime("%H:%M:%S")
    )
    localised = local.dt.tz_localize(LONDON, nonexistent="raise", ambiguous="raise")
    return localised.dt.tz_convert(UTC)
 
 
def cache_path(root: Path, name: str) -> Path:
    return Path(root) / "neso" / f"{name}.parquet"
 
 
def fetch_and_cache(
    resource: NesoResource,
    raw_root: Path = Path("data/raw"),
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a resource, or read it back from the Parquet cache.
 
    The archives are append-only historical records, so a cached copy stays
    valid for a backtest over a closed period. Refetching two years of rows on
    every run would be slow and would make results depend on when they were
    produced.
    """
    path = cache_path(raw_root, resource.name)
 
    if path.exists() and not refresh:
        LOGGER.info("reading cached %s from %s", resource.name, path)
        return pd.read_parquet(path)
 
    LOGGER.info("fetching %s (%s)", resource.name, resource.description)
    frame = fetch_resource(resource.resource_id)
 
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("wrote %s rows to %s", len(frame), path)
    return frame
 
 
def resolve_published_at(
    frame: pd.DataFrame,
    resource: NesoResource,
    target_date_column: str,
    day_start_utc: pd.Series,
) -> pd.DataFrame:
    """Attach a trusted publication instant, and say where it came from.
 
    Three columns are added. `published_at` is the instant the forecast is
    treated as available. `published_at_source` records whether that came from
    the archive or from the publication schedule. `published_at_suspect` marks
    rows whose recorded timestamp failed validation.
 
    The archive's own timestamp is preferred, but it is validated rather than
    trusted. The historic wind archive contains rows whose recorded timestamp
    falls after the settlement day they forecast, which is not a forecast at
    all: the earliest rows appear to carry the instant the archive itself was
    written rather than the instant the forecast was produced. A backtest that
    consumed those rows would be reading the outturn.
 
    The test is the only one that matters for look-ahead: a forecast is usable
    only if it was published strictly before the first settlement period it
    describes. Rows failing it fall back to the imputed schedule and are
    flagged, so a caller can drop them rather than silently inherit a repair.
 
    Timestamps in these archives are naive, and how they are read is a
    per-resource setting (`timestamp_timezone`), not a shared assumption. The
    wind archive is local clock time, established by the seasonal stability of
    its publication hour across the whole archive; the demand archive is
    unresolved and read as UTC, the later and therefore conservative reading.
    """
    out = frame.copy()
    imputed = impute_published_at(out[target_date_column], resource)
 
    if resource.timestamp_column is None or resource.timestamp_column not in out.columns:
        out["published_at"] = imputed
        out["published_at_source"] = "schedule"
        out["published_at_suspect"] = False
        return out
 
    recorded = pd.to_datetime(
        out[resource.timestamp_column], errors="coerce"
    ).dt.tz_localize(resource.timestamp_timezone)
    if resource.timestamp_timezone != "UTC":
        recorded = recorded.dt.tz_convert(UTC)
 
    # Usable means published strictly before the settlement day begins. A row
    # with no timestamp at all fails for the same reason: absence is not proof
    # of timeliness.
    usable = recorded.notna() & (recorded < day_start_utc)
 
    out["published_at"] = recorded.where(usable, imputed)
    out["published_at_source"] = pd.Series("archive", index=out.index).where(
        usable, "schedule"
    )
    out["published_at_suspect"] = ~usable
 
    suspect = int((~usable).sum())
    if suspect:
        LOGGER.warning(
            "%s: %s of %s rows carry a publication timestamp that does not "
            "precede their settlement day; falling back to the schedule and "
            "flagging them",
            resource.name,
            suspect,
            len(out),
        )
    return out
 
 
def normalise_wind(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the wind archive onto the project's canonical columns.
 
    The archive supplies its own UTC instant per row in `Datetime_GMT`, beside
    a settlement date and period. Both are kept: the timestamp becomes the
    index, and the labels remain as columns, which is the same split the price
    pipeline already uses. Carrying both also makes NESO's settlement calendar
    checkable against this project's, rather than assuming they agree.
 
    `Capacity` is installed wind capacity, which roughly doubles across the
    archive. A forecast in MW is therefore not comparable across years, so the
    share of capacity is derived here as the stationary form of the same
    signal.
    """
    required = ["Datetime_GMT", "Date", "Settlement_period", "Incentive_forecast"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(
            f"wind archive is missing {missing}; available columns are "
            f"{sorted(frame.columns)}"
        )
 
    out = pd.DataFrame(index=frame.index)
    out["settlement_date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    out["settlement_period"] = frame["Settlement_period"].astype(int)
    out["wind_forecast_mw"] = pd.to_numeric(frame["Incentive_forecast"], errors="coerce")
 
    if "Capacity" in frame.columns:
        capacity = pd.to_numeric(frame["Capacity"], errors="coerce")
        out["wind_capacity_mw"] = capacity
        # Guard against a zero or missing capacity producing an infinity that
        # would survive into a feature and poison a model silently.
        out["wind_forecast_share"] = (out["wind_forecast_mw"] / capacity).where(
            capacity > 0
        )
 
    utc_index = pd.to_datetime(frame["Datetime_GMT"]).dt.tz_localize(UTC)
    out.index = pd.DatetimeIndex(utc_index, name="timestamp_utc")
 
    if resource_timestamp := "Forecast_Timestamp" in frame.columns:
        out["Forecast_Timestamp"] = frame["Forecast_Timestamp"].to_numpy()
    LOGGER.debug("wind archive carried its own timestamp: %s", bool(resource_timestamp))
 
    return out.sort_index()
 
 
def day_start_utc(settlement_dates: pd.Series) -> pd.Series:
    """The UTC instant at which each settlement day's first period begins.
 
    Settlement day boundaries are local midnight, not 00:00 UTC. In summer that
    is 23:00 UTC on the previous calendar day, which is why the conversion goes
    through London rather than being assumed.
    """
    dates = pd.to_datetime(settlement_dates).dt.normalize()
    return dates.dt.tz_localize(LONDON).dt.tz_convert(UTC)
 
 
# Columns of the demand performance archive that describe the outturn rather
# than the forecast. None of them existed at the decision time, so none may
# enter a feature set. They are retained for forecast-error analysis and are
# named here so that a downstream selection can exclude them by reference
# rather than by remembering.
DEMAND_OUTTURN_COLUMNS: tuple[str, ...] = (
    "demand_outturn_mw",
    "demand_absolute_error_mw",
    "demand_ape_pct",
    "triad_avoidance_estimate_mw",
    "triad_corrected_outturn_mw",
)
 
 
def normalise_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the demand performance archive onto the project's canonical columns.
 
    The archive's `Datetime` column is not comparable with the wind archive's
    `Datetime_GMT`. For settlement period 1 on 2021-04-01, a BST date, wind
    records 23:00 on the previous date and demand records 00:30 on the date
    itself: wind carries the UTC instant the period begins, demand carries the
    local instant it ends. Joining the two frames on their own timestamp
    columns would misalign them by up to ninety minutes and produce a frame
    that looks correct.
 
    Neither column is therefore used as the index. The settlement date and
    period labels are the join key, converted through this project's own
    calendar, which keeps one definition of a settlement period across every
    source.
 
    Outturn columns are carried through but named in DEMAND_OUTTURN_COLUMNS,
    because the file ships the answer beside the forecast and the answer was
    not available when the decision was taken.
    """
    required = ["Date", "Settlement_Period", "Demand_Forecast"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(
            f"demand archive is missing {missing}; available columns are "
            f"{sorted(frame.columns)}"
        )
 
    out = pd.DataFrame(index=frame.index)
    out["settlement_date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    out["settlement_period"] = frame["Settlement_Period"].astype(int)
    out["demand_forecast_mw"] = pd.to_numeric(frame["Demand_Forecast"], errors="coerce")
 
    optional = {
        "Demand_Outturn": "demand_outturn_mw",
        "Absolute_Error": "demand_absolute_error_mw",
        "APE": "demand_ape_pct",
        "TRIAD_Avoidance_Estimate": "triad_avoidance_estimate_mw",
        "TRIAD_Avoidance_Corrected_Demand_Outturn": "triad_corrected_outturn_mw",
    }
    for source, target in optional.items():
        if source in frame.columns:
            out[target] = pd.to_numeric(frame[source], errors="coerce")
 
    if "Publish_Datetime" in frame.columns:
        out["Publish_Datetime"] = frame["Publish_Datetime"].to_numpy()
 
    return out
 