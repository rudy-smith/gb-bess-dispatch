"""Cleaning and data-quality policy for GB half-hourly wholesale price series.
 
The policy is deliberately non-destructive: no row is dropped and no price is
overwritten except by short-gap interpolation, which is recorded in a flag
column. Every judgement the module makes is exposed as a boolean column so that
downstream code chooses its own tolerance rather than inheriting one silently.
"""
 
from __future__ import annotations
 
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
 
import pandas as pd
 
LOGGER = logging.getLogger(__name__)
 
HALF_HOUR = pd.Timedelta(minutes=30)
 
# Plausibility band for GB half-hourly wholesale prices, GBP/MWh. Values outside
# it are treated as suspected data errors rather than market events: the lower
# bound sits below the most negative GB prices observed to date, the upper bound
# above the cash-out price cap. The band exists to catch feed corruption, not to
# tame volatility.
DEFAULT_PRICE_BOUNDS: tuple[float, float] = (-1000.0, 6000.0)
 
# Individual Liquidity Threshold from the Market Index Definition Statement, in
# MWh, applied identically to every Market Index Data Provider. Where the Market
# Index Volume for a settlement period falls below it, the Market Index Data
# Provider defaults BOTH the price and the volume to zero. A defaulted price
# therefore records an absence of qualifying liquidity, not a traded price of
# zero, and must not be consumed as a price.
#
# The threshold is a MIDS parameter subject to annual review by the Imbalance
# Settlement Group, not a constant of nature. Value confirmed against the MIDS
# Review 2025, covering 1 August 2024 to 31 July 2025.
INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH: float = 25.0
 
# Prices arrive as floats, so exact equality against zero is fragile. The
# tolerance is far below the smallest price increment the exchanges quote.
ZERO_PRICE_TOLERANCE: float = 1e-9
 
# Which volume column accompanies which price column. Two of the three follow a
# prefix convention and one does not, so the mapping is written out rather than
# derived: a silent miss here would disable the liquidity test for a column
# while the rest of the pipeline carried on reporting success.
PRICE_TO_VOLUME_COLUMN: dict[str, str] = {
    "price_apx": "volume_apx",
    "price_n2ex": "volume_n2ex",
    "price_mid_gbp_mwh": "volume_mid_mwh",
}
 
 
@dataclass(frozen=True)
class CleaningPolicy:
    """Configuration for the cleaning pass. Frozen so a run cannot mutate it."""
 
    price_columns: Sequence[str] = ("price_apx", "price_n2ex", "price_mid_gbp_mwh")
    primary_price_column: str = "price_apx"
    price_bounds: tuple[float, float] = DEFAULT_PRICE_BOUNDS
    max_interp_periods: int = 2       # gaps of <= 1 hour are filled
    stuck_run_length: int = 6         # >= 3 hours of an identical price is a feed fault
    liquidity_threshold_mwh: float = INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH
    zero_run_min_length: int = 12     # fallback only, used when no volume column exists
    date_column: str = "settlement_date"
    period_column: str = "settlement_period"
 
 
@dataclass
class QualityReport:
    """Everything the cleaning pass observed or changed, in one serialisable object."""
 
    rows_in: int = 0
    rows_out: int = 0
    duplicate_rows_dropped: int = 0
    grid_rows_inserted: int = 0
    index_start: str = ""
    index_end: str = ""
    missing_by_column: dict[str, int] = field(default_factory=dict)
    interpolated_by_column: dict[str, int] = field(default_factory=dict)
    out_of_band_by_column: dict[str, int] = field(default_factory=dict)
    stuck_by_column: dict[str, int] = field(default_factory=dict)
    defaulted_by_column: dict[str, int] = field(default_factory=dict)
    liquidity_anomaly_by_column: dict[str, int] = field(default_factory=dict)
    liquidity_test_applied: dict[str, bool] = field(default_factory=dict)
    days_total: int = 0
    days_incomplete: int = 0
    days_unusable: int = 0
    incomplete_day_detail: dict[str, list[int]] = field(default_factory=dict)
 
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
 
    def summary(self) -> str:
        lines = [
            f"rows in / out            {self.rows_in} / {self.rows_out}",
            f"duplicates dropped       {self.duplicate_rows_dropped}",
            f"grid rows inserted       {self.grid_rows_inserted}",
            f"coverage                 {self.index_start} .. {self.index_end}",
            f"days total / incomplete  {self.days_total} / {self.days_incomplete}",
            f"days unusable            {self.days_unusable}",
        ]
        for col in sorted(self.missing_by_column):
            lines.append(
                f"  {col:<18} missing={self.missing_by_column[col]:<5}"
                f" interp={self.interpolated_by_column.get(col, 0):<5}"
                f" out_of_band={self.out_of_band_by_column.get(col, 0):<5}"
                f" stuck={self.stuck_by_column.get(col, 0):<5}"
                f" defaulted={self.defaulted_by_column.get(col, 0):<5}"
                f" liq_anom={self.liquidity_anomaly_by_column.get(col, 0):<5}"
                f" liq_test={'volume' if self.liquidity_test_applied.get(col) else 'run-length'}"
            )
        return "\n".join(lines)
 
 
# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #
 
def expected_periods(settlement_date: pd.Timestamp | str) -> int:
    """Number of settlement periods in a GB settlement day: 46, 48 or 50.
 
    Derived from the length of the local calendar day rather than hard-coded
    clock-change dates. The date is incremented while naive and localised
    afterwards, because adding a Timedelta to a tz-aware timestamp advances
    absolute time and lands off midnight on a clock-change day.
    """
    naive = pd.Timestamp(settlement_date).normalize()
    start = naive.tz_localize("Europe/London")
    end = (naive + pd.Timedelta(days=1)).tz_localize("Europe/London")
    return int((end - start) / HALF_HOUR)
 
 
# --------------------------------------------------------------------------- #
# Detection primitives — each returns a boolean mask, none mutate the frame
# --------------------------------------------------------------------------- #
 
def out_of_band_mask(series: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    """True where a finite value falls outside the plausibility band."""
    low, high = bounds
    return series.notna() & ((series < low) | (series > high))
 
 
def stuck_mask(series: pd.Series, min_run: int) -> pd.Series:
    """True where a value belongs to a run of >= min_run identical readings.
 
    A frozen feed repeats the last good value; a genuine market event does not
    hold a price to the penny for hours. NaN never compares equal to itself, so
    missing values form singleton runs and are excluded automatically.
    """
    changed = series.ne(series.shift())
    run_id = changed.cumsum()
    run_size = series.groupby(run_id).transform("size")
    return series.notna() & (run_size >= min_run)
 
 
def gap_lengths(series: pd.Series) -> pd.Series:
    """For each row, the length of the contiguous NaN run it belongs to (0 if present)."""
    isna = series.isna()
    block = (~isna).cumsum()
    return isna.groupby(block).transform("sum").where(isna, 0).astype(int)
 
 
def interpolate_short_gaps(
    series: pd.Series, max_gap: int
) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate NaN runs of at most max_gap periods.
 
    Returns the filled series and a mask of the rows that were filled. Runs
    longer than max_gap are left missing on purpose: a long hole cannot be
    invented without manufacturing a price spread that the optimiser will then
    trade against.
    """
    isna = series.isna()
    fillable = isna & (gap_lengths(series) <= max_gap)
    filled_values = series.interpolate(method="time", limit_area="inside")
    # limit_area="inside" refuses to extrapolate, so leading and trailing gaps
    # stay NaN even if short. Exclude them from the mask rather than claiming a
    # fill that did not happen.
    fillable = fillable & filled_values.notna()
    return series.where(~fillable, filled_values), fillable
 
 
# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
 
 
def zeros_to_missing(series: pd.Series, min_run: int = 12) -> pd.Series:
    """Convert zeros to NaN where they form long runs, leaving isolated zeros alone.
 
    A price of exactly zero is legal in a market that prices negatively, so a
    blanket conversion would destroy real observations. A run of many identical
    zeros is instead the signature of a feed encoding absence as a value. The
    run-length test distinguishes the two without assuming which columns are
    affected.
    """
    is_zero = series.eq(0.0)
    run_id = is_zero.ne(is_zero.shift()).cumsum()
    run_size = is_zero.groupby(run_id).transform("size")
    return series.mask(is_zero & (run_size >= min_run))
 
 
def defaulted_price_mask(
    price: pd.Series,
    volume: pd.Series,
    threshold: float = INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH,
    tolerance: float = ZERO_PRICE_TOLERANCE,
) -> pd.Series:
    """True where a zero price is the liquidity default rather than a traded price.
 
    The Market Index Definition Statement defaults both the price and the volume
    to zero when qualifying traded volume falls below the Individual Liquidity
    Threshold. The two cases are therefore separable by volume rather than by
    shape: a genuine traded price of zero carries volume at or above the
    threshold, a defaulted one carries none.
 
    This supersedes the run-length test for any column whose volume is
    available. The run-length test cannot classify an isolated zero at all —
    a single defaulted period surrounded by normal prices is indistinguishable
    from a single genuine one — and an isolated zero is the more dangerous of
    the two, because a perfect-foresight optimiser selects the extremes of each
    day and a spurious zero is an almost irresistible charging price.
    """
    return (price.abs() <= tolerance) & (volume < threshold)
 
 
def liquidity_anomaly_mask(
    price: pd.Series,
    volume: pd.Series,
    threshold: float = INDIVIDUAL_LIQUIDITY_THRESHOLD_MWH,
    tolerance: float = ZERO_PRICE_TOLERANCE,
) -> pd.Series:
    """True where a non-zero price accompanies sub-threshold volume.
 
    Under the default rule this combination cannot occur. Counting it is a test
    of the reading of the rule, not of the data: a non-zero count means either
    the volume field does not mean what it is assumed to mean, or the threshold
    in force over the sample differs from the one configured here. Either way
    the liquidity test above is unsafe and the count must be explained before
    its output is trusted.
    """
    return (price.abs() > tolerance) & volume.notna() & (volume < threshold)
 
 
def clean_prices(
    df: pd.DataFrame, policy: CleaningPolicy | None = None
) -> tuple[pd.DataFrame, QualityReport]:
    """Apply the cleaning policy and return (flagged frame, quality report).
 
    The input is expected to carry a tz-aware UTC DatetimeIndex plus settlement
    date and period columns. Output columns added:
 
      <col>_interpolated   value was filled from neighbours
      <col>_out_of_band    value outside the plausibility band
      <col>_stuck          value inside a frozen-feed run
      period_usable        primary price present, in band and not stuck
      day_complete         settlement day has its full complement of periods
      day_usable           day_complete and every period in the day usable
    """
    policy = policy or CleaningPolicy()
    report = QualityReport(rows_in=len(df))
 
    if df.index.tz is None:
        raise ValueError("index must be tz-aware; UTC is the canonical index")
 
    out = df.sort_index().copy()
 
    # --- 1. duplicates ----------------------------------------------------- #
    # A repeated timestamp is the fingerprint of overlapping fetch windows.
    dupes = out.index.duplicated(keep="first")
    report.duplicate_rows_dropped = int(dupes.sum())
    out = out[~dupes]
 
    # --- 2. reindex onto the complete half-hourly UTC grid ------------------ #
    # Missing periods become explicit NaN rows. In UTC there are no ambiguous or
    # nonexistent timestamps, so a single asfreq exposes every hole.
    before = len(out)
    full_index = pd.date_range(out.index[0], out.index[-1], freq="30min", tz="UTC")
    out = out.reindex(full_index)
    out.index.name = df.index.name or "start_time_utc"
    report.grid_rows_inserted = len(out) - before
    report.index_start = str(out.index[0])
    report.index_end = str(out.index[-1])
 
    # Settlement labels must be rebuilt for inserted rows, which arrived as NaN.
    out = _restore_settlement_labels(out, policy)
 
    # --- 3. per-column detection and short-gap filling ---------------------- #
    if policy.primary_price_column not in out.columns:
        raise KeyError(
            f"primary price column {policy.primary_price_column!r} not in frame. "
            f"available: {sorted(out.columns)}"
        )
    # Scan every configured price column that is actually present, and always
    # include the primary column even if the caller overrode it to something
    # outside the configured list.
    present_cols = [c for c in policy.price_columns if c in out.columns]
    if policy.primary_price_column not in present_cols:
        present_cols.append(policy.primary_price_column)
 
    for col in present_cols:
        raw = out[col].astype(float)
        volume_col = PRICE_TO_VOLUME_COLUMN.get(col)
 
        if volume_col is not None and volume_col in out.columns:
            # Preferred path: classify zeros by the rule that produced them.
            volume = out[volume_col].astype(float)
            defaulted = defaulted_price_mask(raw, volume, policy.liquidity_threshold_mwh)
            anomalies = liquidity_anomaly_mask(raw, volume, policy.liquidity_threshold_mwh)
            series = raw.mask(defaulted)
            report.liquidity_test_applied[col] = True
        else:
            # Fallback: no volume field, so only the shape of the zeros is
            # available. Logged rather than silent, because the two tests do not
            # detect the same set and the difference is not cosmetic.
            LOGGER.warning(
                "no volume column for %r; falling back to the run-length zero test, "
                "which cannot classify isolated defaulted periods",
                col,
            )
            defaulted = raw.eq(0.0) & zeros_to_missing(raw, policy.zero_run_min_length).isna()
            anomalies = pd.Series(False, index=raw.index)
            series = zeros_to_missing(raw, policy.zero_run_min_length)
            report.liquidity_test_applied[col] = False
 
        report.defaulted_by_column[col] = int(defaulted.sum())
        report.liquidity_anomaly_by_column[col] = int(anomalies.sum())
        report.missing_by_column[col] = int(series.isna().sum())
 
        oob = out_of_band_mask(series, policy.price_bounds)
        stuck = stuck_mask(series, policy.stuck_run_length)
        report.out_of_band_by_column[col] = int(oob.sum())
        report.stuck_by_column[col] = int(stuck.sum())
 
        filled, was_filled = interpolate_short_gaps(series, policy.max_interp_periods)
        report.interpolated_by_column[col] = int(was_filled.sum())
 
        out[col] = filled
        out[f"{col}_interpolated"] = was_filled
        out[f"{col}_out_of_band"] = oob
        out[f"{col}_stuck"] = stuck
        out[f"{col}_defaulted"] = defaulted
        out[f"{col}_liquidity_anomaly"] = anomalies
 
    # --- 4. period- and day-level usability -------------------------------- #
    primary = policy.primary_price_column
    out["period_usable"] = (
        out[primary].notna()
        & ~out[f"{primary}_out_of_band"]
        & ~out[f"{primary}_stuck"]
    )
 
    out = _flag_days(out, policy, report)
 
    report.rows_out = len(out)
    return out, report
 
 
def _restore_settlement_labels(
    df: pd.DataFrame, policy: CleaningPolicy
) -> pd.DataFrame:
    """Recompute settlement date and period for rows inserted by the reindex.
 
    Both are derived from local time: the settlement date is the local calendar
    date, and the period is the count of half hours elapsed since local midnight
    plus one. Deriving rather than forward-filling keeps the labels correct
    across a clock change, where the count per day is 46 or 50 rather than 48.
    """
    local = df.index.tz_convert("Europe/London")
    local_date = pd.Series(local.normalize().tz_localize(None), index=df.index)
    midnight = local_date.dt.tz_localize("Europe/London", nonexistent="shift_forward")
    elapsed = (df.index.tz_convert("UTC") - midnight.dt.tz_convert("UTC"))
    df[policy.date_column] = local_date
    df[policy.period_column] = (elapsed // HALF_HOUR).astype(int) + 1
    # Rebuild the local-time column for inserted rows. It is derived from the
    # UTC index, never the reverse: local time is not a valid index because
    # 01:00 on an autumn clock-change day occurs twice.
    if "start_time_local" in df.columns:
        df["start_time_local"] = local
    return df
 
 
def _flag_days(
    df: pd.DataFrame, policy: CleaningPolicy, report: QualityReport
) -> pd.DataFrame:
    """Mark settlement days as complete and usable."""
    dates = pd.to_datetime(df[policy.date_column])
    observed = df.groupby(dates).size()
    expected = pd.Series(
        [expected_periods(d) for d in observed.index], index=observed.index
    )
    complete = observed.eq(expected)
 
    all_usable = df.groupby(dates)["period_usable"].transform("all")
    day_complete = dates.map(complete)
 
    df["day_complete"] = day_complete.to_numpy()
    df["day_usable"] = (day_complete.to_numpy() & all_usable.to_numpy())
 
    report.days_total = len(observed)
    report.days_incomplete = int((~complete).sum())
    report.days_unusable = int((~df.groupby(dates)["day_usable"].first()).sum())
    report.incomplete_day_detail = {
        d.strftime("%Y-%m-%d"): [int(observed[d]), int(expected[d])]
        for d in observed.index[~complete]
    }
    return df
 
 
 
 
