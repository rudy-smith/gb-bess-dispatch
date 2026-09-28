"""
Forecast-skill measurement, and the test for whether a block of archived forecasts
was regenerated after the fact.

The problem this answers
------------------------
The NESO day-ahead wind archive has a contiguous block of 2023 rows stamped with
a publication instant after the day they forecast. A late stamp alone would be a
metadata fault, repaired by imputing the publication schedule. The dangerous
reading is that the block was *regenerated* when it was rewritten, using weather
or outturn information that did not exist at the original decision time. Those
values would look like forecasts and behave like the outturn, and every model fed
with them would look better than it could have been, with no visible symptom.

The test
--------
A regenerated forecast has a tell: it is too accurate. So the block's error is
compared with the same calendar window in years whose archive is untouched. The
seasonal control matters because wind forecast error is strongly seasonal (lower
absolute error in the summer lull), so comparing a summer block with the rest of
2023 would find it "too accurate" for reasons that have nothing to do with
contamination.

Error is normalised by installed capacity, because the fleet grows across the
comparison years and MW error scales with the fleet.

The statistic is the ratio of mean absolute error, block over controls, with a
confidence interval from a moving-block bootstrap over days. Days are resampled
in runs of consecutive days rather than individually, because forecast error is
driven by weather systems that persist for several days; resampling days one at
a time treats correlated days as independent and gives an interval that is too
narrow, which here would make an inconclusive result look decisive.

A second, corroborating signal: a genuine day-ahead forecast gets worse with
lead time, so error late in the settlement day (further from publication)
exceeds error early in it. A forecast rebuilt from later information has no
reason to show that growth.

Decision thresholds are fixed here, before the data is examined, so that the
verdict cannot be tuned to the result. They are judgement, not theory; the
control-versus-control ratio printed beside the verdict shows how much genuine
year-to-year variation there is, which is the check on whether the thresholds
were sensible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["settlement_date", "settlement_period"]

# Lower bound of the ratio CI at or above which the block is consistent with a
# genuine forecast: it is no more than 20% more accurate than its seasonal peers.
GENUINE_FLOOR = 0.80
# Upper bound of the ratio CI below which the block is too accurate to be a
# genuine day-ahead forecast: at least 30% better than its peers, with confidence.
REGENERATED_CEILING = 0.70

# Consecutive days per bootstrap block. About the lifetime of a synoptic weather
# pattern over the UK.
BOOTSTRAP_BLOCK_DAYS = 7


def join_forecast_outturn(forecast: pd.DataFrame, outturn: pd.DataFrame) -> pd.DataFrame:
    """Inner-join forecast and outturn on the settlement labels, strictly one to one.

    `validate="one_to_one"` makes pandas raise if either side has a repeated key.
    Without it, a duplicate forecast row would silently double-count that period
    in every error statistic.

    forecast needs: settlement_date, settlement_period, wind_forecast_mw,
    wind_capacity_mw. outturn needs: settlement_date, settlement_period,
    wind_outturn_mw.
    """
    f = forecast.copy()
    o = outturn.copy()
    for frame in (f, o):
        frame["settlement_date"] = pd.to_datetime(frame["settlement_date"]).dt.normalize()
        frame["settlement_period"] = frame["settlement_period"].astype(int)
    return f.merge(o, on=KEYS, how="inner", validate="one_to_one")


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Signed and absolute error as a share of installed capacity.

    Sign convention: forecast minus outturn, so a positive error is an
    over-forecast. Over-forecasting wind leaves the system short, which is the
    direction that raises imbalance prices.
    """
    out = frame.copy()
    cap = out["wind_capacity_mw"].where(out["wind_capacity_mw"] > 0)
    out["error_mw"] = out["wind_forecast_mw"] - out["wind_outturn_mw"]
    out["error_share"] = out["error_mw"] / cap
    out["abs_error_share"] = out["error_share"].abs()
    return out


def contiguous_runs(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group a set of dates into runs of consecutive days, as (first, last) pairs."""
    days = pd.Series(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    if days.empty:
        return []
    breaks = days.diff() != pd.Timedelta(days=1)
    run_id = breaks.cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in days.groupby(run_id)]


def in_calendar_window(dates: pd.Series, first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """True where the date's month-day falls within [first, last], in any year.

    Used to take the same calendar window from each control year. Assumes the
    window does not wrap past 31 December, which holds for a June to October block.
    """
    d = pd.to_datetime(dates)
    md = d.dt.month * 100 + d.dt.day
    return (md >= first.month * 100 + first.day) & (md <= last.month * 100 + last.day)


def daily_mae(frame: pd.DataFrame) -> pd.Series:
    """Mean absolute capacity-normalised error per settlement day, in date order."""
    return frame.groupby("settlement_date")["abs_error_share"].mean().sort_index()


def summarise(frame: pd.DataFrame) -> dict:
    """Headline skill numbers for one window."""
    sp = frame["settlement_period"]
    early = frame.loc[sp <= 16, "abs_error_share"].mean()
    late = frame.loc[sp >= 33, "abs_error_share"].mean()
    return {
        "days": frame["settlement_date"].nunique(),
        "periods": len(frame),
        "mae_pct_cap": 100 * frame["abs_error_share"].mean(),
        "rmse_pct_cap": 100 * np.sqrt((frame["error_share"] ** 2).mean()),
        "bias_pct_cap": 100 * frame["error_share"].mean(),
        "corr": frame["wind_forecast_mw"].corr(frame["wind_outturn_mw"]),
        "mae_sp1_16": 100 * early,
        "mae_sp33_48": 100 * late,
        "growth_late_over_early": late / early if early else np.nan,
    }


def _block_bootstrap_means(
    values: np.ndarray, n_boot: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular moving-block bootstrap of the mean of a daily series.

    Each replicate stitches together runs of `block` consecutive days, starting at
    random positions and wrapping round the end, until it has as many days as the
    original. Circular wrapping gives every day the same chance of selection,
    which a non-wrapping scheme does not for days near the ends.
    """
    n = len(values)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    return values[idx].mean(axis=1)


def bootstrap_ratio(
    numerator_daily: pd.Series,
    denominator_daily: pd.Series,
    n_boot: int = 10_000,
    block: int = BOOTSTRAP_BLOCK_DAYS,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI for mean(numerator) / mean(denominator).

    The two samples are resampled independently, since they are different years.
    A fixed seed makes the interval reproducible between runs.
    """
    rng = np.random.default_rng(seed)
    num = numerator_daily.to_numpy(dtype=float)
    den = denominator_daily.to_numpy(dtype=float)
    ratios = _block_bootstrap_means(num, n_boot, block, rng) / _block_bootstrap_means(
        den, n_boot, block, rng
    )
    alpha = (1 - level) / 2
    lo, hi = np.quantile(ratios, [alpha, 1 - alpha])
    return float(num.mean() / den.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str


def verdict(ratio_lo: float, ratio_hi: float) -> Verdict:
    """Apply the pre-registered thresholds to the ratio's confidence interval."""
    if ratio_lo >= GENUINE_FLOOR:
        return Verdict(
            "GENUINE",
            f"block error is not materially below its seasonal peers "
            f"(CI lower bound {ratio_lo:.2f} >= {GENUINE_FLOOR})",
        )
    if ratio_hi < REGENERATED_CEILING:
        return Verdict(
            "REGENERATED",
            f"block error is well below its seasonal peers "
            f"(CI upper bound {ratio_hi:.2f} < {REGENERATED_CEILING})",
        )
    return Verdict(
        "INCONCLUSIVE",
        f"CI [{ratio_lo:.2f}, {ratio_hi:.2f}] straddles the thresholds "
        f"({REGENERATED_CEILING}, {GENUINE_FLOOR})",
    )
