"""
Look-ahead audit: recompute every feature independently and time its sources.

Two questions, for every registered feature:

1. Does the SQL compute what it claims? Each price and wind-error feature is
   rebuilt here in plain pandas, sharing no code with sql/features.sql: local
   clock keys come from pandas' own timezone conversion rather than the
   calendar table, the latest published price comes from pandas.merge_asof
   rather than DuckDB's ASOF JOIN, and windows are filtered day by day rather
   than joined. The two results must agree. This is the same principle as the
   dispatch replay, which checks the optimiser with code that shares nothing
   with it: a mistake common to both cannot cancel out and pass.

2. When did each source become available? For every feature value the audit
   finds the latest instant any of its inputs became public and reports the
   smallest margin before the decision. Every margin must be positive.

Calendar features are fixed by the clock and have no source instant. The NESO
forecast features are timed from the publication instant carried beside them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prep.features import FEATURES

LONDON = "Europe/London"
CLOCK_FEATURES = {"f_settlement_period", "f_minute_of_day", "f_day_of_week", "f_is_weekend",
                  "f_month"}


def _num(s: pd.Series) -> pd.Series:
    """Any numeric-looking column as plain float64 with NaN for missing.

    Parquet files written by different tools come back with different dtypes:
    numpy float64 with NaN, pandas' nullable Float64 or Int64 with pd.NA, or
    object arrays holding pd.NA. The audit compares values across all of them,
    so everything is normalised here once, on the way in.
    """
    out = pd.to_numeric(pd.Series(s), errors="coerce").astype("Float64")
    return pd.Series(out.to_numpy(dtype="float64", na_value=np.nan), index=out.index)


def _ts(s: pd.Series) -> pd.Series:
    """Any timestamp column as tz-aware UTC datetime64 with NaT for missing."""
    return pd.to_datetime(pd.Series(s), utc=True, errors="coerce")


def _normalise(features, prices, point_in_time, generation):
    features = features.copy()
    for c in ("target_time", "decision_time"):
        features[c] = _ts(features[c])
    prices = prices.copy()
    prices["price_apx"] = _num(prices["price_apx"])
    prices.index = pd.DatetimeIndex(_ts(pd.Series(prices.index)))
    pit = point_in_time.copy()
    for c in ("target_time", "decision_time", "wind_published_at", "demand_published_at"):
        pit[c] = _ts(pit[c])
    for c in ("wind_forecast_mw", "wind_capacity_mw", "wind_forecast_share",
              "demand_forecast_mw"):
        pit[c] = _num(pit[c])
    gen = generation[["gen_wind_mw"]].copy()
    gen["gen_wind_mw"] = _num(gen["gen_wind_mw"])
    gen.index = pd.DatetimeIndex(_ts(pd.Series(generation.index)))
    return features, prices, pit, gen


def _local_keys(utc: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    local = pd.DatetimeIndex(utc).tz_convert(LONDON)
    date = pd.Series(local.tz_localize(None).normalize())
    minute = pd.Series(local.hour * 60 + local.minute)
    return date, minute


def price_sources(prices: pd.DataFrame, lag_minutes: int) -> pd.DataFrame:
    """Observed, non-interpolated prices with local keys and publication instant.

    `first` is computed over every half hour before any filtering, so a missing
    first pass through the repeated autumn hour does not promote the second.
    """
    idx = pd.DatetimeIndex(prices.index)
    date, minute = _local_keys(idx)
    s = pd.DataFrame(
        {
            "source_time": idx,
            "price": prices["price_apx"].to_numpy(),
            "local_date": date.to_numpy(),
            "minute_of_day": minute.to_numpy(),
        }
    )
    s["first"] = ~s.duplicated(["local_date", "minute_of_day"])
    filled = prices.get("price_apx_interpolated", pd.Series(False, index=prices.index))
    keep = s["price"].notna().to_numpy() & ~filled.fillna(False).astype(bool).to_numpy()
    s = s.loc[keep].reset_index(drop=True)
    s["available_at"] = s["source_time"] + pd.Timedelta(minutes=30 + lag_minutes)
    return s


def _same_time_lag(f: pd.DataFrame, s: pd.DataFrame, days: int) -> tuple[pd.Series, pd.Series]:
    src = s[s["first"]].assign(local_date=lambda x: x["local_date"] + pd.Timedelta(days=days))
    m = f[["local_date", "minute_of_day", "decision_time"]].merge(
        src[["local_date", "minute_of_day", "price", "available_at"]],
        on=["local_date", "minute_of_day"], how="left",
    )
    ok = m["available_at"] < m["decision_time"]
    return m["price"].where(ok).set_axis(f.index), m["available_at"].where(ok).set_axis(f.index)


def recompute(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    point_in_time: pd.DataFrame,
    generation: pd.DataFrame,
    lag_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (recomputed feature values, latest source availability), both
    aligned to `features`' rows and keyed by feature name."""
    f = features[["target_time", "decision_time"]].copy()
    date, minute = _local_keys(f["target_time"])
    f["local_date"], f["minute_of_day"] = date.to_numpy(), minute.to_numpy()
    s = price_sources(prices, lag_minutes)

    val: dict[str, pd.Series] = {}
    avail: dict[str, pd.Series] = {}

    for name, days in (("f_price_d2_same_time", 2), ("f_price_d7_same_time", 7)):
        val[name], avail[name] = _same_time_lag(f, s, days)

    daily = s.groupby("local_date").agg(
        mean=("price", "mean"),
        hi=("price", "max"),
        lo=("price", "min"),
        available_at=("available_at", "max"),
    )
    d2 = daily.reindex(f["local_date"] - pd.Timedelta(days=2)).set_axis(f.index)
    ok = d2["available_at"] < f["decision_time"]
    val["f_price_d2_mean"] = d2["mean"].where(ok)
    val["f_price_d2_spread"] = (d2["hi"] - d2["lo"]).where(ok)
    avail["f_price_d2_mean"] = avail["f_price_d2_spread"] = d2["available_at"].where(ok)

    # Seven-day windows, one delivery day at a time.
    per_day = []
    for day, decision in f.groupby("local_date")["decision_time"].first().items():
        w = s[
            (s["local_date"] >= day - pd.Timedelta(days=8))
            & (s["local_date"] <= day - pd.Timedelta(days=2))
            & (s["available_at"] < decision)
        ]
        same = w[w["first"]].groupby("minute_of_day")["price"].mean()
        per_day.append(
            pd.DataFrame(
                {
                    "local_date": day,
                    "minute_of_day": same.index,
                    "same_time_mean": same.to_numpy(),
                    "mean": w["price"].mean(),
                    "std": w["price"].std(ddof=1),
                    "available_at": w["available_at"].max(),
                }
            )
        )
    win = pd.concat(per_day, ignore_index=True)
    day_level = win.drop_duplicates("local_date").set_index("local_date")
    wd = day_level.reindex(f["local_date"]).set_axis(f.index)
    val["f_price_7d_mean"], val["f_price_7d_std"] = wd["mean"], wd["std"]
    avail["f_price_7d_mean"] = avail["f_price_7d_std"] = wd["available_at"]
    st = f[["local_date", "minute_of_day"]].merge(
        win[["local_date", "minute_of_day", "same_time_mean", "available_at"]],
        on=["local_date", "minute_of_day"], how="left",
    ).set_axis(f.index)
    val["f_price_7d_same_time_mean"] = st["same_time_mean"]
    avail["f_price_7d_same_time_mean"] = st["available_at"]

    # Latest published price: merge_asof, strictly before the decision.
    order = f.sort_values("decision_time")
    last = pd.merge_asof(
        order[["decision_time"]].reset_index(),
        s.sort_values("available_at")[["available_at", "price"]],
        left_on="decision_time", right_on="available_at",
        direction="backward", allow_exact_matches=False,
    ).set_index("index").reindex(f.index)
    val["f_last_known_price"], avail["f_last_known_price"] = last["price"], last["available_at"]

    # Wind forecast error, from the forecast attached at each day's own decision.
    pit = point_in_time.set_index("target_time")
    gen = generation["gen_wind_mw"]
    both = pd.DataFrame({"fc": pit["wind_forecast_mw"], "cap": pit["wind_capacity_mw"]}).join(
        gen.rename("out"), how="inner"
    ).dropna(subset=["fc", "out"])
    edate, _ = _local_keys(both.index)
    both["local_date"] = edate.to_numpy()
    both["share"] = (both["fc"] - both["out"]) / both["cap"].where(both["cap"] > 0)
    both["available_at"] = both.index + pd.Timedelta(minutes=30 + lag_minutes)
    err = both.groupby("local_date").agg(
        err=("share", "mean"), available_at=("available_at", "max")
    )
    e2 = err.reindex(f["local_date"] - pd.Timedelta(days=2)).set_axis(f.index)
    ok = e2["available_at"] < f["decision_time"]
    val["f_wind_err_d2"], avail["f_wind_err_d2"] = e2["err"].where(ok), e2["available_at"].where(ok)

    bias_rows = {}
    for day, decision in f.groupby("local_date")["decision_time"].first().items():
        w = err[
            (err.index >= day - pd.Timedelta(days=29))
            & (err.index <= day - pd.Timedelta(days=2))
            & (err["available_at"] < decision)
        ]
        bias_rows[day] = (w["err"].mean(), w["available_at"].max())
    bias = pd.DataFrame.from_dict(bias_rows, orient="index", columns=["bias", "available_at"])
    b = bias.reindex(f["local_date"]).set_axis(f.index)
    val["f_wind_bias_28d"], avail["f_wind_bias_28d"] = b["bias"], b["available_at"]
    val["f_wind_err_d2_demeaned"] = val["f_wind_err_d2"] - val["f_wind_bias_28d"]
    avail["f_wind_err_d2_demeaned"] = pd.concat(
        [avail["f_wind_err_d2"], avail["f_wind_bias_28d"]], axis=1
    ).max(axis=1)

    # NESO forecasts: values straight from the point-in-time frame, timed by
    # their own publication instants.
    p = pit.reindex(f["target_time"]).set_axis(f.index)
    for name, col, pub in (
        ("f_wind_forecast_mw", "wind_forecast_mw", "wind_published_at"),
        ("f_wind_forecast_share", "wind_forecast_share", "wind_published_at"),
        ("f_demand_forecast_mw", "demand_forecast_mw", "demand_published_at"),
    ):
        val[name], avail[name] = p[col], p[pub].where(p[col].notna())
    val["f_residual_demand_mw"] = p["demand_forecast_mw"] - p["wind_forecast_mw"]
    avail["f_residual_demand_mw"] = pd.concat(
        [avail["f_wind_forecast_mw"], avail["f_demand_forecast_mw"]], axis=1
    ).max(axis=1).where(val["f_residual_demand_mw"].notna())

    return pd.DataFrame(val), pd.DataFrame(avail)


def audit(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    point_in_time: pd.DataFrame,
    generation: pd.DataFrame,
    lag_minutes: int,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """One row per registered feature: rows with a value, disagreements with the
    independent recomputation, and the smallest margin between source
    availability and decision, in hours. PASS needs zero disagreements and a
    positive margin."""
    features, prices, point_in_time, generation = _normalise(
        features, prices, point_in_time, generation
    )
    values, availability = recompute(features, prices, point_in_time, generation, lag_minutes)
    rows = []
    for name in FEATURES:
        sql = _num(features[name])
        if name in CLOCK_FEATURES:
            rows.append({"feature": name, "source": "clock", "rows": int(sql.notna().sum()),
                         "mismatches": 0, "min_margin_h": np.nan, "verdict": "PASS"})
            continue
        ind = _num(values[name])
        both_nan = sql.isna() & ind.isna()
        close = np.isclose(sql, ind, rtol=tol, atol=tol)
        mismatches = int((~(both_nan | close)).sum())
        available = pd.to_datetime(availability[name], utc=True)
        margin = (features["decision_time"] - available).dt.total_seconds() / 3600
        min_margin = float(margin.min()) if margin.notna().any() else np.nan
        ok = mismatches == 0 and (np.isnan(min_margin) or min_margin > 0)
        rows.append({"feature": name, "source": FEATURES[name][0], "rows": int(sql.notna().sum()),
                     "mismatches": mismatches, "min_margin_h": min_margin,
                     "verdict": "PASS" if ok else "FAIL"})
    return pd.DataFrame(rows).set_index("feature")
