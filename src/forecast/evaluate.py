"""
Forecast scoring: MAE and RMSE against naive baselines, pinball loss, reliability.

MAPE is not used anywhere. GB prices cross zero, and a percentage error on a
price of £0.50 is enormous for a trivial miss, so MAPE would be dominated by
the half hours that matter least.

Every comparison is made on one matched set of rows: those where the target and
every forecast being compared exist. A model scored on more rows than a baseline
is not being compared with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = {
    "baseline_d2": "f_price_d2_same_time",  # "yesterday" as known at 11:00 on D-1
    "baseline_d7": "f_price_d7_same_time",  # same weekday, previous seven days back
}


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name, col in BASELINES.items():
        out[name] = out[col]
    return out


def matched(frame: pd.DataFrame, forecasts: list[str]) -> pd.DataFrame:
    return frame.dropna(subset=["target_price", *forecasts])


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def pinball(y, q_pred, q: float) -> float:
    """Mean pinball (quantile) loss. Minimised in expectation by the true q quantile."""
    diff = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score_table(frame: pd.DataFrame, forecasts: list[str], reference: str) -> pd.DataFrame:
    """MAE, RMSE and MAE skill relative to `reference`, on matched rows."""
    m = matched(frame, forecasts)
    y = m["target_price"]
    ref = mae(y, m[reference])
    rows = []
    for f in forecasts:
        e = mae(y, m[f])
        rows.append({"forecast": f, "mae": e, "rmse": rmse(y, m[f]), "skill_vs_" + reference: 1 - e / ref})
    return pd.DataFrame(rows).set_index("forecast")


def mae_by_hour(frame: pd.DataFrame, forecasts: list[str]) -> pd.DataFrame:
    """MAE by local clock hour, on matched rows."""
    m = matched(frame, forecasts)
    hour = m["f_minute_of_day"] // 60
    return pd.DataFrame(
        {f: (m["target_price"] - m[f]).abs().groupby(hour).mean() for f in forecasts}
    ).rename_axis("local_hour")


def reliability(frame: pd.DataFrame, quantile_cols: dict[float, str]) -> pd.DataFrame:
    """For each nominal quantile, the observed share of outcomes at or below it.

    A calibrated q quantile has the outcome below it a fraction q of the time.
    Pinball loss rewards calibration and sharpness together; this separates out
    the calibration.
    """
    m = matched(frame, list(quantile_cols.values()))
    rows = []
    for q, col in quantile_cols.items():
        rows.append(
            {
                "nominal": q,
                "observed": float((m["target_price"] <= m[col]).mean()),
                "pinball": pinball(m["target_price"], m[col], q),
            }
        )
    return pd.DataFrame(rows).set_index("nominal")


def interval_coverage(frame: pd.DataFrame, lo: str, hi: str) -> float:
    m = matched(frame, [lo, hi])
    return float(((m["target_price"] >= m[lo]) & (m["target_price"] <= m[hi])).mean())


def block_bootstrap_skill(
    frame: pd.DataFrame,
    model: str,
    reference: str,
    n_boot: int = 2000,
    block_days: int = 7,
    seed: int = 0,
) -> tuple[float, float, float]:
    """MAE skill of `model` over `reference` with a CI from a moving-block
    bootstrap over days.

    Days are resampled in runs because forecast errors on neighbouring days are
    correlated through weather and fuel prices; resampling single half hours
    would treat 17,000 strongly dependent errors as independent and give an
    interval far too narrow to mean anything.
    """
    m = matched(frame, [model, reference])
    daily = m.assign(
        e_model=(m["target_price"] - m[model]).abs(),
        e_ref=(m["target_price"] - m[reference]).abs(),
    ).groupby("settlement_date")[["e_model", "e_ref"]].sum()
    a, b = daily["e_model"].to_numpy(), daily["e_ref"].to_numpy()
    n = len(daily)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block_days)) % n).reshape(n_boot, -1)[:, :n]
    skill = 1 - a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.quantile(skill, [0.025, 0.975])
    return float(1 - a.sum() / b.sum()), float(lo), float(hi)
