"""Non-optimising dispatch rules used as comparators for the MILP.

A benchmark is only meaningful against something. The threshold rule below is
the simplest strategy a human would write down, and the optimiser must beat it
by a margin large enough to justify the machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.battery import BatterySpec, simulate_schedule


def trailing_thresholds(
    prices: pd.Series,
    dates: pd.Series,
    window_days: int = 7,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> pd.DataFrame:
    """Percentile thresholds for each day, computed from the preceding days only.

    The current day is excluded from its own thresholds. Including it would let
    the rule see prices it could not have known when the first decision of the
    day was taken, which is the same look-ahead the perfect-foresight benchmark
    admits openly and a baseline must not.
    """
    daily = pd.DataFrame({"price": prices.to_numpy(), "date": pd.to_datetime(dates).to_numpy()})
    by_day = daily.groupby("date")["price"]
    lows = by_day.quantile(low_q)
    highs = by_day.quantile(high_q)

    # shift(1) drops the current day out of its own window; min_periods lets the
    # first few days run on a partial window rather than being discarded.
    out = pd.DataFrame(
        {
            "low": lows.shift(1).rolling(window_days, min_periods=1).mean(),
            "high": highs.shift(1).rolling(window_days, min_periods=1).mean(),
        }
    )
    return out


def threshold_dispatch(
    prices: np.ndarray,
    spec: BatterySpec,
    low: float,
    high: float,
    dt_hours: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge below the low threshold, discharge above the high one, else idle.

    Greedy and myopic: each period is decided on its own price with no regard
    for what follows. Power is capped both by the rating and by the remaining
    headroom, so the schedule is feasible by construction rather than by repair.
    """
    n = len(prices)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    eta_c, eta_d = spec.charge_efficiency, spec.discharge_efficiency
    soc = spec.initial_soc_mwh

    for t in range(n):
        if prices[t] <= low:
            headroom = spec.soc_max_mwh - soc
            power = min(spec.power_mw, headroom / (eta_c * dt_hours)) if headroom > 0 else 0.0
            charge[t] = max(power, 0.0)
            soc += eta_c * charge[t] * dt_hours
        elif prices[t] >= high:
            available = soc - spec.soc_min_mwh
            power = min(spec.power_mw, available * eta_d / dt_hours) if available > 0 else 0.0
            discharge[t] = max(power, 0.0)
            soc -= discharge[t] * dt_hours / eta_d

    return charge, discharge


def run_threshold_baseline(
    df: pd.DataFrame,
    spec: BatterySpec,
    price_col: str,
    window_days: int = 7,
    dt_hours: float = 0.5,
) -> pd.DataFrame:
    """Apply the threshold rule to every usable day and return daily results.

    The rule does not return to its opening state of charge, so any residual
    inventory is marked to market at the day's mean price. Valuing it at the
    closing price instead would let the rule bank an unrealised gain it never
    traded; the mean is the neutral choice and is stated as an assumption.
    """
    thresholds = trailing_thresholds(df[price_col], df["settlement_date"], window_days)
    rows: list[dict[str, object]] = []

    for date, day in df.groupby(pd.to_datetime(df["settlement_date"]).dt.date, sort=True):
        if "day_usable" in day.columns and not bool(day["day_usable"].iloc[0]):
            continue
        if day[price_col].isna().any():
            continue

        key = pd.Timestamp(date)
        if key not in thresholds.index:
            continue
        low, high = thresholds.loc[key, "low"], thresholds.loc[key, "high"]
        if not np.isfinite(low) or not np.isfinite(high):
            continue

        prices = day.sort_index()[price_col].to_numpy(dtype=float)
        charge, discharge = threshold_dispatch(prices, spec, low, high, dt_hours)
        replay = simulate_schedule(spec, charge, discharge, prices, dt_hours)

        residual = float(replay["final_soc_mwh"]) - spec.initial_soc_mwh
        mark_to_market = residual * float(prices.mean())

        rows.append(
            {
                "settlement_date": key,
                "low_threshold": float(low),
                "high_threshold": float(high),
                "net_revenue_gbp": float(replay["gross_revenue_gbp"]) + mark_to_market,
                "traded_revenue_gbp": float(replay["gross_revenue_gbp"]),
                "residual_mwh": residual,
                "throughput_mwh": float(replay["throughput_mwh"]),
                "equivalent_full_cycles": float(replay["equivalent_full_cycles"]),
            }
        )

    return pd.DataFrame(rows).set_index("settlement_date")
