"""
Forecast-driven dispatch, settled at actual prices, measured against perfect foresight.

The protocol, for each delivery day D in the out-of-sample forecast period:

1. At 11:00 on D-1, the optimiser is given the forecast price for every half hour
   of D and solves for a schedule. This is the same MILP as the benchmark, with
   the same battery and the same day-boundary state of charge; only the prices
   it sees differ.
2. The battery delivers that schedule exactly. It is fully controllable and the
   schedule is feasible by construction, so there is no physical deviation.
3. Each half hour is settled at the actual price. Revenue = sum over periods of
   actual price x (export - import), less degradation on throughput.

Perfect foresight solves the same day on the actual prices and is settled the
same way. The ratio of the two is the share of the benchmark a strategy captures.
Because the battery, the constraints and the settlement price are identical, the
gap is the cost of the information alone.

What this is not:
- Not an intraday rolling re-optimisation. The forecasts are a single day-ahead
  vintage, so there is nothing new to re-optimise on during the day; the
  horizon rolls one delivery day at a time.
- No imbalance exposure. The schedule is traded at the index price of each half
  hour and delivered exactly. Imbalance arises when a position fixed at one
  price is delivered differently; with one price series and a fully
  controllable asset, that does not occur here. It would with a separate
  day-ahead auction price, and is recorded as a limitation, not modelled as
  zero by accident.

The invariant that guards against the most dangerous bug: on every day, a
schedule settled at actual prices cannot beat the perfect-foresight schedule,
because perfect foresight is the optimum over all feasible schedules for those
prices. A day where it does means revenue is being settled at the wrong price,
typically the forecast. run_day raises if it happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.battery import BatterySpec, check_feasibility, simulate_schedule
from src.model.milp import solve_dispatch

INVARIANT_TOL = 1e-6


def settle(
    spec: BatterySpec,
    charge_mw: np.ndarray,
    discharge_mw: np.ndarray,
    actual_prices: np.ndarray,
    degradation_cost: float,
) -> dict:
    """Settle a fixed schedule at actual prices with the independent replay."""
    result = simulate_schedule(
        spec, charge_mw, discharge_mw, actual_prices,
        degradation_cost_gbp_per_mwh=degradation_cost,
    )
    violations = check_feasibility(spec, result)
    if violations:
        raise RuntimeError(f"schedule infeasible when replayed: {violations}")
    return result


def run_day(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    spec: BatterySpec,
    degradation_cost: float = 0.0,
    solver=None,
) -> dict[str, float]:
    """Perfect foresight and every forecast strategy for one delivery day.

    Returns net revenue and throughput per strategy, keyed "<name>_net" and
    "<name>_mwh", with perfect foresight under the name "perfect".
    """
    actual = np.asarray(actual, dtype=float)
    pf = solve_dispatch(actual, spec, degradation_cost_gbp_per_mwh=degradation_cost, solver=solver)
    if not pf.solved or pf.violations:
        raise RuntimeError(f"perfect-foresight solve failed: {pf.status} {pf.violations}")
    out = {"perfect_net": pf.net_revenue_gbp, "perfect_mwh": pf.throughput_mwh}

    for name, forecast in forecasts.items():
        plan = solve_dispatch(
            np.asarray(forecast, dtype=float), spec,
            degradation_cost_gbp_per_mwh=degradation_cost, solver=solver,
        )
        if not plan.solved or plan.violations:
            raise RuntimeError(f"{name}: solve failed: {plan.status} {plan.violations}")
        settled = settle(
            spec,
            plan.trace["charge_mw"].to_numpy(),
            plan.trace["discharge_mw"].to_numpy(),
            actual,
            degradation_cost,
        )
        net = float(settled["net_revenue_gbp"])
        if net > pf.net_revenue_gbp + INVARIANT_TOL:
            raise RuntimeError(
                f"{name} earned {net:.6f} at actual prices, more than perfect foresight "
                f"{pf.net_revenue_gbp:.6f}: revenue is being settled at the wrong price"
            )
        out[f"{name}_net"] = net
        out[f"{name}_mwh"] = float(settled["throughput_mwh"])
    return out


def matched_days(frame: pd.DataFrame, strategies: list[str]) -> tuple[list, list]:
    """Days usable for every strategy at once: (kept, dropped).

    A day is kept only if it is usable for the benchmark and every half hour has
    an actual price and a value for every strategy. Scoring strategies on
    different day sets would compare different things.
    """
    kept, dropped = [], []
    for day, g in frame.groupby("settlement_date", sort=True):
        ok = bool(g["day_usable"].fillna(False).all()) and g[["target_price", *strategies]].notna().all().all()
        (kept if ok else dropped).append(day)
    return kept, dropped


def run_backtest(
    frame: pd.DataFrame,
    strategies: dict[str, str],
    spec: BatterySpec,
    degradation_cost: float = 0.0,
    solver=None,
) -> tuple[pd.DataFrame, list]:
    """Run every matched day. `strategies` maps a name to a forecast column.

    Returns (one row per day, dropped days).
    """
    cols = list(strategies.values())
    kept, dropped = matched_days(frame, cols)
    rows = []
    for day in kept:
        g = frame[frame["settlement_date"] == day].sort_values("target_time")
        forecasts = {name: g[col].to_numpy() for name, col in strategies.items()}
        r = run_day(g["target_price"].to_numpy(), forecasts, spec, degradation_cost, solver)
        r["settlement_date"] = day
        r["wind_suspect"] = bool(g["wind_suspect"].fillna(False).astype(bool).any())
        rows.append(r)
    return pd.DataFrame(rows).set_index("settlement_date"), dropped


def capture(daily: pd.DataFrame, name: str) -> float:
    """Share of perfect-foresight revenue captured, as a ratio of sums."""
    return float(daily[f"{name}_net"].sum() / daily["perfect_net"].sum())


def bootstrap_capture(
    daily: pd.DataFrame, name: str, n_boot: int = 2000, block_days: int = 7, seed: int = 0
) -> tuple[float, float, float]:
    """Capture with a 95% CI from a moving-block bootstrap over days.

    A ratio of sums, resampled in runs of consecutive days because volatile
    spells, which dominate both revenue and forecast error, last several days.
    """
    a = daily[f"{name}_net"].to_numpy()
    b = daily["perfect_net"].to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block_days)) % n).reshape(n_boot, -1)[:, :n]
    ratios = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return float(a.sum() / b.sum()), float(lo), float(hi)


def bootstrap_capture_difference(
    daily: pd.DataFrame, a: str, b: str, n_boot: int = 2000, block_days: int = 7, seed: int = 0
) -> tuple[float, float, float]:
    """Capture of strategy a minus capture of b, with a paired 95% CI.

    Both strategies are resampled on the same days in each replicate. Comparing
    two separate intervals ignores that they share days, and a volatile spell
    that helps one helps the other, so overlapping intervals do not mean the
    difference is zero.
    """
    x = daily[f"{a}_net"].to_numpy()
    y = daily[f"{b}_net"].to_numpy()
    p = daily["perfect_net"].to_numpy()
    n = len(p)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block_days)) % n).reshape(n_boot, -1)[:, :n]
    diff = (x[idx].sum(axis=1) - y[idx].sum(axis=1)) / p[idx].sum(axis=1)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return float((x.sum() - y.sum()) / p.sum()), float(lo), float(hi)
