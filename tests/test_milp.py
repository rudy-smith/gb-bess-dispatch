"""Constraint and degenerate-case tests for the battery model and MILP."""

import numpy as np
import pandas as pd
import pytest

from src.model.battery import BatterySpec, simulate_schedule
from src.model.milp import solve_dispatch

SPEC = BatterySpec(power_mw=1.0, duration_h=2.0, round_trip_efficiency=0.90)


def _flat(value: float, n: int = 48) -> np.ndarray:
    return np.full(n, value, dtype=float)


# --- specification ---------------------------------------------------------- #

def test_energy_capacity_is_power_times_duration():
    assert SPEC.energy_mwh == pytest.approx(2.0)


def test_one_way_efficiencies_multiply_to_the_round_trip():
    assert SPEC.charge_efficiency * SPEC.discharge_efficiency == pytest.approx(0.90)


def test_invalid_specifications_are_rejected():
    with pytest.raises(ValueError):
        BatterySpec(round_trip_efficiency=1.4)
    with pytest.raises(ValueError):
        BatterySpec(power_mw=-1)
    with pytest.raises(ValueError):
        BatterySpec(soc_min_frac=0.8, soc_max_frac=0.2)


# --- degenerate cases, checkable by hand ------------------------------------ #

def test_flat_price_yields_zero_revenue():
    """With no spread there is nothing to arbitrage and losses make trading strictly bad."""
    result = solve_dispatch(_flat(60.0), SPEC)
    assert result.solved
    assert result.objective_gbp == pytest.approx(0.0, abs=1e-6)
    assert result.throughput_mwh == pytest.approx(0.0, abs=1e-6)


def test_lossless_flat_price_still_yields_zero_revenue():
    spec = BatterySpec(round_trip_efficiency=1.0)
    result = solve_dispatch(_flat(60.0), spec)
    assert result.objective_gbp == pytest.approx(0.0, abs=1e-6)


def test_single_spike_is_sold_into_and_cheapest_periods_are_bought():
    """One high period: the optimiser must buy in the cheapest available periods."""
    prices = _flat(50.0, 48)
    prices[:4] = [10.0, 12.0, 40.0, 45.0]   # four cheap periods, two very cheap
    prices[30] = 500.0                       # one spike
    spec = BatterySpec(terminal_soc_frac=0.0, initial_soc_frac=0.0)
    result = solve_dispatch(prices, spec)
    trace = result.trace
    assert trace["discharge_mw"].iloc[30] == pytest.approx(spec.power_mw)
    cheapest_two = trace["charge_mw"].iloc[:2].sum()
    assert cheapest_two > trace["charge_mw"].iloc[2:4].sum()


def test_arbitrage_below_the_efficiency_threshold_is_declined():
    """A spread narrower than the round-trip loss must not be traded."""
    prices = np.concatenate([_flat(100.0, 24), _flat(104.0, 24)])
    result = solve_dispatch(prices, SPEC)
    assert result.throughput_mwh == pytest.approx(0.0, abs=1e-6)


# --- constraints ------------------------------------------------------------ #

def test_power_and_soc_bounds_are_respected_on_volatile_prices():
    rng = np.random.default_rng(7)
    prices = rng.normal(80, 60, 48)
    result = solve_dispatch(prices, SPEC)
    trace = result.trace
    assert trace["charge_mw"].max() <= SPEC.power_mw + 1e-9
    assert trace["discharge_mw"].max() <= SPEC.power_mw + 1e-9
    assert trace["soc_mwh"].min() >= SPEC.soc_min_mwh - 1e-9
    assert trace["soc_mwh"].max() <= SPEC.soc_max_mwh + 1e-9
    assert result.violations == []


def test_charge_and_discharge_are_never_simultaneous():
    rng = np.random.default_rng(11)
    prices = rng.normal(60, 90, 48)   # wide spread, many negative periods
    result = solve_dispatch(prices, SPEC)
    trace = result.trace
    assert not ((trace["charge_mw"] > 1e-6) & (trace["discharge_mw"] > 1e-6)).any()


def test_terminal_state_of_charge_is_met():
    rng = np.random.default_rng(3)
    result = solve_dispatch(rng.normal(80, 50, 48), SPEC)
    assert result.trace["soc_mwh"].iloc[-1] == pytest.approx(SPEC.terminal_soc_mwh)


def test_energy_in_times_efficiency_equals_energy_out_over_a_closed_cycle():
    """With equal start and end SoC, exported energy is imported energy times the round trip."""
    rng = np.random.default_rng(5)
    result = solve_dispatch(rng.normal(80, 70, 48), SPEC)
    trace = result.trace
    imported = trace["import_mwh"].sum()
    exported = trace["export_mwh"].sum()
    assert exported == pytest.approx(imported * SPEC.round_trip_efficiency, rel=1e-6, abs=1e-9)


# --- solver validation ------------------------------------------------------ #

def test_objective_matches_independently_recomputed_revenue():
    rng = np.random.default_rng(13)
    prices = rng.normal(80, 60, 48)
    result = solve_dispatch(prices, SPEC)
    replay = simulate_schedule(
        SPEC, result.trace["charge_mw"], result.trace["discharge_mw"], prices
    )
    assert result.objective_gbp == pytest.approx(replay["net_revenue_gbp"], abs=1e-6)


def test_relaxation_bounds_the_integer_optimum_from_above():
    rng = np.random.default_rng(17)
    prices = rng.normal(40, 100, 48)   # deliberately many negative prices
    integer = solve_dispatch(prices, SPEC)
    relaxed = solve_dispatch(prices, SPEC, relax_binaries=True)
    assert relaxed.objective_gbp >= integer.objective_gbp - 1e-6


def test_degradation_cost_reduces_throughput():
    rng = np.random.default_rng(19)
    prices = rng.normal(80, 40, 48)
    free = solve_dispatch(prices, SPEC, degradation_cost_gbp_per_mwh=0.0)
    costly = solve_dispatch(prices, SPEC, degradation_cost_gbp_per_mwh=60.0)
    assert costly.throughput_mwh <= free.throughput_mwh + 1e-6
    assert costly.gross_revenue_gbp <= free.gross_revenue_gbp + 1e-6


def test_nan_prices_are_refused():
    prices = _flat(50.0)
    prices[10] = np.nan
    with pytest.raises(ValueError):
        solve_dispatch(prices, SPEC)


def test_series_index_is_preserved_in_the_trace():
    idx = pd.date_range("2024-10-01", periods=48, freq="30min", tz="UTC")
    prices = pd.Series(np.linspace(20, 200, 48), index=idx)
    result = solve_dispatch(prices, SPEC)
    assert result.trace.index.equals(idx)
