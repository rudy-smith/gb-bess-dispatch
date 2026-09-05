"""Physical model of a grid-scale lithium-ion battery energy storage system.

The module owns two things: the specification of the asset, and an independent
simulator that replays a dispatch schedule to recompute state of charge and
revenue. The simulator exists to check the optimiser rather than to replace it,
so it deliberately shares no code with the solver: an error common to both would
otherwise cancel out and pass validation.

Sign and measurement conventions, fixed here and used everywhere downstream:

  * Power is measured in MW at the grid connection point, not at the cell
    terminals. Charging is import, discharging is export. Money changes hands at
    the meter, so the traded quantity and the settled quantity are the same
    number and no conversion is needed in the revenue accounting.
  * Energy is MWh, obtained as power multiplied by the timestep in hours.
  * Charge and discharge are separate non-negative variables rather than one
    signed variable, because the efficiency losses differ in direction and a
    single signed variable cannot carry both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatterySpec:
    """Physical specification of the asset.

    Attributes
    ----------
    power_mw:
        Maximum import and export power at the connection point. Assumed
        symmetric, which is typical for a grid-scale lithium-ion system but is
        an assumption the sensitivity analysis can relax.
    duration_h:
        Energy capacity expressed as hours at full power. A "2-hour system" is
        a 1 MW / 2 MWh asset. Duration rather than raw MWh is the industry's
        unit and makes per-MW revenue figures directly comparable.
    round_trip_efficiency:
        Energy exported divided by energy imported over a full cycle, measured
        at the connection point. Includes inverter and transformer losses.
    soc_min_frac, soc_max_frac:
        Usable state-of-charge window as a fraction of nameplate energy.
        Operators reserve headroom at both ends to limit calendar ageing, so
        the usable window is narrower than the nameplate.
    initial_soc_frac:
        State of charge at the first period of the horizon.
    terminal_soc_frac:
        State of charge required at the end of the horizon. None leaves it
        free.
    """

    power_mw: float = 1.0
    duration_h: float = 2.0
    round_trip_efficiency: float = 0.90
    soc_min_frac: float = 0.0
    soc_max_frac: float = 1.0
    initial_soc_frac: float = 0.5
    terminal_soc_frac: float | None = 0.5

    def __post_init__(self) -> None:
        if self.power_mw <= 0:
            raise ValueError("power_mw must be positive")
        if self.duration_h <= 0:
            raise ValueError("duration_h must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must lie in (0, 1]")
        if not 0 <= self.soc_min_frac < self.soc_max_frac <= 1:
            raise ValueError("require 0 <= soc_min_frac < soc_max_frac <= 1")
        for name in ("initial_soc_frac", "terminal_soc_frac"):
            value = getattr(self, name)
            if value is not None and not self.soc_min_frac <= value <= self.soc_max_frac:
                raise ValueError(f"{name}={value} lies outside the usable SoC window")

    # --- derived quantities ------------------------------------------------ #

    @property
    def energy_mwh(self) -> float:
        """Nameplate energy capacity."""
        return self.power_mw * self.duration_h

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def usable_energy_mwh(self) -> float:
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def initial_soc_mwh(self) -> float:
        return self.initial_soc_frac * self.energy_mwh

    @property
    def terminal_soc_mwh(self) -> float | None:
        if self.terminal_soc_frac is None:
            return None
        return self.terminal_soc_frac * self.energy_mwh

    @property
    def charge_efficiency(self) -> float:
        """One-way charging efficiency.

        The round-trip loss is split symmetrically between the two legs, so
        each one-way efficiency is the square root of the round trip. The split
        is a modelling convention, not a measurement: only the product is
        observable from meter data. It matters because the two legs are applied
        at different prices, so an asymmetric split would shift value between
        the buy and the sell without changing the round trip.
        """
        return math.sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    def describe(self) -> str:
        return (
            f"{self.power_mw:g} MW / {self.energy_mwh:g} MWh "
            f"({self.duration_h:g} h), round trip {self.round_trip_efficiency:.0%}, "
            f"usable SoC {self.soc_min_frac:.0%}-{self.soc_max_frac:.0%}"
        )


def simulate_schedule(
    spec: BatterySpec,
    charge_mw: np.ndarray | pd.Series,
    discharge_mw: np.ndarray | pd.Series,
    prices: np.ndarray | pd.Series,
    dt_hours: float = 0.5,
    degradation_cost_gbp_per_mwh: float = 0.0,
) -> dict[str, object]:
    """Replay a dispatch schedule and recompute state of charge and revenue.

    Written independently of the optimiser so that agreement between the two is
    evidence rather than tautology. Returns the per-period trace and the
    aggregate figures.

    State of charge evolves as

        soc[t] = soc[t-1] + eta_c * charge[t] * dt - discharge[t] * dt / eta_d

    Charging loses energy on the way in, so less arrives than was imported.
    Discharging loses energy on the way out, so more must be drawn from storage
    than reaches the meter — hence division rather than multiplication.
    """
    charge = np.asarray(charge_mw, dtype=float)
    discharge = np.asarray(discharge_mw, dtype=float)
    price = np.asarray(prices, dtype=float)
    if not (len(charge) == len(discharge) == len(price)):
        raise ValueError("charge, discharge and prices must have equal length")

    eta_c = spec.charge_efficiency
    eta_d = spec.discharge_efficiency

    soc = np.empty(len(price), dtype=float)
    level = spec.initial_soc_mwh
    for t in range(len(price)):
        level += eta_c * charge[t] * dt_hours - discharge[t] * dt_hours / eta_d
        soc[t] = level

    export_mwh = discharge * dt_hours
    import_mwh = charge * dt_hours
    gross_revenue = float(np.sum(price * (export_mwh - import_mwh)))
    throughput_mwh = float(np.sum(export_mwh))
    degradation_cost = degradation_cost_gbp_per_mwh * throughput_mwh

    trace = pd.DataFrame(
        {
            "price_gbp_mwh": price,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "import_mwh": import_mwh,
            "export_mwh": export_mwh,
            "soc_mwh": soc,
            "cashflow_gbp": price * (export_mwh - import_mwh),
        }
    )
    if isinstance(prices, pd.Series):
        trace.index = prices.index

    return {
        "trace": trace,
        "gross_revenue_gbp": gross_revenue,
        "degradation_cost_gbp": degradation_cost,
        "net_revenue_gbp": gross_revenue - degradation_cost,
        "throughput_mwh": throughput_mwh,
        "equivalent_full_cycles": throughput_mwh / spec.usable_energy_mwh,
        "final_soc_mwh": float(soc[-1]) if len(soc) else spec.initial_soc_mwh,
    }


def check_feasibility(
    spec: BatterySpec,
    result: dict[str, object],
    dt_hours: float = 0.5,
    tol: float = 1e-6,
) -> list[str]:
    """Return a list of physical constraint violations; empty means feasible.

    Returning violations rather than raising lets a caller report every problem
    at once instead of only the first.
    """
    trace: pd.DataFrame = result["trace"]  # type: ignore[assignment]
    problems: list[str] = []

    if (trace["charge_mw"] < -tol).any() or (trace["discharge_mw"] < -tol).any():
        problems.append("negative power in schedule")
    if (trace["charge_mw"] > spec.power_mw + tol).any():
        problems.append("charge power exceeds rating")
    if (trace["discharge_mw"] > spec.power_mw + tol).any():
        problems.append("discharge power exceeds rating")

    simultaneous = (trace["charge_mw"] > tol) & (trace["discharge_mw"] > tol)
    if simultaneous.any():
        problems.append(f"simultaneous charge and discharge in {int(simultaneous.sum())} periods")

    if (trace["soc_mwh"] < spec.soc_min_mwh - tol).any():
        problems.append("state of charge below lower bound")
    if (trace["soc_mwh"] > spec.soc_max_mwh + tol).any():
        problems.append("state of charge above upper bound")

    terminal = spec.terminal_soc_mwh
    if terminal is not None and abs(float(trace["soc_mwh"].iloc[-1]) - terminal) > 1e-4:
        problems.append("terminal state of charge not met")

    return problems
