"""Single-horizon mixed-integer linear programme for battery dispatch.

Formulation, for periods t = 1..T of length dt hours, with price p_t:

    maximise   sum_t p_t * (d_t - c_t) * dt  -  k * sum_t d_t * dt

    subject to
        s_t = s_{t-1} + eta_c * c_t * dt - d_t * dt / eta_d      energy balance
        s_min <= s_t <= s_max                                    usable window
        0 <= c_t <= P * u_t                                      charge limit
        0 <= d_t <= P * (1 - u_t)                                discharge limit
        s_0 = s_init,  s_T = s_term                              boundary
        u_t in {0, 1}

where c_t and d_t are charge and discharge power at the connection point, s_t is
stored energy, and k is the degradation cost per MWh discharged.

The objective is linear and every constraint is linear, so with u relaxed this
is a linear programme; the binary u is what makes it mixed-integer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from src.model.battery import BatterySpec, check_feasibility, simulate_schedule

LOGGER = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Solved schedule plus everything needed to audit it."""

    status: str
    objective_gbp: float
    trace: pd.DataFrame
    gross_revenue_gbp: float
    degradation_cost_gbp: float
    net_revenue_gbp: float
    throughput_mwh: float
    equivalent_full_cycles: float
    binaries_relaxed: bool
    violations: list[str] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.status == "Optimal"

    def summary(self) -> str:
        return (
            f"status                 {self.status}\n"
            f"objective              GBP {self.objective_gbp:,.2f}\n"
            f"gross revenue          GBP {self.gross_revenue_gbp:,.2f}\n"
            f"degradation cost       GBP {self.degradation_cost_gbp:,.2f}\n"
            f"net revenue            GBP {self.net_revenue_gbp:,.2f}\n"
            f"throughput             {self.throughput_mwh:,.2f} MWh\n"
            f"equivalent full cycles {self.equivalent_full_cycles:,.2f}\n"
            f"violations             {self.violations or 'none'}"
        )


def get_solver(msg: bool = False, time_limit: int | None = None) -> pulp.LpSolver:
    """Return HiGHS if available, otherwise the CBC bundled with PuLP.

    HiGHS is preferred because it is faster on this class of problem and is
    actively maintained; CBC is retained as a fallback so the repository runs on
    a machine without highspy installed.
    """
    for factory in (
        lambda: pulp.HiGHS(msg=msg, timeLimit=time_limit),
        lambda: pulp.HiGHS_CMD(msg=msg, timeLimit=time_limit),
        lambda: pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit),
    ):
        try:
            solver = factory()
            if solver.available():
                return solver
        except Exception:  # noqa: BLE001 - probing for an optional backend
            continue
    raise RuntimeError("no MILP solver available; install highspy or use CBC")


def solve_dispatch(
    prices: pd.Series | np.ndarray,
    spec: BatterySpec,
    dt_hours: float = 0.5,
    degradation_cost_gbp_per_mwh: float = 0.0,
    relax_binaries: bool = False,
    solver: pulp.LpSolver | None = None,
    problem_name: str = "bess_dispatch",
) -> DispatchResult:
    """Optimise dispatch over one horizon and audit the result.

    Parameters
    ----------
    prices:
        Price per MWh for each period, in chronological order. Perfect foresight
        is implied by passing outturn prices; passing forecasts turns the same
        function into the forecast-driven optimiser with no change to the model.
    relax_binaries:
        Replace the integrality requirement with a continuous variable on [0, 1].
        Used to demonstrate what the binary is preventing, and to obtain an
        upper bound on the MILP objective.
    """
    price_values = np.asarray(prices, dtype=float)
    if np.isnan(price_values).any():
        raise ValueError("prices contain NaN; the horizon must be complete before solving")
    horizon = len(price_values)
    if horizon == 0:
        raise ValueError("empty horizon")

    eta_c = spec.charge_efficiency
    eta_d = spec.discharge_efficiency
    power = spec.power_mw

    model = pulp.LpProblem(problem_name, pulp.LpMaximize)
    periods = range(horizon)

    # --- decision variables ------------------------------------------------ #
    # Charge and discharge are bounded by the power rating directly, so the
    # variable bounds already imply the physical limit; the binary constraints
    # below add only the mutual exclusion.
    charge = pulp.LpVariable.dicts("charge_mw", periods, lowBound=0, upBound=power)
    discharge = pulp.LpVariable.dicts("discharge_mw", periods, lowBound=0, upBound=power)
    soc = pulp.LpVariable.dicts(
        "soc_mwh", periods, lowBound=spec.soc_min_mwh, upBound=spec.soc_max_mwh
    )
    if relax_binaries:
        mode = pulp.LpVariable.dicts("mode", periods, lowBound=0, upBound=1, cat=pulp.LpContinuous)
    else:
        mode = pulp.LpVariable.dicts("mode", periods, cat=pulp.LpBinary)

    # --- objective --------------------------------------------------------- #
    # Revenue is settled at the meter, so the traded volume is the untransformed
    # charge and discharge power. Efficiency enters through the energy balance
    # below, never through the cashflow: applying it in both places would double
    # count the loss.
    trading = pulp.lpSum(
        price_values[t] * (discharge[t] - charge[t]) * dt_hours for t in periods
    )
    # Degradation is charged on the discharge leg only. Charging the same energy
    # on both legs would double the effective cost of a cycle relative to the
    # per-MWh figure quoted in the literature.
    wear = pulp.lpSum(
        degradation_cost_gbp_per_mwh * discharge[t] * dt_hours for t in periods
    )
    model += trading - wear, "net_revenue"

    # --- energy balance ---------------------------------------------------- #
    for t in periods:
        previous = spec.initial_soc_mwh if t == 0 else soc[t - 1]
        model += (
            soc[t]
            == previous
            + eta_c * charge[t] * dt_hours
            - discharge[t] * dt_hours / eta_d
        ), f"energy_balance_{t}"

    # --- mutual exclusion -------------------------------------------------- #
    # Without these, the solver may charge and discharge in the same period to
    # dissipate stored energy through the efficiency loss at no cash cost. That
    # is a free energy sink which the physical asset does not offer for free.
    # The big-M constant is the power rating itself, which is the tightest valid
    # bound: a larger M would weaken the linear relaxation and slow the branch
    # and bound without changing the feasible integer set.
    for t in periods:
        model += charge[t] <= power * mode[t], f"charge_mode_{t}"
        model += discharge[t] <= power * (1 - mode[t]), f"discharge_mode_{t}"

    # --- terminal condition ------------------------------------------------ #
    if spec.terminal_soc_mwh is not None:
        model += soc[horizon - 1] == spec.terminal_soc_mwh, "terminal_soc"

    solver = solver or get_solver()
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        LOGGER.warning("solver returned status %s", status)

    charge_values = np.array([charge[t].value() or 0.0 for t in periods])
    discharge_values = np.array([discharge[t].value() or 0.0 for t in periods])

    # --- independent audit ------------------------------------------------- #
    # The schedule is replayed through the simulator, which reconstructs state
    # of charge and revenue without reference to the solver's own variables. Any
    # disagreement indicates a formulation error rather than a solver error.
    replay = simulate_schedule(
        spec,
        charge_values,
        discharge_values,
        price_values,
        dt_hours=dt_hours,
        degradation_cost_gbp_per_mwh=degradation_cost_gbp_per_mwh,
    )
    trace: pd.DataFrame = replay["trace"]  # type: ignore[assignment]
    if isinstance(prices, pd.Series):
        trace.index = prices.index

    objective = float(pulp.value(model.objective) or 0.0)
    violations = check_feasibility(spec, replay, dt_hours=dt_hours)
    if status == "Optimal" and abs(objective - float(replay["net_revenue_gbp"])) > 1e-6:
        violations.append(
            f"objective {objective:.10f} disagrees with replayed revenue "
            f"{replay['net_revenue_gbp']:.10f}"
        )

    return DispatchResult(
        status=status,
        objective_gbp=objective,
        trace=trace,
        gross_revenue_gbp=float(replay["gross_revenue_gbp"]),
        degradation_cost_gbp=float(replay["degradation_cost_gbp"]),
        net_revenue_gbp=float(replay["net_revenue_gbp"]),
        throughput_mwh=float(replay["throughput_mwh"]),
        equivalent_full_cycles=float(replay["equivalent_full_cycles"]),
        binaries_relaxed=relax_binaries,
        violations=violations,
    )
