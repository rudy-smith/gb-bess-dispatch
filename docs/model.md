# Model formulation and design decisions

Reasoning behind each modelling choice, what the obvious alternative was, and what it would
have cost.

## Battery model

### Power measured at the grid connection point

Every power variable is what the meter sees, not what the cells see. Charging is import,
discharging is export.

The consequence is that the revenue expression contains no efficiency term at all: the asset
buys `charge × dt` MWh and sells `discharge × dt` MWh at the posted price. Round-trip
efficiency enters exclusively through the energy balance.

**Alternative:** measure at the cell terminals. This forces an efficiency factor into both the
energy balance and the cashflow. The most common error in these models is applying the loss
twice, and it is invisible in the output — revenue is simply too low by a plausible-looking
margin. Choosing the meter side makes that error structurally impossible rather than
something to be careful about.

### Round-trip loss split symmetrically

One-way efficiencies are `√η` on each leg, so their product is the round trip.

Only the product is observable from meter data. The split is therefore a convention, not a
measurement. It matters because the two legs are applied at different prices: a 95%/95% split
and a 100%/90% split have identical round trips but move value between the buy and the sell.
Symmetric is the standard choice and the sensitivity analysis can perturb it.

### Separate charge and discharge variables

Charge and discharge are two non-negative variables rather than one signed variable.

**Alternative:** a single signed power variable is more compact. But the energy balance needs
`×η` when the variable is positive and `÷η` when negative, and that is a conditional.
Conditionals are not linear. Splitting into two variables is what makes the balance expressible
in a linear programme at all.

## MILP formulation

For periods `t = 1..T` of length `dt` hours with price `p_t`:

```
maximise   Σ p_t (d_t − c_t) dt  −  k Σ d_t dt

subject to  s_t = s_{t−1} + η_c c_t dt − d_t dt / η_d      energy balance
            s_min ≤ s_t ≤ s_max                            usable window
            0 ≤ c_t ≤ P u_t                                charge limit
            0 ≤ d_t ≤ P (1 − u_t)                          discharge limit
            s_0 = s_init,  s_T = s_term                    boundary
            u_t ∈ {0,1}
```

`c` and `d` are charge and discharge power at the meter, `s` is stored energy, `k` is the
degradation cost per MWh discharged, `P` is the power rating.

Charging loses energy on the way in, so less arrives than was imported — hence `× η_c`.
Discharging loses energy on the way out, so more must be drawn from storage than reaches the
meter — hence `÷ η_d`.

Solved with PuLP and the HiGHS solver, one settlement day at a time.

### Why the binary variable is necessary

Without `u`, the linear programme can set charge and discharge equal in the same period. Net
meter flow is zero, so no cash changes hands, but the energy balance still loses
`x·dt·(1/η_d − η_c)` MWh. That is a free energy sink: a way to destroy stored energy at no
cost, which the physical asset does not offer.

It becomes valuable exactly when the optimiser wants to shed energy without selling it — most
obviously under negative prices with a terminal state-of-charge constraint to satisfy. The
sample contains hundreds of negative-price periods, so this is not hypothetical.

The linear relaxation is available via `solve_dispatch(..., relax_binaries=True)` and is
asserted in the test suite to be an upper bound on the integer optimum.

### Big-M is the power rating

In `c_t ≤ P·u_t`, the constant `P` is the big-M. It is the tightest valid bound.

A loose constant — say 10⁶ — gives an identical integer feasible set but a much weaker linear
relaxation. Branch-and-bound then explores far more nodes, and on some solvers the constraint
becomes numerically degenerate. Deriving M from a physical bound rather than picking a large
number is the defence against both.

### Day-boundary state of charge

Closing state of charge is forced equal to the opening 50%.

**Alternatives and their costs:**

- *Leave it free.* The optimiser liquidates its inventory in the final period of every day
  regardless of price — a trade it could only make once, repeated 731 times. Inflates revenue.
- *Force it to zero.* Arbitrary, and wastes the option value of holding charge overnight.
- *Chain state of charge across the whole period.* Most realistic, but daily revenues are no
  longer additive, so the monthly breakdown becomes ill-defined and the single-day solve
  becomes a single 35,088-period solve.

Matching start to end makes each day self-contained and daily revenues additive. The cost is
genuine overnight arbitrage between a cheap night and the following evening peak, which this
formulation cannot capture. The headline is understated by that amount, currently unquantified.

### Degradation charged on discharge only

Charging the same energy on both legs would double the effective cost per cycle relative to
the £/MWh figures quoted in the literature. Degradation cost is a shadow price steering the
optimiser, not a cash expense — which is why the frontier reports gross revenue.

### Why the degradation curve has the shape it does

At zero wear cost the optimiser accepts every trade that improves the objective, including
spreads that barely compensate for the round-trip losses. Those marginal cycles carry
negligible revenue but full wear. Pricing wear removes them in order of profitability, so the
first cycles sacrificed are the least valuable ones — which is why 32% of cycling can be given
up for 4.8% of revenue.

The curve flattens near the origin because the distribution of daily spreads is heavily
right-skewed: a small number of volatile days carry most of the revenue. October 2024 shows
this directly, with a mean daily spread of £91.86 against a median of £76.61.

## Data cleaning policy

Non-destructive. No row is dropped and no price is overwritten except by short-gap
interpolation, which is recorded in a flag column.

**Reindex onto a complete half-hourly UTC grid.** Missing periods become explicit NaN rows.
In UTC there are no ambiguous or nonexistent timestamps, so one `reindex` call exposes every
hole. Under a `(date, period)` index this would require constructing the expected period count
per day first — which means solving the clock-change problem anyway, for nothing gained.

**Duplicates dropped, keeping the first.** These arise from the Elexon API filtering
inclusively at both ends of a fetch window, so both copies are the same row from the same
source. If they came from a genuine restatement, the later one would be correct.

**Interpolate gaps of at most 2 periods, linearly, never extrapolating.** One hour is
defensible because half-hourly prices are strongly autocorrelated at that horizon. A longer
limit fabricates a smooth ramp with no spread, which the optimiser then trades against, with
a bias whose sign depends on where the hole falls.

*This is look-ahead.* Interpolation uses both neighbours, so a filled value at time `t` depends
on the price at `t+1`. Acceptable for a hindsight benchmark; not acceptable inside a
forecast-driven path, where filled values must be constructed from past observations only.
Flagging every filled value is what makes that split enforceable later.

**Plausibility band of −£1,000 to £6,000/MWh** flags suspected feed corruption. Prices are
**not** winsorised. The tails are the revenue — a battery earns from the spread, and a
percentile clip deletes exactly the observations that carry it.

**Frozen-feed detection** flags runs of six or more identical prices. A stuck feed and a flat
market look identical in a price column, and only one should worry you. This matters because a
flat run destroys spread — the opposite failure mode to a spurious spike, and it depresses
rather than inflates the benchmark. Both directions are bias.

**Zeros in long runs converted to missing.** A price of exactly zero is legal in a market that
prices negatively, so a blanket conversion would destroy real observations. A long run of
identical zeros is a feed encoding absence as a value. The run-length test distinguishes them.

**Day-level gating.** The optimiser consumes whole days with a state-of-charge boundary
condition, so a day with a hole cannot be optimised honestly — the optimiser routes around the
missing period and reports revenue that assumes the market did not exist for half an hour.
Days are flagged, not deleted: the rows stay for lagged features, and a report that says "these
days were excluded, here they are" is defensible where a silently shorter dataset is not.

## Provider selection

The volume-weighted market index price was rejected in favour of APX alone.

The original rationale for the blend was that it makes the market assumption explicit rather
than pretending the battery trades on one exchange. The data contradicted it:

| | APX | N2EX |
|---|---|---|
| Volume share, Oct 2024 | 99.99% | 0.01% |
| Distinct values in 1,490 periods | 1,355 | 7 |
| Longest constant run | 2 | 715 |
| Periods reporting exactly £0.00 | 0 | 1,484 |

Correlation between the two series is −0.02, and the mean difference is −£81.93, which is
approximately the negated APX mean — the signature of one series being near-constant at zero.

A weighted average of one real input is that input. The blend was not harmless: because N2EX
carried a small non-zero weight, it shifted individual settlement prices by up to £7.73/MWh.

**What would have changed the decision:** a plausible distinct-value count, correlation near 1,
and a mean difference near zero. All three failed.

## Time indexing

The canonical index is a timezone-aware UTC `DatetimeIndex`. Settlement date and period are
carried as columns — they are the join key for Elexon and NESO data and the audit trail.
Local time is derived for calendar features and plot axes only, never used as an index.

Local time is unusable as an index because 01:00 on 27 October 2024 occurs twice, at settlement
periods 3 and 5.

### Two bugs this indexing decision produced

**`pd.Timedelta(days=1)` on a tz-aware timestamp advances 24 absolute hours, not one calendar
day.** On a clock-change day it lands on 23:00 or 01:00 local, not midnight, and a single-day
grid for 27 October returns 48 periods instead of 50. Fix: increment the naive date, then
localise. Timedelta is physics; calendar arithmetic is calendar; they diverge twice a year.

**An out-of-range settlement period does not produce an invalid timestamp.** SP50 on a normal
48-period day maps to a valid half-hour belonging to the next settlement day, colliding with
its SP2 and silently growing the frame by one row. Fix: validate `(date, period)` labels before
converting to timestamps — validate where the error is still visible as itself.

## Validation design

**The replay shares no code with the solver.** `simulate_schedule` reconstructs state of charge
and revenue from the dispatch schedule alone. If it shared code with the optimiser, an error
common to both would cancel out and pass. Agreement between two independent implementations is
evidence; agreement between one implementation and itself is not.

**Degenerate cases are checkable by hand.** With flat prices, revenue must be exactly zero —
with and without losses. A spread insufficient to compensate for the round-trip losses must be
declined. A single spike must be sold into at full power with charging placed in the cheapest
available periods. These are the tests that catch a sign error or a doubled efficiency, which
the constraint tests cannot see.

**The optimiser's curse.** A perfect-foresight optimiser sorts each day's prices and selects the
extremes, so it preferentially picks periods whose noise runs in its favour. Zero-mean input
noise therefore produces a strictly positive bias in output revenue, and it does not average
away with more data — each day contributes its own positive bias. The consequence is that
sloppy data cleaning inflates the benchmark and therefore deflates any "% of perfect foresight
captured" figure measured against it.

## Baseline comparison

The threshold rule charges below the 25th percentile and discharges above the 75th, with
thresholds computed from a trailing 7-day window that excludes the current day. Excluding the
current day matters: including it would let the rule see prices it could not have known when
the first decision of the day was taken, which is the look-ahead the perfect-foresight
benchmark admits openly and a baseline must not.

The rule does not return to its opening state of charge, so residual inventory is marked to
market at the day's mean price. Valuing it at the closing price instead would let the rule bank
an unrealised gain it never traded; the mean is the neutral choice.
