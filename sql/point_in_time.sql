-- Point-in-time frame: one row per delivery half hour, carrying the price and
-- each forecast exactly as it stood at that half hour's decision time.
--
-- Needs views: decisions (src.db.duck.register_decisions), prices,
-- wind_forecasts, demand_forecasts.
--
-- ASOF LEFT JOIN, per decision row: among forecast rows for the same target half
-- hour, take the one with the latest published_at that is strictly earlier than
-- decision_time. "Strictly" because a forecast published at the decision instant
-- itself could not have been read and acted on in the same instant. LEFT, so a
-- half hour with no qualifying forecast keeps its row with NULLs instead of
-- disappearing: a silently shorter frame would hide exactly the cases that matter.
--
-- published_at is carried out beside every value, so any consumer can re-check
-- the rule rather than trust this query.

SELECT
    d.target_time,
    d.settlement_date,
    d.settlement_period,
    d.decision_time,

    p.price_apx,
    p.period_usable,
    p.day_usable,

    w.wind_forecast_mw,
    w.wind_capacity_mw,
    w.wind_forecast_share,
    w.published_at          AS wind_published_at,
    w.published_at_suspect  AS wind_suspect,

    m.demand_forecast_mw,
    m.published_at          AS demand_published_at,
    m.published_at_suspect  AS demand_suspect

FROM decisions AS d
LEFT JOIN prices AS p
    ON p.start_time_utc = d.target_time
ASOF LEFT JOIN wind_forecasts AS w
    ON  w.target_time = d.target_time
    AND d.decision_time > w.published_at
ASOF LEFT JOIN demand_forecasts AS m
    ON  m.target_time = d.target_time
    AND d.decision_time > m.published_at
ORDER BY d.target_time
