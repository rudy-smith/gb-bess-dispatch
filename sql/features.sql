-- Feature frame: one row per delivery half hour, every feature computable at that
-- half hour's decision time (11:00 London on the previous day, by default).
--
-- Needs views: point_in_time, prices, generation, calendar
-- (src.db.duck.register_calendar, covering at least 30 days before the first
-- delivery day). Parameter: $lag_minutes, the delay after a half hour ENDS before
-- its price or metered output is treated as published.
--
-- The rule is enforced in the joins, not assumed. Every price and outturn row
-- carries `available_at`, and every lag join requires available_at < decision_time.
-- Setting $lag_minutes too long therefore turns features NULL; it cannot make
-- them leak. Note what that rules out: "yesterday's price profile" means D-2 here,
-- because at 11:00 on D-1 most of D-1 has not happened yet.
--
-- Lags follow the local wall clock, not UTC: price shape follows human activity,
-- and across a clock change "48 hours earlier" and "same clock time two days
-- earlier" differ by an hour. Source rows use only the first pass through the
-- repeated autumn hour, so a (local_date, minute_of_day) key names one half hour.
--
-- Columns: identifiers, target_price (the value to forecast, never a feature),
-- flags, then every feature prefixed f_.

WITH

base AS (
    SELECT pit.*, c.local_date, c.minute_of_day, c.day_of_week, c.month
    FROM point_in_time AS pit
    JOIN calendar AS c ON c.target_time = pit.target_time
),

-- Every observed price, with the instant it became public.
px AS (
    SELECT
        c.local_date,
        c.minute_of_day,
        c.first_occurrence,
        p.start_time_utc AS source_time,
        p.price_apx      AS price,
        p.start_time_utc + INTERVAL 30 MINUTE + to_minutes(CAST($lag_minutes AS INTEGER))
                         AS available_at
    FROM prices AS p
    JOIN calendar AS c ON c.target_time = p.start_time_utc
    WHERE p.price_apx IS NOT NULL
      -- A gap filled by interpolation depends on the price AFTER it, which may
      -- not be public at the decision. Filled values are never feature sources.
      AND NOT coalesce(p.price_apx_interpolated, false)
),

px_daily AS (
    SELECT local_date,
           avg(price)        AS mean,
           max(price) - min(price) AS spread,
           max(available_at) AS available_at
    FROM px
    GROUP BY local_date
),

-- One decision per delivery day.
days AS (
    SELECT DISTINCT local_date, decision_time FROM base
),

-- Trailing seven days D-8 .. D-2: overall level and volatility.
prior_days AS (
    SELECT d.local_date,
           avg(px.price)         AS mean,
           stddev_samp(px.price) AS std
    FROM days AS d
    JOIN px
      ON px.local_date BETWEEN d.local_date - INTERVAL 8 DAY AND d.local_date - INTERVAL 2 DAY
     AND px.available_at < d.decision_time
    GROUP BY d.local_date
),

-- Trailing seven days D-8 .. D-2 at the same local clock time.
prior_days_same_time AS (
    SELECT d.local_date, px.minute_of_day, avg(px.price) AS mean
    FROM days AS d
    JOIN px
      ON px.local_date BETWEEN d.local_date - INTERVAL 8 DAY AND d.local_date - INTERVAL 2 DAY
     AND px.available_at < d.decision_time
     AND px.first_occurrence
    GROUP BY d.local_date, px.minute_of_day
),

-- Daily wind forecast error as a share of capacity, positive = over-forecast.
-- The forecast is the one attached at that day's own decision time, so this is
-- the error a forecaster could actually have observed.
wind_err AS (
    SELECT c.local_date,
           avg((pit.wind_forecast_mw - g.gen_wind_mw) / nullif(pit.wind_capacity_mw, 0))
                AS err_share,
           max(g.start_time_utc) + INTERVAL 30 MINUTE + to_minutes(CAST($lag_minutes AS INTEGER))
                AS available_at
    FROM point_in_time AS pit
    JOIN generation AS g ON g.start_time_utc = pit.target_time
    JOIN calendar  AS c ON c.target_time = pit.target_time
    WHERE pit.wind_forecast_mw IS NOT NULL AND g.gen_wind_mw IS NOT NULL
    GROUP BY c.local_date
),

-- Trailing 28-day mean error, D-29 .. D-2. The forecast and the outturn measure
-- slightly different fleets, which leaves a bias of order 1 GW that changes sign
-- between years; subtracting a trailing mean removes it using past data only.
wind_bias AS (
    SELECT d.local_date, avg(w.err_share) AS bias
    FROM days AS d
    JOIN wind_err AS w
      ON w.local_date BETWEEN d.local_date - INTERVAL 29 DAY AND d.local_date - INTERVAL 2 DAY
     AND w.available_at < d.decision_time
    GROUP BY d.local_date
)

SELECT
    b.target_time,
    b.settlement_date,
    b.settlement_period,
    b.decision_time,

    b.price_apx     AS target_price,
    b.period_usable,
    b.day_usable,
    b.wind_suspect,

    -- calendar: known in advance
    b.settlement_period                   AS f_settlement_period,
    b.minute_of_day                       AS f_minute_of_day,
    b.day_of_week                         AS f_day_of_week,
    CAST(b.day_of_week >= 5 AS INTEGER)   AS f_is_weekend,
    b.month                               AS f_month,

    -- NESO forecasts, as attached by the point-in-time join
    b.wind_forecast_mw                    AS f_wind_forecast_mw,
    b.wind_forecast_share                 AS f_wind_forecast_share,
    b.demand_forecast_mw                  AS f_demand_forecast_mw,
    b.demand_forecast_mw - b.wind_forecast_mw AS f_residual_demand_mw,

    -- prices, all published before the decision
    d2.price                              AS f_price_d2_same_time,
    d7.price                              AS f_price_d7_same_time,
    wst.mean                              AS f_price_7d_same_time_mean,
    dd2.mean                              AS f_price_d2_mean,
    dd2.spread                            AS f_price_d2_spread,
    wk.mean                               AS f_price_7d_mean,
    wk.std                                AS f_price_7d_std,
    lastp.price                           AS f_last_known_price,

    -- wind forecast error, from outturn already metered
    we.err_share                          AS f_wind_err_d2,
    wb.bias                               AS f_wind_bias_28d,
    we.err_share - wb.bias                AS f_wind_err_d2_demeaned

FROM base AS b
LEFT JOIN px AS d2
       ON d2.local_date = b.local_date - INTERVAL 2 DAY
      AND d2.minute_of_day = b.minute_of_day
      AND d2.first_occurrence
      AND d2.available_at < b.decision_time
LEFT JOIN px AS d7
       ON d7.local_date = b.local_date - INTERVAL 7 DAY
      AND d7.minute_of_day = b.minute_of_day
      AND d7.first_occurrence
      AND d7.available_at < b.decision_time
LEFT JOIN px_daily AS dd2
       ON dd2.local_date = b.local_date - INTERVAL 2 DAY
      AND dd2.available_at < b.decision_time
LEFT JOIN prior_days AS wk
       ON wk.local_date = b.local_date
LEFT JOIN prior_days_same_time AS wst
       ON wst.local_date = b.local_date
      AND wst.minute_of_day = b.minute_of_day
LEFT JOIN wind_err AS we
       ON we.local_date = b.local_date - INTERVAL 2 DAY
      AND we.available_at < b.decision_time
LEFT JOIN wind_bias AS wb
       ON wb.local_date = b.local_date
ASOF LEFT JOIN px AS lastp
       ON b.decision_time > lastp.available_at
ORDER BY b.target_time
