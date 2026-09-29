"""
The feature list, with the availability rule for each feature written down.

This is the register the look-ahead audit works from: every feature the model
may see is named here, with where it comes from and why it is known at the
decision time. A column that reaches a model without an entry here is a bug,
and check_feature_frame refuses it.

Decision time: 11:00 London on D-1 for every half hour of delivery day D.
Publication lag: a price or metered output is treated as public PUBLICATION_LAG
minutes after its half hour ends. Elexon publishes both within minutes; an hour
is deliberately generous, and the choice cannot create look-ahead because every
lag join requires availability strictly before the decision.
"""

from __future__ import annotations

import pandas as pd

from src.fetch.neso import DEMAND_OUTTURN_COLUMNS

PUBLICATION_LAG_MINUTES = 60

TARGET = "target_price"

# name -> (source, why it is known at the decision time)
FEATURES: dict[str, tuple[str, str]] = {
    "f_settlement_period": ("calendar", "fixed by the clock"),
    "f_minute_of_day": ("calendar", "fixed by the clock; local time"),
    "f_day_of_week": ("calendar", "fixed by the clock"),
    "f_is_weekend": ("calendar", "fixed by the clock"),
    "f_month": ("calendar", "fixed by the clock"),
    "f_wind_forecast_mw": ("NESO day-ahead wind", "as-of join: published before decision"),
    "f_wind_forecast_share": ("NESO day-ahead wind", "as-of join: published before decision"),
    "f_demand_forecast_mw": ("NESO day-ahead demand", "as-of join: published before decision"),
    "f_residual_demand_mw": ("NESO demand minus wind", "both inputs published before decision"),
    "f_price_d2_same_time": ("MID price, D-2", "D-2 ends before D-1 01:00; join requires it"),
    "f_price_d7_same_time": ("MID price, D-7", "join requires availability"),
    "f_price_7d_same_time_mean": ("MID prices, D-8..D-2", "join requires availability"),
    "f_price_d2_mean": ("MID prices, D-2", "join requires the whole day available"),
    "f_price_d2_spread": ("MID prices, D-2", "join requires the whole day available"),
    "f_price_7d_mean": ("MID prices, D-8..D-2", "join requires availability"),
    "f_price_7d_std": ("MID prices, D-8..D-2", "join requires availability"),
    "f_last_known_price": ("MID price, latest", "as-of join on availability"),
    "f_wind_err_d2": ("NESO forecast minus FUELHH outturn, D-2", "join requires the whole day"),
    "f_wind_bias_28d": ("same, D-29..D-2", "join requires availability"),
    "f_wind_err_d2_demeaned": ("f_wind_err_d2 minus f_wind_bias_28d", "both inputs available"),
}

IDENTIFIERS = ["target_time", "settlement_date", "settlement_period", "decision_time"]
FLAGS = ["period_usable", "day_usable", "wind_suspect"]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("f_")]


def check_feature_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    """Refuse a feature frame that does not match the register.

    - every f_ column is registered, and every registered feature is present;
    - no column that describes the outturn of the same half hour is present;
    - one row per delivery half hour.
    """
    present = set(feature_columns(frame))
    unregistered = present - set(FEATURES)
    missing = set(FEATURES) - present
    if unregistered or missing:
        raise ValueError(
            f"feature frame does not match the register: unregistered {sorted(unregistered)}, "
            f"missing {sorted(missing)}"
        )

    forbidden = set(DEMAND_OUTTURN_COLUMNS) | {"gen_wind_mw", "wind_outturn_mw"}
    leaked = forbidden & set(frame.columns)
    if leaked:
        raise ValueError(f"outturn columns present in the feature frame: {sorted(leaked)}")

    if len(frame) != expected_rows or not frame["target_time"].is_unique:
        raise ValueError(
            f"feature frame has {len(frame)} rows, expected {expected_rows} unique half hours"
        )
    