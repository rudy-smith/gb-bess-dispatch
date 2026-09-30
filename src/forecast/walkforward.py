"""
Walk-forward evaluation: train on the past, forecast the next month, roll on.

Folds. Each test fold is one calendar month of delivery days. The model for
that month is trained on every earlier delivery day whose price was already
known when the month's first decision was taken. That decision is at 11:00 on
the day before the month starts, when the last complete day of prices is two
days before the month starts. So training stops GAP_DAYS = 2 days before the
test month, not one: the day immediately before is still being delivered.

Refitting once a month rather than before every day is a cost choice. It means
late-month forecasts use a model a few weeks stale, which understates skill
slightly and never overstates it.

Random or K-fold splits are not used anywhere. They would train on days after
the ones being forecast, and on half hours adjacent to the test half hours,
whose prices are strongly correlated with them. The skill measured would be
interpolation, not forecasting.

Target anchoring. Trees predict piecewise constants learned from the training
range, so they cannot follow a shift in the overall price level (2023 averaged
well above 2024). The model is therefore trained on the price minus its own
trailing 7-day mean (f_price_7d_mean, known at the decision time) and the
anchor is added back. The model learns shape and deviation; the level comes
from recent prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

GAP_DAYS = 2
ANCHOR = "f_price_7d_mean"

# Fixed, deliberately unremarkable settings. Tuning inside the walk-forward
# would need its own nested validation to avoid tuning on the test months; that
# is not worth the complexity until a baseline comparison says the model is
# worth refining.
LGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}


@dataclass(frozen=True)
class Fold:
    label: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive
    train_end: pd.Timestamp  # inclusive


def monthly_folds(first_test: str, last_test: str, gap_days: int = GAP_DAYS) -> list[Fold]:
    """One fold per calendar month from first_test to last_test (month starts)."""
    folds = []
    for start in pd.date_range(first_test, last_test, freq="MS"):
        end = start + pd.offsets.MonthEnd(0)
        folds.append(
            Fold(
                label=f"{start:%Y-%m}",
                test_start=start,
                test_end=end,
                train_end=start - pd.Timedelta(days=gap_days),
            )
        )
    return folds


def _split(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = pd.to_datetime(frame["settlement_date"])
    train = frame[(day <= fold.train_end) & frame["target_price"].notna()]
    test = frame[(day >= fold.test_start) & (day <= fold.test_end)]
    return train, test


def _anchor(frame: pd.DataFrame, anchored: bool) -> np.ndarray:
    if not anchored:
        return np.zeros(len(frame))
    return frame[ANCHOR].fillna(frame[ANCHOR].median()).to_numpy()


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    objective: str = "regression_l1",
    alpha: float | None = None,
    anchored: bool = True,
) -> np.ndarray:
    """Fit one LightGBM model and predict the test rows, in price units.

    The default objective is L1, so the point forecast targets the conditional
    median and is scored by MAE, the metric it is judged on. With
    objective="quantile" it fits the alpha quantile.
    """
    params = dict(LGB_PARAMS, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    y = train["target_price"].to_numpy() - _anchor(train, anchored)
    model.fit(train[features], y)
    return model.predict(test[features]) + _anchor(test, anchored)


def walk_forward(
    frame: pd.DataFrame,
    features: list[str],
    folds: list[Fold],
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    anchored: bool = True,
) -> pd.DataFrame:
    """Out-of-sample point and quantile forecasts for every test row of every fold.

    Returns the test rows with columns fold, pred_point, and pred_qNN for each
    quantile. Quantiles are fitted independently, so they can cross; they are
    sorted row by row afterwards, which leaves each quantile's own pinball loss
    no worse and removes impossible intervals.
    """
    out = []
    for fold in folds:
        train, test = _split(frame, fold)
        if train.empty or test.empty:
            continue
        test = test.copy()
        test["fold"] = fold.label
        test["pred_point"] = fit_predict(train, test, features, anchored=anchored)
        for q in quantiles:
            test[f"pred_q{round(q * 100):02d}"] = fit_predict(
                train, test, features, objective="quantile", alpha=q, anchored=anchored
            )
        out.append(test)

    result = pd.concat(out, ignore_index=True)
    qcols = [f"pred_q{round(q * 100):02d}" for q in quantiles]
    result[qcols] = np.sort(result[qcols].to_numpy(), axis=1)
    return result
