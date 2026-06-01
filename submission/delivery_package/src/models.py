"""
Forecasting models used in the stage 2 baseline analysis.

The functions are intentionally simple and auditable. They are designed for
short-horizon mathematical-modeling competition forecasts rather than for a
production forecasting service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """Small descriptor for a forecasting method."""

    name: str
    label: str


MODEL_SPECS = {
    "moving_average_7": ModelSpec("moving_average_7", "移动平均(7日)"),
    "same_weekday_mean_8": ModelSpec("same_weekday_mean_8", "同星期均值(近8周)"),
    "exp_smoothing": ModelSpec("exp_smoothing", "简单指数平滑"),
}


def moving_average_prediction(history: pd.Series, window: int = 7) -> float:
    """Forecast with the mean of the latest window observations."""
    history = pd.Series(history).dropna()
    if history.empty:
        return 0.0
    return float(history.tail(window).mean())


def same_weekday_mean_prediction(
    series: pd.DataFrame,
    target_date: pd.Timestamp,
    target_col: str,
    weeks: int = 8,
) -> float:
    """Forecast with the mean of previous observations from the same weekday."""
    target_date = pd.Timestamp(target_date)
    history = series[series["date"] < target_date].copy()
    same_weekday = history[history["date"].dt.dayofweek == target_date.dayofweek]
    if not same_weekday.empty:
        return float(same_weekday[target_col].tail(weeks).mean())
    return moving_average_prediction(history[target_col], window=7)


def exp_smoothing_fitted_value(history: Iterable[float], alpha: float) -> float:
    """Return the latest simple exponential smoothing level."""
    values = pd.Series(history, dtype=float).dropna().to_numpy()
    if len(values) == 0:
        return 0.0
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def optimize_exp_smoothing_alpha(
    history: pd.Series,
    candidates: Iterable[float] = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2)),
    min_train_points: int = 30,
) -> float:
    """Choose alpha by one-step-ahead MAE inside the training history."""
    history = pd.Series(history, dtype=float).dropna().reset_index(drop=True)
    if len(history) <= min_train_points + 1:
        return 0.3

    best_alpha = 0.3
    best_mae = np.inf
    for alpha in candidates:
        errors = []
        level = float(history.iloc[0])
        for idx in range(1, len(history)):
            pred = level
            actual = float(history.iloc[idx])
            if idx >= min_train_points:
                errors.append(abs(actual - pred))
            level = alpha * actual + (1.0 - alpha) * level
        score = float(np.mean(errors)) if errors else np.inf
        if score < best_mae:
            best_mae = score
            best_alpha = float(alpha)
    return best_alpha


def exp_smoothing_prediction(history: pd.Series, alpha: float | None = None) -> float:
    """Forecast the next value with simple exponential smoothing."""
    history = pd.Series(history, dtype=float).dropna()
    if history.empty:
        return 0.0
    if alpha is None:
        alpha = optimize_exp_smoothing_alpha(history)
    return float(exp_smoothing_fitted_value(history, alpha))


def prepare_series_table(
    df: pd.DataFrame,
    id_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Aggregate a panel to one row per date and series id."""
    out = (
        df.groupby(["date", *id_cols], dropna=False)[target_col]
        .sum()
        .reset_index()
        .sort_values([*id_cols, "date"])
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def validate_univariate_models(
    series_df: pd.DataFrame,
    id_cols: list[str],
    target_col: str,
    validation_start: str | pd.Timestamp,
    ma_window: int = 7,
    weekday_weeks: int = 8,
) -> pd.DataFrame:
    """Rolling one-step validation for the three stage-2 models.

    Every validation prediction uses only observations whose date is earlier
    than the target date. This prevents time-series leakage.
    """
    validation_start = pd.Timestamp(validation_start)
    rows = []
    for keys, group in series_df.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("date").reset_index(drop=True)
        train_history = group[group["date"] < validation_start][target_col]
        alpha = optimize_exp_smoothing_alpha(train_history)

        for _, row in group[group["date"] >= validation_start].iterrows():
            target_date = row["date"]
            history = group[group["date"] < target_date]
            if history.empty:
                continue

            preds = {
                "moving_average_7": moving_average_prediction(history[target_col], ma_window),
                "same_weekday_mean_8": same_weekday_mean_prediction(
                    group, target_date, target_col, weekday_weeks
                ),
                "exp_smoothing": exp_smoothing_prediction(history[target_col], alpha),
            }
            for model_name, pred in preds.items():
                result = dict(zip(id_cols, keys))
                result.update(
                    {
                        "date": target_date,
                        "model": model_name,
                        "model_label": MODEL_SPECS[model_name].label,
                        "actual": float(row[target_col]),
                        "prediction": max(0.0, float(pred)),
                        "alpha": alpha if model_name == "exp_smoothing" else np.nan,
                    }
                )
                rows.append(result)
    return pd.DataFrame(rows)


def forecast_future_univariate(
    series_df: pd.DataFrame,
    id_cols: list[str],
    target_col: str,
    future_dates: Iterable[pd.Timestamp],
    model_name: str,
    ma_window: int = 7,
    weekday_weeks: int = 8,
) -> pd.DataFrame:
    """Forecast future dates for each series with one chosen model."""
    future_dates = [pd.Timestamp(date) for date in future_dates]
    rows = []
    for keys, group in series_df.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("date").reset_index(drop=True)
        alpha = optimize_exp_smoothing_alpha(group[target_col])
        for target_date in future_dates:
            if model_name == "moving_average_7":
                pred = moving_average_prediction(group[target_col], ma_window)
            elif model_name == "same_weekday_mean_8":
                pred = same_weekday_mean_prediction(group, target_date, target_col, weekday_weeks)
            elif model_name == "exp_smoothing":
                pred = exp_smoothing_prediction(group[target_col], alpha)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            row = dict(zip(id_cols, keys))
            row.update(
                {
                    "date": target_date,
                    "model": model_name,
                    "model_label": MODEL_SPECS[model_name].label,
                    "prediction": max(0.0, float(pred)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a preprocessing transformer for tabular forecasting models."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_ridge_forecaster(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float = 10.0,
) -> Pipeline:
    """Ridge regression pipeline for the stage 5 comprehensive model."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features, True)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_random_forest_forecaster(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Small random forest pipeline for feature-rich short-horizon forecasts."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features, False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=12,
                    min_samples_leaf=8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def clipped_predict(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Predict and clip negative sales forecasts to zero."""
    return np.maximum(0.0, np.asarray(model.predict(x), dtype=float))
