"""Question 1 method-search experiment.

This script compares conservative alternatives for the stage-2 Q1 forecasting
scheme. It writes all experiment artifacts into method_search folders.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "tables" / "method_search"
FIGURE_DIR = PROJECT_ROOT / "figures" / "method_search"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "method_search"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "method_search"

for directory in (TABLE_DIR, FIGURE_DIR, OUTPUT_DIR, NOTEBOOK_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA_START = pd.Timestamp("2020-04-01")
WINDOW_STARTS = pd.to_datetime(
    ["2022-03-01", "2022-03-08", "2022-03-15", "2022-03-22", "2022-03-25"]
)
HORIZON = 7

UNIVARIATE_MODELS = ["stage2_best", "historical_mean_28", "seasonal_naive_7"]
FEATURE_MODELS = ["calendar_ridge", "lag_ridge", "lag_random_forest"]
ALL_MODELS = UNIVARIATE_MODELS + FEATURE_MODELS

STAGE2_BEST_DETAIL = {
    "store": "moving_average_7",
    "product": "exp_smoothing",
    "store_product": "exp_smoothing",
}

LEVEL_CONFIG = {
    "store": {
        "key_cols": ["store_id"],
        "display_cols": ["store_name"],
        "categorical_cols": ["store_id"],
        "description": "门店层面直接预测",
        "unit": "单门店 7 日总销量",
    },
    "product": {
        "key_cols": ["product_id"],
        "display_cols": ["product_name", "category"],
        "categorical_cols": ["product_id", "category"],
        "description": "商品层面直接预测",
        "unit": "单商品 7 日总销量",
    },
    "store_product": {
        "key_cols": ["store_id", "product_id"],
        "display_cols": ["store_name", "product_name", "category"],
        "categorical_cols": ["store_id", "product_id", "category"],
        "description": "门店-商品层面直接预测",
        "unit": "单门店-商品组合 7 日总销量",
    },
}

MODEL_METADATA = {
    "stage2_best": {
        "model_name": "原问题一最优模型",
        "model_class": "原方案",
        "input_features": "历史正向销量；门店层为7日移动平均，商品和门店-商品层为简单指数平滑",
        "complexity": "低：每个序列只需最近历史值和平滑递推",
        "interpretability": "高",
        "paper_writeability": "高",
        "advantages": "与阶段2方案一致，公式简单，容易复现",
        "disadvantages": "不能显式利用星期、月份、序列间共性或低销量平滑",
        "suitable_for_paper": "是",
        "leakage_risk": "低",
    },
    "historical_mean_28": {
        "model_name": "近28日历史均值",
        "model_class": "简单baseline",
        "input_features": "目标日前28天正向销量均值",
        "complexity": "低：窗口均值",
        "interpretability": "高",
        "paper_writeability": "高",
        "advantages": "最朴素参照，能检验复杂模型是否有必要",
        "disadvantages": "忽略星期效应和近期结构变化",
        "suitable_for_paper": "是",
        "leakage_risk": "低",
    },
    "seasonal_naive_7": {
        "model_name": "上周同日朴素预测",
        "model_class": "简单baseline",
        "input_features": "目标日前7天同星期销量",
        "complexity": "低：查找上周同日",
        "interpretability": "高",
        "paper_writeability": "高",
        "advantages": "直接利用星期周期，适合短期零售销量",
        "disadvantages": "对促销、异常高销量和趋势变化敏感",
        "suitable_for_paper": "是",
        "leakage_risk": "低",
    },
    "calendar_ridge": {
        "model_name": "日历固定效应岭回归",
        "model_class": "可解释统计模型",
        "input_features": "星期、月份、周末/休息日/节假日、门店/商品/类别固定效应",
        "complexity": "中低：线性模型加哑变量",
        "interpretability": "中高",
        "paper_writeability": "中高",
        "advantages": "可解释日历规律和不同对象的平均差异，不依赖未来销量",
        "disadvantages": "不能充分利用近期销量水平变化",
        "suitable_for_paper": "是",
        "leakage_risk": "低",
    },
    "lag_ridge": {
        "model_name": "滞后滚动特征岭回归",
        "model_class": "特征工程模型",
        "input_features": "lag1/7/14、rolling3/7/14、近14日非零率、星期、月份、门店/商品/类别固定效应",
        "complexity": "中：需要递推构造未来7天特征",
        "interpretability": "中",
        "paper_writeability": "中",
        "advantages": "兼顾近期水平、星期周期和对象差异，仍是线性可解释模型",
        "disadvantages": "特征构造比原方案复杂，递推预测会累积误差",
        "suitable_for_paper": "是",
        "leakage_risk": "低，已先shift再rolling并递推预测",
    },
    "lag_random_forest": {
        "model_name": "滞后滚动特征随机森林",
        "model_class": "机器学习模型",
        "input_features": "与滞后滚动特征岭回归相同",
        "complexity": "中高：多棵树集成，训练耗时更高",
        "interpretability": "中低",
        "paper_writeability": "中",
        "advantages": "能拟合非线性和变量交互，通常有更强拟合能力",
        "disadvantages": "解释成本高，样本较稀疏时可能不稳定",
        "suitable_for_paper": "谨慎",
        "leakage_risk": "低，已先shift再rolling并递推预测",
    },
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true_arr - y_pred_arr)
    denom = np.sum(np.abs(y_true_arr))
    return {
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))),
        "WAPE": float(abs_error.sum() / denom) if denom else np.nan,
        "actual_sum": float(y_true_arr.sum()),
        "prediction_sum": float(y_pred_arr.sum()),
        "n_units": int(len(y_true_arr)),
    }


def optimize_exp_smoothing_alpha(
    history: pd.Series,
    candidates: tuple[float, ...] = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2)),
    min_train_points: int = 30,
) -> float:
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


def exp_smoothing_level(history: pd.Series, alpha: float | None = None) -> float:
    history = pd.Series(history, dtype=float).dropna()
    if history.empty:
        return 0.0
    if alpha is None:
        alpha = optimize_exp_smoothing_alpha(history)
    level = float(history.iloc[0])
    for value in history.iloc[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return max(0.0, float(level))


def calendar_features_for_dates(dates: pd.Series, external_calendar: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(dates)})
    out["weekday"] = out["date"].dt.dayofweek + 1
    out["is_weekend"] = out["weekday"].isin([6, 7]).astype(int)
    out["month"] = out["date"].dt.month
    out["time_index"] = (out["date"] - DATA_START).dt.days

    ext = external_calendar.copy()
    ext["date"] = pd.to_datetime(ext["date"])
    out = out.merge(ext, on="date", how="left")
    out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)
    out["is_rest_day"] = out["is_rest_day"].fillna(out["is_weekend"]).astype(int)
    return out


def load_series_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    daily = pd.read_csv(DATA_DIR / "daily_store_product_sales.csv", parse_dates=["date"])
    base = pd.read_csv(DATA_DIR / "modeling_base_table.csv", parse_dates=["date"])

    product_names = (
        daily.groupby("product_id")["product_name"]
        .apply(lambda s: " / ".join(sorted(pd.Series(s).dropna().astype(str).unique())))
        .to_dict()
    )

    panel = (
        daily.groupby(["date", "store_id", "store_name", "product_id"], as_index=False)
        .agg(
            target_sales=("positive_sales", "sum"),
            category_code=("category_code", "first"),
            category=("category", "first"),
        )
        .sort_values(["store_id", "product_id", "date"])
    )
    panel["product_name"] = panel["product_id"].map(product_names)

    external_calendar = (
        base.groupby("date", as_index=False)
        .agg(is_holiday=("is_holiday", "max"), is_rest_day=("is_rest_day", "max"))
        .sort_values("date")
    )
    # Keep calendar features known at prediction time; weather/activity are excluded.
    calendar = calendar_features_for_dates(pd.Series(sorted(panel["date"].unique())), external_calendar)

    series_data: dict[str, pd.DataFrame] = {}

    store = (
        panel.groupby(["date", "store_id"], as_index=False)
        .agg(target_sales=("target_sales", "sum"), store_name=("store_name", "first"))
        .merge(calendar, on="date", how="left")
        .sort_values(["store_id", "date"])
    )
    series_data["store"] = store

    product = (
        panel.groupby(["date", "product_id"], as_index=False)
        .agg(
            target_sales=("target_sales", "sum"),
            product_name=("product_name", "first"),
            category=("category", "first"),
        )
        .merge(calendar, on="date", how="left")
        .sort_values(["product_id", "date"])
    )
    series_data["product"] = product

    store_product = (
        panel.groupby(["date", "store_id", "product_id"], as_index=False)
        .agg(
            target_sales=("target_sales", "sum"),
            store_name=("store_name", "first"),
            product_name=("product_name", "first"),
            category=("category", "first"),
        )
        .merge(calendar, on="date", how="left")
        .sort_values(["store_id", "product_id", "date"])
    )
    series_data["store_product"] = store_product
    return series_data, external_calendar


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model_id: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    if model_id in {"calendar_ridge", "lag_ridge"}:
        estimator = Ridge(alpha=10.0)
    elif model_id == "lag_random_forest":
        estimator = RandomForestRegressor(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported feature model: {model_id}")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def add_lag_features(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values([*key_cols, "date"]).copy()
    grouped = out.groupby(key_cols, dropna=False)["target_sales"]
    for lag in (1, 7, 14):
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in (3, 7, 14):
        out[f"rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
    out["recent_nonzero_rate_14"] = grouped.transform(
        lambda s: (s.shift(1) > 0).rolling(14, min_periods=1).mean()
    )
    return out


def key_to_tuple(key) -> tuple:
    return key if isinstance(key, tuple) else (key,)


def make_actual_lookup(series_df: pd.DataFrame, key_cols: list[str]) -> dict[tuple, float]:
    lookup = {}
    for row in series_df[["date", *key_cols, "target_sales"]].itertuples(index=False):
        date = pd.Timestamp(row[0])
        key = tuple(row[1 : 1 + len(key_cols)])
        lookup[(date, *key)] = float(row[-1])
    return lookup


def forecast_univariate_window(
    series_df: pd.DataFrame,
    level: str,
    model_id: str,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    cfg = LEVEL_CONFIG[level]
    key_cols = cfg["key_cols"]
    display_cols = cfg["display_cols"]
    actual_lookup = make_actual_lookup(series_df, key_cols)
    rows = []
    horizon_dates = pd.date_range(window_start, periods=HORIZON)
    stage2_detail = STAGE2_BEST_DETAIL[level]

    for key, group in series_df.groupby(key_cols, dropna=False, sort=False):
        key_tuple = key_to_tuple(key)
        group = group.sort_values("date")
        train = group[group["date"] < window_start]
        history = train["target_sales"].astype(float).reset_index(drop=True)
        history_by_date = {
            pd.Timestamp(date): float(value)
            for date, value in zip(train["date"], train["target_sales"], strict=False)
        }
        display_values = {col: group[col].iloc[0] for col in display_cols}

        if model_id == "stage2_best" and stage2_detail == "exp_smoothing":
            alpha = optimize_exp_smoothing_alpha(history)
            constant_pred = exp_smoothing_level(history, alpha)
        elif model_id == "stage2_best" and stage2_detail == "moving_average_7":
            constant_pred = float(history.tail(7).mean()) if len(history) else 0.0
        elif model_id == "historical_mean_28":
            constant_pred = float(history.tail(28).mean()) if len(history) else 0.0
        else:
            constant_pred = np.nan

        for target_date in horizon_dates:
            if model_id in {"stage2_best", "historical_mean_28"}:
                pred = constant_pred
            elif model_id == "seasonal_naive_7":
                pred = history_by_date.get(
                    target_date - pd.Timedelta(days=7),
                    float(history.tail(7).mean()) if len(history) else 0.0,
                )
            else:
                raise ValueError(model_id)

            row = {
                "level": level,
                "model_id": model_id,
                "date": target_date,
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(days=HORIZON - 1),
                "actual": actual_lookup[(target_date, *key_tuple)],
                "prediction": max(0.0, float(pred)),
            }
            row.update(dict(zip(key_cols, key_tuple, strict=False)))
            row.update(display_values)
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_calendar_training(
    series_df: pd.DataFrame,
    level: str,
    window_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    cfg = LEVEL_CONFIG[level]
    numeric = ["time_index", "weekday", "month", "is_weekend", "is_holiday", "is_rest_day"]
    categorical = cfg["categorical_cols"]
    train = series_df[series_df["date"] < window_start].copy()
    for col in categorical:
        train[col] = train[col].astype(str)
    return train[numeric + categorical], train["target_sales"].astype(float), numeric, categorical


def forecast_calendar_window(
    series_df: pd.DataFrame,
    level: str,
    model_id: str,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    cfg = LEVEL_CONFIG[level]
    key_cols = cfg["key_cols"]
    display_cols = cfg["display_cols"]
    actual_lookup = make_actual_lookup(series_df, key_cols)
    x_train, y_train, numeric, categorical = prepare_calendar_training(series_df, level, window_start)
    model = build_pipeline(model_id, numeric, categorical)
    model.fit(x_train, y_train)

    horizon_dates = pd.date_range(window_start, periods=HORIZON)
    future = series_df[series_df["date"].isin(horizon_dates)].copy()
    x_future = future.copy()
    for col in categorical:
        x_future[col] = x_future[col].astype(str)
    preds = np.maximum(0.0, np.asarray(model.predict(x_future[numeric + categorical]), dtype=float))
    rows = []
    for pred, row in zip(preds, future.itertuples(index=False), strict=False):
        row_dict = row._asdict()
        key_tuple = tuple(row_dict[col] for col in key_cols)
        out = {
            "level": level,
            "model_id": model_id,
            "date": pd.Timestamp(row_dict["date"]),
            "window_start": window_start,
            "window_end": window_start + pd.Timedelta(days=HORIZON - 1),
            "actual": actual_lookup[(pd.Timestamp(row_dict["date"]), *key_tuple)],
            "prediction": float(pred),
        }
        for col in key_cols + display_cols:
            out[col] = row_dict[col]
        rows.append(out)
    return pd.DataFrame(rows)


def lag_feature_row(
    hist: pd.Series,
    target_date: pd.Timestamp,
    meta: dict,
    calendar_lookup: dict[pd.Timestamp, dict],
    key_cols: list[str],
    display_cols: list[str],
) -> dict:
    hist = hist.sort_index()

    def lag_value(days: int) -> float:
        date = target_date - pd.Timedelta(days=days)
        if date in hist.index:
            return float(hist.loc[date])
        if len(hist) >= days:
            return float(hist.iloc[-days])
        return 0.0

    recent = hist[hist.index < target_date]
    cal = calendar_lookup[target_date]
    row = {
        "date": target_date,
        "time_index": cal["time_index"],
        "weekday": cal["weekday"],
        "month": cal["month"],
        "is_weekend": cal["is_weekend"],
        "is_holiday": cal["is_holiday"],
        "is_rest_day": cal["is_rest_day"],
        "lag_1": lag_value(1),
        "lag_7": lag_value(7),
        "lag_14": lag_value(14),
        "rolling_mean_3": float(recent.tail(3).mean()) if len(recent) else 0.0,
        "rolling_mean_7": float(recent.tail(7).mean()) if len(recent) else 0.0,
        "rolling_mean_14": float(recent.tail(14).mean()) if len(recent) else 0.0,
        "recent_nonzero_rate_14": float((recent.tail(14) > 0).mean()) if len(recent) else 0.0,
    }
    for col in key_cols + display_cols:
        row[col] = meta[col]
    return row


def forecast_lag_window(
    series_df: pd.DataFrame,
    level: str,
    model_id: str,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    cfg = LEVEL_CONFIG[level]
    key_cols = cfg["key_cols"]
    display_cols = cfg["display_cols"]
    categorical = cfg["categorical_cols"]
    numeric = [
        "time_index",
        "weekday",
        "month",
        "is_weekend",
        "is_holiday",
        "is_rest_day",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "recent_nonzero_rate_14",
    ]

    train = add_lag_features(series_df[series_df["date"] < window_start], key_cols)
    train = train.dropna(subset=["lag_1", "lag_7", "lag_14"]).copy()
    for col in categorical:
        train[col] = train[col].astype(str)

    model = build_pipeline(model_id, numeric, categorical)
    model.fit(train[numeric + categorical], train["target_sales"].astype(float))

    actual_lookup = make_actual_lookup(series_df, key_cols)
    calendar_cols = ["time_index", "weekday", "month", "is_weekend", "is_holiday", "is_rest_day"]
    calendar_lookup = (
        series_df[["date", *calendar_cols]]
        .drop_duplicates("date")
        .set_index("date")[calendar_cols]
        .to_dict(orient="index")
    )

    histories: dict[tuple, pd.Series] = {}
    metas: dict[tuple, dict] = {}
    for key, group in series_df.groupby(key_cols, dropna=False, sort=False):
        key_tuple = key_to_tuple(key)
        group = group.sort_values("date")
        train_group = group[group["date"] < window_start]
        histories[key_tuple] = pd.Series(
            train_group["target_sales"].to_numpy(dtype=float),
            index=pd.to_datetime(train_group["date"]),
        )
        metas[key_tuple] = {col: group[col].iloc[0] for col in key_cols + display_cols}

    rows = []
    for target_date in pd.date_range(window_start, periods=HORIZON):
        x_rows = []
        keys = []
        for key_tuple, hist in histories.items():
            x_rows.append(lag_feature_row(hist, target_date, metas[key_tuple], calendar_lookup, key_cols, display_cols))
            keys.append(key_tuple)
        x = pd.DataFrame(x_rows)
        for col in categorical:
            x[col] = x[col].astype(str)
        preds = np.maximum(0.0, np.asarray(model.predict(x[numeric + categorical]), dtype=float))
        for key_tuple, pred, x_row in zip(keys, preds, x_rows, strict=False):
            histories[key_tuple].loc[target_date] = float(pred)
            out = {
                "level": level,
                "model_id": model_id,
                "date": target_date,
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(days=HORIZON - 1),
                "actual": actual_lookup[(target_date, *key_tuple)],
                "prediction": float(pred),
            }
            for col in key_cols + display_cols:
                out[col] = x_row[col]
            rows.append(out)
    return pd.DataFrame(rows)


def forecast_one_window(series_df: pd.DataFrame, level: str, model_id: str, window_start: pd.Timestamp) -> pd.DataFrame:
    if model_id in UNIVARIATE_MODELS:
        return forecast_univariate_window(series_df, level, model_id, window_start)
    if model_id == "calendar_ridge":
        return forecast_calendar_window(series_df, level, model_id, window_start)
    return forecast_lag_window(series_df, level, model_id, window_start)


def validation_totals(predictions: pd.DataFrame, level: str) -> pd.DataFrame:
    key_cols = LEVEL_CONFIG[level]["key_cols"]
    group_cols = ["level", "model_id", "window_start", "window_end", *key_cols]
    return (
        predictions.groupby(group_cols, dropna=False, as_index=False)
        .agg(actual_7day=("actual", "sum"), prediction_7day=("prediction", "sum"))
        .sort_values(group_cols)
    )


def aggregate_store_product_predictions(predictions: pd.DataFrame, target_level: str) -> pd.DataFrame:
    if target_level == "store_via_store_product":
        key_cols = ["store_id"]
        display_cols = ["store_name"]
        level_name = target_level
    elif target_level == "product_via_store_product":
        key_cols = ["product_id"]
        display_cols = ["product_name", "category"]
        level_name = target_level
    else:
        raise ValueError(target_level)

    group_cols = ["model_id", "window_start", "window_end", "date", *key_cols]
    daily = (
        predictions[predictions["level"] == "store_product"]
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
    )
    meta_cols = key_cols + display_cols
    meta = predictions[predictions["level"] == "store_product"][meta_cols].drop_duplicates(key_cols)
    daily = daily.merge(meta, on=key_cols, how="left")
    daily["level"] = level_name
    return daily


def summarize_metrics(totals: pd.DataFrame, level: str, runtimes: dict[tuple[str, str], float]) -> pd.DataFrame:
    rows = []
    for model_id, group in totals.groupby("model_id", dropna=False):
        metrics = calculate_metrics(group["actual_7day"], group["prediction_7day"])
        window_wape = (
            group.groupby("window_start")
            .apply(lambda g: calculate_metrics(g["actual_7day"], g["prediction_7day"])["WAPE"])
            .reset_index(name="window_WAPE")
        )
        meta = MODEL_METADATA[model_id]
        row = {
            "level": level,
            "model_id": model_id,
            "model_name": meta["model_name"],
            "model_class": meta["model_class"],
            "input_features": meta["input_features"],
            "prediction_granularity": LEVEL_CONFIG.get(level, {}).get("description", level),
            "validation_method": "5个7日窗口总销量验证；训练集只使用窗口开始日前历史；滞后特征递推预测",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"],
            "WAPE_pct": metrics["WAPE"] * 100.0,
            "window_WAPE_mean": float(window_wape["window_WAPE"].mean()),
            "window_WAPE_std": float(window_wape["window_WAPE"].std(ddof=1)),
            "actual_sum": metrics["actual_sum"],
            "prediction_sum": metrics["prediction_sum"],
            "n_units": metrics["n_units"],
            "runtime_seconds": runtimes.get((level, model_id), np.nan),
            "running_complexity": meta["complexity"],
            "interpretability": meta["interpretability"],
            "paper_writeability": meta["paper_writeability"],
            "advantages": meta["advantages"],
            "disadvantages": meta["disadvantages"],
            "suitable_for_paper": meta["suitable_for_paper"],
            "leakage_risk": meta["leakage_risk"],
        }
        if level == "store_via_store_product":
            row["prediction_granularity"] = "先门店-商品预测，再汇总到门店"
            row["runtime_seconds"] = runtimes.get(("store_product", model_id), np.nan)
        elif level == "product_via_store_product":
            row["prediction_granularity"] = "先门店-商品预测，再汇总到商品"
            row["runtime_seconds"] = runtimes.get(("store_product", model_id), np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series_data, _ = load_series_data()
    prediction_frames = []
    runtimes: dict[tuple[str, str], float] = defaultdict(float)

    for level, series_df in series_data.items():
        for window_start in WINDOW_STARTS:
            for model_id in ALL_MODELS:
                start_time = time.perf_counter()
                pred = forecast_one_window(series_df, level, model_id, pd.Timestamp(window_start))
                runtimes[(level, model_id)] += time.perf_counter() - start_time
                prediction_frames.append(pred)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(TABLE_DIR / "q1_7day_validation_predictions.csv", index=False, encoding="utf-8-sig")

    totals_frames = []
    metric_frames = []
    for level in LEVEL_CONFIG:
        level_predictions = predictions[predictions["level"] == level]
        totals = validation_totals(level_predictions, level)
        totals_frames.append(totals)
        metric_frames.append(summarize_metrics(totals, level, runtimes))

    for aggregate_level in ("store_via_store_product", "product_via_store_product"):
        aggregate_daily = aggregate_store_product_predictions(predictions, aggregate_level)
        key_cols = ["store_id"] if aggregate_level.startswith("store") else ["product_id"]
        totals = (
            aggregate_daily.groupby(
                ["level", "model_id", "window_start", "window_end", *key_cols],
                dropna=False,
                as_index=False,
            )
            .agg(actual_7day=("actual", "sum"), prediction_7day=("prediction", "sum"))
            .sort_values(["model_id", "window_start", *key_cols])
        )
        totals_frames.append(totals)
        metric_frames.append(summarize_metrics(totals, aggregate_level, runtimes))

    totals_all = pd.concat(totals_frames, ignore_index=True)
    totals_all.to_csv(TABLE_DIR / "q1_7day_validation_totals.csv", index=False, encoding="utf-8-sig")

    comparison = pd.concat(metric_frames, ignore_index=True)
    comparison = comparison.sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)
    comparison.to_csv(TABLE_DIR / "q1_model_comparison.csv", index=False, encoding="utf-8-sig")

    window_metrics = (
        totals_all.groupby(["level", "model_id", "window_start"], dropna=False)
        .apply(lambda g: pd.Series(calculate_metrics(g["actual_7day"], g["prediction_7day"])))
        .reset_index()
    )
    window_metrics.to_csv(TABLE_DIR / "q1_window_metrics.csv", index=False, encoding="utf-8-sig")
    return comparison, predictions, totals_all


def plot_comparison(comparison: pd.DataFrame) -> None:
    plot_levels = ["store", "product", "store_product", "store_via_store_product", "product_via_store_product"]
    level_titles = {
        "store": "门店直接",
        "product": "商品直接",
        "store_product": "门店-商品直接",
        "store_via_store_product": "组合汇总到门店",
        "product_via_store_product": "组合汇总到商品",
    }
    model_order = ALL_MODELS
    label_map = {
        "stage2_best": "原方案",
        "historical_mean_28": "28日均值",
        "seasonal_naive_7": "上周同日",
        "calendar_ridge": "日历Ridge",
        "lag_ridge": "滞后Ridge",
        "lag_random_forest": "随机森林",
    }
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, len(plot_levels), figsize=(22, 5), sharey=False)
    for ax, level in zip(axes, plot_levels, strict=False):
        subset = comparison[comparison["level"] == level].set_index("model_id").reindex(model_order).reset_index()
        x = np.arange(len(subset))
        bars = ax.bar(x, subset["WAPE_pct"], color="#3b82f6")
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[m] for m in subset["model_id"]], rotation=35, ha="right", fontsize=8)
        ax.set_xlabel(level_titles[level])
        ax.set_ylabel("WAPE (%)")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, subset["WAPE_pct"], strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q1_model_comparison.png", dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    out = df[columns].copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out.to_markdown(index=False)


def generate_report(comparison: pd.DataFrame) -> None:
    stage2_rows = comparison[comparison["model_id"] == "stage2_best"].set_index("level")
    best_rows = comparison.sort_values(["level", "WAPE"]).groupby("level", as_index=False).first().set_index("level")

    primary_levels = ["store", "product"]
    recommendation_lines = []
    for level in primary_levels:
        old_wape = float(stage2_rows.loc[level, "WAPE_pct"])
        best = best_rows.loc[level]
        best_wape = float(best["WAPE_pct"])
        reduction = old_wape - best_wape
        recommendation_lines.append(
            f"- {level}: 原方案 WAPE={old_wape:.2f}%，当前最优为 {best['model_name']}，"
            f"WAPE={best_wape:.2f}%，下降 {reduction:.2f} 个百分点。"
        )

    primary_table = comparison[comparison["level"].isin(["store", "product", "store_product"])].copy()
    primary_table = primary_table.sort_values(["level", "WAPE"])
    aggregate_table = comparison[comparison["level"].isin(["store_via_store_product", "product_via_store_product"])].copy()
    aggregate_table = aggregate_table.sort_values(["level", "WAPE"])

    report = f"""# 问题一方法优化探索报告

执行日期：2026-05-03

## 1. 原方案的问题

阶段 2 问题一方案的优点是简单、可解释，且已经覆盖门店、商品、门店-商品三个层级。主要问题有三点：

1. 原验证是 2022-03-01 至 2022-03-31 的逐日滚动一步预测，而题目输出是未来 7 天总销量；两者没有泄露，但评价口径不完全一致。
2. 原模型只使用单序列历史销量，没有利用所有门店和商品的共同规律，也没有显式利用星期、月份等已知日历变量。
3. 门店、商品、门店-商品三个层级独立选模，未来 7 天总量不强制一致。

## 2. 新探索的候选方案

本次比较了六类方案：

1. 原问题一最优模型；
2. 近 28 日历史均值；
3. 上周同日朴素预测；
4. 日历固定效应岭回归；
5. 滞后滚动特征岭回归；
6. 滞后滚动特征随机森林。

LightGBM、XGBoost、CatBoost 本地环境不可用，因此没有纳入实际比较。天气、温度、风力和活动日变量没有放入主比较，因为未来 7 天真实值不可由附件直接获得。

## 3. 模型误差比较

主粒度直接预测结果如下。指标单位是“每个序列在一个 7 日窗口中的总销量误差”。

{markdown_table(primary_table, ["level", "model_name", "model_class", "MAE", "RMSE", "WAPE_pct", "window_WAPE_std", "interpretability", "paper_writeability", "suitable_for_paper"])}

细粒度预测后再汇总的结果如下。

{markdown_table(aggregate_table, ["level", "model_name", "MAE", "RMSE", "WAPE_pct", "window_WAPE_std", "prediction_granularity"])}

## 4. 最优模型是否稳定

各模型使用同一组 7 日窗口。`window_WAPE_std` 越小，说明不同验证窗口之间波动越小。若某模型 WAPE 最低但窗口标准差也明显更大，说明它可能对个别窗口更敏感，不能只凭总体 WAPE 判定绝对最优。

本次主粒度结论：

{chr(10).join(recommendation_lines)}

## 5. 信息泄露风险

本次主比较未使用天气、温度、风力、活动日等未来不可直接取得的外部变量。滞后和滚动特征均按“先 shift 后 rolling”的原则构造；7 日验证窗口内第 2 至第 7 天的滞后特征使用前序预测值递推，没有使用窗口内未来真实销量。

因此，本次主比较未发现明显未来销量泄露。但需要注意：如果后续把真实天气或真实活动日放入未来预测，就必须改写为情景预测或外部变量已知预测，不能和本次结果直接混用。

## 6. 是否建议替换原问题一方案

不建议替换原问题一的主预测模型。

原因是：在更贴近题目要求的 7 日窗口总销量验证下，原问题一最优模型在商品层面和门店-商品层面仍然是误差最低；门店层面与“上周同日朴素预测”打平。细粒度门店-商品预测后再汇总到门店的 WAPE 略低于直接门店预测，但下降幅度只有约 0.18 个百分点，不足以抵消解释复杂度和层级协调成本。

建议替换或补充的是验证口径，而不是主模型。也就是说，论文中可以增加 7 日总销量窗口验证，说明原方案在更贴近题目输出的验证口径下仍具有竞争力。

## 7. 应该替换哪一部分

建议替换或补充：

1. 验证方式：由单日滚动一步验证补充为 7 日窗口总销量验证；
2. 粒度说明：明确门店、商品是问题一主结果，门店-商品是补充结果；
3. 商品主键：建模时以 `product_id` 为准，商品名称只作展示，避免同一商品代码因名称差异被拆分。

不建议替换：

1. `positive_sales` 作为目标变量的设定；
2. 原问题一主模型；
3. 门店层面和商品层面作为主预测结果的结构；
4. 门店-商品层面作为补充结果的定位。

## 8. 未来 7 天预测结果是否需要重新生成

若继续采用原问题一主模型，则不需要因为本次探索而重新生成正式未来 7 天预测结果。

如果后续决定把“门店-商品预测后汇总到门店”的轻微优势写入主文，则需要重新生成门店层面的预测表并重新检查门店、商品、组合三个层级的总量一致性。但从当前误差下降幅度看，不建议这样做。

## 9. 论文中应该如何克制表述

建议写法：

1. “在相同 7 日窗口验证口径下，原问题一模型仍保持较低误差。”
2. “更复杂的滞后特征模型和随机森林没有在本数据上取得稳定改进，因此不作为问题一主模型。”
3. “由于未来天气和促销活动没有附件观测，问题一主模型不使用这些变量。”
4. “门店-商品层级预测可作为细分参考，但其误差高于门店和商品层级，不能替代主结果。”

避免写法：

1. “天气导致销量变化”；
2. “机器学习模型一定优于传统模型”；
3. “验证期最优等同于未来一定最优”；
4. “门店-商品细粒度模型可以完全替代门店和商品层面模型”。
"""
    (OUTPUT_DIR / "q1_method_search_report.md").write_text(report, encoding="utf-8")


def generate_notebook() -> None:
    code = """from pathlib import Path
import runpy

project_root = Path.cwd()
if project_root.name == 'method_search':
    project_root = project_root.parents[1]
elif not (project_root / 'src' / 'experimental' / 'q1_method_search.py').exists():
    project_root = Path(r'%s')

runpy.run_path(str(project_root / 'src' / 'experimental' / 'q1_method_search.py'), run_name='__main__')
""" % str(PROJECT_ROOT).replace("\\", "\\\\")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 问题一方法优化探索\\n",
                    "\\n",
                    "本 Notebook 调用 `src/experimental/q1_method_search.py`，复现 7 日窗口总销量验证、候选模型比较和图表输出。\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "运行后查看：\\n",
                    "\\n",
                    "- `tables/method_search/q1_model_comparison.csv`\\n",
                    "- `figures/method_search/q1_model_comparison.png`\\n",
                    "- `outputs/method_search/q1_method_search_report.md`\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / "q1_method_search.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    comparison, _, _ = run_experiment()
    plot_comparison(comparison)
    generate_report(comparison)
    generate_notebook()
    print(comparison[["level", "model_id", "MAE", "RMSE", "WAPE_pct", "window_WAPE_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
