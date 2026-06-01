from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "tables" / "method_search"
FIGURE_DIR = ROOT / "figures" / "method_search"
OUTPUT_DIR = ROOT / "outputs" / "method_search"
NOTEBOOK_DIR = ROOT / "notebooks" / "method_search"

TARGET = "positive_sales"
WINDOW_LENGTH = 7
VALIDATION_WINDOWS = [
    (pd.Timestamp("2022-03-01"), pd.Timestamp("2022-03-07")),
    (pd.Timestamp("2022-03-08"), pd.Timestamp("2022-03-14")),
    (pd.Timestamp("2022-03-15"), pd.Timestamp("2022-03-21")),
    (pd.Timestamp("2022-03-22"), pd.Timestamp("2022-03-28")),
]
FUTURE_DATES = pd.date_range("2022-04-01", "2022-04-07", freq="D")

ORIGINAL_NUMERIC = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
]
BASIC_NUMERIC = ORIGINAL_NUMERIC + [
    "history_mean",
    "history_nonzero_rate",
]
EXTERNAL_NUMERIC = [
    "max_temperature",
    "min_temperature",
    "wind_power",
    "is_activity_day",
]
CONTEXT_NUMERIC = [
    "store_category_lag_7",
    "store_category_rolling_mean_7",
    "store_total_lag_7",
    "store_total_rolling_mean_7",
    "product_total_lag_7",
    "product_total_rolling_mean_7",
]
BASIC_CATEGORICAL = ["store_id_str", "product_id_str", "category"]
WEATHER_CATEGORICAL = ["weather"]


@dataclass(frozen=True)
class CandidateSpec:
    model: str
    model_label: str
    family: str
    complexity: str
    interpretability: str
    paper_writability: str
    feature_set: str
    uses_future_weather_scenario: bool
    notes: str


CANDIDATES = [
    CandidateSpec(
        "same_weekday_mean_8",
        "同星期均值(近8周)",
        "强baseline",
        "低",
        "高",
        "高",
        "history_only",
        False,
        "只用历史相同星期几销量。",
    ),
    CandidateSpec(
        "moving_average_7",
        "近7日移动平均",
        "强baseline",
        "低",
        "高",
        "高",
        "history_only",
        False,
        "只用最近 7 天历史销量。",
    ),
    CandidateSpec(
        "moving_average_14",
        "近14日移动平均",
        "强baseline",
        "低",
        "高",
        "高",
        "history_only",
        False,
        "只用最近 14 天历史销量，比 7 日均值更平滑。",
    ),
    CandidateSpec(
        "exp_smoothing",
        "问题一门店-商品指数平滑",
        "强baseline",
        "低",
        "高",
        "高",
        "history_only",
        False,
        "问题一门店-商品层级最优简单模型。",
    ),
    CandidateSpec(
        "linear_basic",
        "多元线性回归(基础特征)",
        "可解释回归",
        "中",
        "高",
        "高",
        "q4_original_external",
        True,
        "滞后、滚动、日历、门店、商品、类别和外部变量。",
    ),
    CandidateSpec(
        "ridge_basic",
        "Ridge回归(原Q4特征)",
        "可解释回归",
        "中",
        "高",
        "高",
        "q4_original_external",
        True,
        "复核原 Q4 Ridge 的严格递推表现。",
    ),
    CandidateSpec(
        "ridge_no_weather_activity",
        "Ridge回归(不含天气活动)",
        "可解释回归",
        "中",
        "高",
        "高",
        "basic_no_external",
        False,
        "不使用未来难以准确获得的天气和活动日。",
    ),
    CandidateSpec(
        "ridge_context",
        "Ridge回归(类别/门店上下文特征)",
        "可解释回归",
        "中",
        "较高",
        "较高",
        "context_external",
        True,
        "新增门店-类别、门店总量、商品全店总量的历史上下文特征。",
    ),
    CandidateSpec(
        "lasso_context",
        "Lasso回归(类别/门店上下文特征)",
        "可解释回归",
        "中",
        "较高",
        "中",
        "context_external",
        True,
        "带变量筛选倾向，但系数稳定性需谨慎。",
    ),
    CandidateSpec(
        "random_forest_context",
        "随机森林(上下文特征)",
        "树模型",
        "高",
        "中",
        "中",
        "context_external",
        True,
        "捕捉非线性和交互，论文解释比回归更困难。",
    ),
    CandidateSpec(
        "extra_trees_context",
        "ExtraTrees(上下文特征)",
        "树模型",
        "高",
        "中",
        "中",
        "context_external",
        True,
        "更随机的树集成，检验树模型稳定性。",
    ),
    CandidateSpec(
        "ridge_by_store_context",
        "按门店分别建Ridge",
        "分组模型",
        "中高",
        "中",
        "中",
        "context_external",
        True,
        "每家门店单独拟合，样本减少，可能不稳定。",
    ),
    CandidateSpec(
        "ridge_by_category_context",
        "按类别分别建Ridge",
        "分组模型",
        "中高",
        "中",
        "中",
        "context_external",
        True,
        "每个类别单独拟合，低销量类别样本不足风险更高。",
    ),
    CandidateSpec(
        "ensemble_q1_ridge_context_50",
        "指数平滑与Ridge上下文等权集成",
        "集成模型",
        "中",
        "较高",
        "较高",
        "ensemble",
        True,
        "简单平均，避免在验证集上调权。",
    ),
    CandidateSpec(
        "hybrid_scale_q1_ridge_context",
        "低销量指数平滑/高销量Ridge混合",
        "分层集成",
        "中",
        "较高",
        "中",
        "ensemble",
        True,
        "阈值由每个窗口训练期门店-商品平均销量中位数确定。",
    ),
]

SPEC_BY_MODEL = {spec.model: spec for spec in CANDIDATES}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None, digits: int = 3) -> str:
    if df is None or df.empty:
        return "无。"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append(
            "| "
            + " | ".join("" if pd.isna(row[col]) else str(row[col]) for col in cols)
            + " |"
        )
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def combine_unique_names(values: pd.Series) -> str:
    names = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    return " / ".join(names)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    product_names = data.groupby("product_id")["product_name"].apply(combine_unique_names)
    panel = (
        data.groupby(["date", "store_id", "product_id"], as_index=False)
        .agg(
            positive_sales=(TARGET, "sum"),
            store_name=("store_name", "first"),
            category=("category", lambda s: s.dropna().mode().iloc[0]),
            category_code=("category_code", lambda s: s.dropna().mode().iloc[0]),
        )
        .sort_values(["store_id", "product_id", "date"])
        .reset_index(drop=True)
    )
    panel["product_name"] = panel["product_id"].map(product_names)
    panel["weekday"] = panel["date"].dt.isocalendar().day.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(int)
    external_cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_activity_day",
        "is_rest_day",
        "is_holiday",
        "has_external_data",
    ]
    external = data[[c for c in external_cols if c in data.columns]].drop_duplicates("date")
    panel = panel.merge(external, on="date", how="left")
    for col in ["is_activity_day", "is_rest_day", "is_holiday", "has_external_data"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)
    return panel.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def add_shifted_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = df.sort_values([*group_cols, "date"]).copy()
    grouped = out.groupby(group_cols, dropna=False)[target_col]
    shifted = grouped.shift(1)
    out[f"{prefix}lag_1"] = grouped.shift(1)
    out[f"{prefix}lag_7"] = grouped.shift(7)
    out[f"{prefix}lag_14"] = grouped.shift(14)
    out[f"{prefix}rolling_mean_7"] = shifted.groupby([out[c] for c in group_cols]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out[f"{prefix}rolling_mean_14"] = shifted.groupby([out[c] for c in group_cols]).transform(
        lambda s: s.rolling(14, min_periods=1).mean()
    )
    out[f"{prefix}rolling_std_7"] = shifted.groupby([out[c] for c in group_cols]).transform(
        lambda s: s.rolling(7, min_periods=2).std()
    )
    out[f"{prefix}history_mean"] = shifted.groupby([out[c] for c in group_cols]).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    out[f"{prefix}history_nonzero_rate"] = (
        shifted.gt(0)
        .astype(float)
        .groupby([out[c] for c in group_cols])
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    return out


def prepare_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    feature = add_shifted_features(panel, ["store_id", "product_id"], TARGET, "")
    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "history_mean",
        "history_nonzero_rate",
    ]

    store_category = (
        panel.groupby(["date", "store_id", "category"], as_index=False)[TARGET]
        .sum()
        .rename(columns={TARGET: "store_category_sales"})
    )
    store_category = add_shifted_features(
        store_category,
        ["store_id", "category"],
        "store_category_sales",
        "store_category_",
    )
    keep = [
        "date",
        "store_id",
        "category",
        "store_category_lag_7",
        "store_category_rolling_mean_7",
    ]
    feature = feature.merge(store_category[keep], on=["date", "store_id", "category"], how="left")

    store_total = (
        panel.groupby(["date", "store_id"], as_index=False)[TARGET]
        .sum()
        .rename(columns={TARGET: "store_total_sales"})
    )
    store_total = add_shifted_features(store_total, ["store_id"], "store_total_sales", "store_total_")
    keep = ["date", "store_id", "store_total_lag_7", "store_total_rolling_mean_7"]
    feature = feature.merge(store_total[keep], on=["date", "store_id"], how="left")

    product_total = (
        panel.groupby(["date", "product_id"], as_index=False)[TARGET]
        .sum()
        .rename(columns={TARGET: "product_total_sales"})
    )
    product_total = add_shifted_features(product_total, ["product_id"], "product_total_sales", "product_total_")
    keep = ["date", "product_id", "product_total_lag_7", "product_total_rolling_mean_7"]
    feature = feature.merge(product_total[keep], on=["date", "product_id"], how="left")

    all_numeric = BASIC_NUMERIC + EXTERNAL_NUMERIC + CONTEXT_NUMERIC
    for col in all_numeric:
        feature[col] = pd.to_numeric(feature[col], errors="coerce").fillna(0.0)
    feature["store_id_str"] = feature["store_id"].astype(str)
    feature["product_id_str"] = feature["product_id"].astype(str)
    feature["category"] = feature["category"].astype(str)
    feature["weather"] = feature["weather"].fillna("未知").astype(str)
    return feature.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "q4_original_external":
        return ORIGINAL_NUMERIC + EXTERNAL_NUMERIC, BASIC_CATEGORICAL + WEATHER_CATEGORICAL
    if feature_set == "basic_external":
        return BASIC_NUMERIC + EXTERNAL_NUMERIC, BASIC_CATEGORICAL + WEATHER_CATEGORICAL
    if feature_set == "basic_no_external":
        return ORIGINAL_NUMERIC, BASIC_CATEGORICAL
    if feature_set == "context_external":
        return BASIC_NUMERIC + EXTERNAL_NUMERIC + CONTEXT_NUMERIC, BASIC_CATEGORICAL + WEATHER_CATEGORICAL
    raise ValueError(f"Unsupported feature set for ML model: {feature_set}")


def build_preprocessor(numeric_features: list[str], categorical_features: list[str], scale: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(model_name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    if model_name.startswith("linear"):
        estimator = LinearRegression()
        scale = True
    elif model_name.startswith("ridge"):
        estimator = Ridge(alpha=10.0)
        scale = True
    elif model_name.startswith("lasso"):
        estimator = Lasso(alpha=0.001, max_iter=5000, random_state=42)
        scale = True
    elif model_name.startswith("random_forest"):
        estimator = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        scale = False
    elif model_name.startswith("extra_trees"):
        estimator = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        scale = False
    else:
        raise ValueError(model_name)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features, scale)),
            ("model", estimator),
        ]
    )


def metric_dict(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(actual))
    return {
        "MAE": float(np.mean(np.abs(actual - pred))),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "WAPE": float(np.sum(np.abs(actual - pred)) / denom) if denom else np.nan,
    }


def optimize_alpha(values: list[float]) -> float:
    series = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(series) <= 31:
        return 0.3
    best_alpha = 0.3
    best_mae = math.inf
    for alpha in np.round(np.arange(0.1, 1.0, 0.1), 2):
        level = float(series.iloc[0])
        errors = []
        for idx in range(1, len(series)):
            pred = level
            actual = float(series.iloc[idx])
            if idx >= 30:
                errors.append(abs(actual - pred))
            level = alpha * actual + (1.0 - alpha) * level
        score = float(np.mean(errors)) if errors else math.inf
        if score < best_mae:
            best_mae = score
            best_alpha = float(alpha)
    return best_alpha


def exp_smooth(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return max(0.0, float(level))


def same_weekday_mean(history_dates: list[pd.Timestamp], values: list[float], target_date: pd.Timestamp) -> float:
    same = [
        float(value)
        for hist_date, value in zip(history_dates, values)
        if pd.Timestamp(hist_date).dayofweek == target_date.dayofweek
    ]
    if same:
        return max(0.0, float(np.mean(same[-8:])))
    return max(0.0, float(np.mean(values[-7:]))) if values else 0.0


def simple_prediction(
    model_name: str,
    values: list[float],
    history_dates: list[pd.Timestamp],
    target_date: pd.Timestamp,
    alpha: float | None = None,
) -> float:
    if model_name == "moving_average_7":
        return max(0.0, float(np.mean(values[-7:]))) if values else 0.0
    if model_name == "moving_average_14":
        return max(0.0, float(np.mean(values[-14:]))) if values else 0.0
    if model_name == "same_weekday_mean_8":
        return same_weekday_mean(history_dates, values, target_date)
    if model_name == "exp_smoothing":
        return exp_smooth(values, alpha if alpha is not None else optimize_alpha(values))
    raise ValueError(model_name)


def init_history(panel: pd.DataFrame, combos: pd.DataFrame, window_start: pd.Timestamp) -> tuple[dict, list[pd.Timestamp]]:
    before = panel[panel["date"] < window_start].sort_values("date")
    dates = sorted(before["date"].drop_duplicates().tolist())
    history = {}
    for row in combos.itertuples():
        vals = (
            before[(before["store_id"] == row.store_id) & (before["product_id"] == row.product_id)]
            .sort_values("date")[TARGET]
            .astype(float)
            .tolist()
        )
        history[(row.store_id, row.product_id)] = vals
    return history, dates


def history_stats(values: list[float]) -> dict[str, float]:
    recent7 = values[-7:] if values else [0.0]
    recent14 = values[-14:] if values else [0.0]
    return {
        "lag_1": float(values[-1]) if len(values) >= 1 else 0.0,
        "lag_7": float(values[-7]) if len(values) >= 7 else 0.0,
        "lag_14": float(values[-14]) if len(values) >= 14 else 0.0,
        "rolling_mean_7": float(np.mean(recent7)),
        "rolling_mean_14": float(np.mean(recent14)),
        "rolling_std_7": float(np.std(recent7, ddof=1)) if len(recent7) >= 2 else 0.0,
        "history_mean": float(np.mean(values)) if values else 0.0,
        "history_nonzero_rate": float(np.mean(np.asarray(values) > 0)) if values else 0.0,
    }


def aggregate_history(
    history: dict[tuple[int, int], list[float]],
    keys: list[tuple[int, int]],
) -> list[float]:
    if not keys:
        return []
    length = min(len(history[key]) for key in keys)
    if length == 0:
        return []
    arr = np.zeros(length, dtype=float)
    for key in keys:
        arr += np.asarray(history[key][-length:], dtype=float)
    return arr.tolist()


def aggregate_context_stats(values: list[float], prefix: str) -> dict[str, float]:
    recent7 = values[-7:] if values else [0.0]
    return {
        f"{prefix}lag_7": float(values[-7]) if len(values) >= 7 else 0.0,
        f"{prefix}rolling_mean_7": float(np.mean(recent7)),
    }


def build_batch_features(
    combos: pd.DataFrame,
    history: dict[tuple[int, int], list[float]],
    target_date: pd.Timestamp,
    external_lookup: dict,
) -> pd.DataFrame:
    rows = []
    ext = external_lookup[pd.Timestamp(target_date)]
    combo_keys = [(int(row.store_id), int(row.product_id)) for row in combos.itertuples()]
    keys_by_store_category: dict[tuple[int, str], list[tuple[int, int]]] = {}
    keys_by_store: dict[int, list[tuple[int, int]]] = {}
    keys_by_product: dict[int, list[tuple[int, int]]] = {}
    meta_by_key = {}
    for row in combos.itertuples():
        key = (int(row.store_id), int(row.product_id))
        meta_by_key[key] = row
        keys_by_store_category.setdefault((int(row.store_id), str(row.category)), []).append(key)
        keys_by_store.setdefault(int(row.store_id), []).append(key)
        keys_by_product.setdefault(int(row.product_id), []).append(key)

    for row in combos.itertuples():
        key = (int(row.store_id), int(row.product_id))
        own = history_stats(history[key])
        store_category_values = aggregate_history(
            history, keys_by_store_category[(int(row.store_id), str(row.category))]
        )
        store_values = aggregate_history(history, keys_by_store[int(row.store_id)])
        product_values = aggregate_history(history, keys_by_product[int(row.product_id)])
        record = {
            "date": pd.Timestamp(target_date),
            "store_id": int(row.store_id),
            "store_name": row.store_name,
            "product_id": int(row.product_id),
            "product_name": row.product_name,
            "category": str(row.category),
            **own,
            **aggregate_context_stats(store_category_values, "store_category_"),
            **aggregate_context_stats(store_values, "store_total_"),
            **aggregate_context_stats(product_values, "product_total_"),
            "weekday": int(pd.Timestamp(target_date).isocalendar().weekday),
            "month": int(pd.Timestamp(target_date).month),
            "is_weekend": int(pd.Timestamp(target_date).isocalendar().weekday in [6, 7]),
            "is_holiday": int(ext.get("is_holiday", 0)),
            "max_temperature": float(ext.get("max_temperature", 0.0)),
            "min_temperature": float(ext.get("min_temperature", 0.0)),
            "wind_power": float(ext.get("wind_power", 0.0)),
            "is_activity_day": int(ext.get("is_activity_day", 0)),
            "weather": str(ext.get("weather", "未知")),
            "store_id_str": str(row.store_id),
            "product_id_str": str(row.product_id),
        }
        rows.append(record)
    return pd.DataFrame(rows)


def external_lookup_from_feature(feature_df: pd.DataFrame) -> dict:
    external_cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_activity_day",
        "is_holiday",
    ]
    return feature_df[external_cols].drop_duplicates("date").set_index("date").to_dict("index")


def actual_lookup_from_panel(panel: pd.DataFrame) -> dict:
    return panel.set_index(["date", "store_id", "product_id"])[TARGET].astype(float).to_dict()


def append_predictions(
    rows: list[dict],
    batch: pd.DataFrame,
    predictions: np.ndarray,
    actual_lookup: dict,
    model_name: str,
    window_start: pd.Timestamp,
) -> None:
    spec = SPEC_BY_MODEL[model_name]
    for idx, row in enumerate(batch.itertuples()):
        actual = float(actual_lookup.get((pd.Timestamp(row.date), row.store_id, row.product_id), 0.0))
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(days=WINDOW_LENGTH - 1),
                "date": pd.Timestamp(row.date),
                "horizon": int((pd.Timestamp(row.date) - window_start).days + 1),
                "store_id": row.store_id,
                "store_name": row.store_name,
                "product_id": row.product_id,
                "product_name": row.product_name,
                "category": row.category,
                "actual": actual,
                "prediction": max(0.0, float(predictions[idx])),
                "model": model_name,
                "model_label": spec.model_label,
                "family": spec.family,
                "feature_set": spec.feature_set,
            }
        )


def recursive_simple_validate(
    panel: pd.DataFrame,
    combos: pd.DataFrame,
    actual_lookup: dict,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for window_start, window_end in VALIDATION_WINDOWS:
        history, history_dates = init_history(panel, combos, window_start)
        alphas = {
            (row.store_id, row.product_id): optimize_alpha(history[(row.store_id, row.product_id)])
            for row in combos.itertuples()
        }
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch = combos.copy()
            batch["date"] = date
            preds = []
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                preds.append(
                    simple_prediction(
                        model_name,
                        history[key],
                        history_dates,
                        date,
                        alphas.get(key),
                    )
                )
            preds_array = np.asarray(preds, dtype=float)
            append_predictions(rows, batch, preds_array, actual_lookup, model_name, window_start)
            for row, pred in zip(combos.itertuples(), preds_array):
                history[(row.store_id, row.product_id)].append(float(pred))
            history_dates.append(pd.Timestamp(date))
    return pd.DataFrame(rows)


def fit_global_model(train: pd.DataFrame, model_name: str, feature_set: str) -> tuple[Pipeline, list[str], list[str]]:
    numeric, categorical = feature_columns(feature_set)
    model = make_estimator(model_name, numeric, categorical)
    model.fit(train[numeric + categorical], train[TARGET])
    return model, numeric, categorical


def recursive_ml_validate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    combos: pd.DataFrame,
    actual_lookup: dict,
    model_name: str,
) -> pd.DataFrame:
    spec = SPEC_BY_MODEL[model_name]
    external_lookup = external_lookup_from_feature(feature_df)
    rows = []
    for window_start, window_end in VALIDATION_WINDOWS:
        train = feature_df[feature_df["date"] < window_start].copy()
        if spec.uses_future_weather_scenario:
            train = train[train["has_external_data"] == 1].copy()
        model, numeric, categorical = fit_global_model(train, model_name, spec.feature_set)
        history, _ = init_history(panel, combos, window_start)
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch = build_batch_features(combos, history, date, external_lookup)
            preds = np.maximum(0.0, np.asarray(model.predict(batch[numeric + categorical]), dtype=float))
            append_predictions(rows, batch, preds, actual_lookup, model_name, window_start)
            for row, pred in zip(batch.itertuples(), preds):
                history[(row.store_id, row.product_id)].append(float(pred))
    return pd.DataFrame(rows)


def recursive_grouped_ridge_validate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    combos: pd.DataFrame,
    actual_lookup: dict,
    model_name: str,
    group_col: str,
) -> pd.DataFrame:
    spec = SPEC_BY_MODEL[model_name]
    external_lookup = external_lookup_from_feature(feature_df)
    numeric, categorical = feature_columns("context_external")
    rows = []
    for window_start, window_end in VALIDATION_WINDOWS:
        train_all = feature_df[(feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)].copy()
        fallback = make_estimator("ridge_context", numeric, categorical)
        fallback.fit(train_all[numeric + categorical], train_all[TARGET])
        models = {}
        for group_value, train in train_all.groupby(group_col, dropna=False):
            if len(train) < 250:
                models[group_value] = fallback
            else:
                model = make_estimator("ridge_context", numeric, categorical)
                model.fit(train[numeric + categorical], train[TARGET])
                models[group_value] = model
        history, _ = init_history(panel, combos, window_start)
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch = build_batch_features(combos, history, date, external_lookup)
            preds = np.zeros(len(batch), dtype=float)
            for group_value, idx in batch.groupby(group_col, dropna=False).groups.items():
                model = models.get(group_value, fallback)
                idx_list = list(idx)
                preds[idx_list] = np.maximum(
                    0.0,
                    np.asarray(model.predict(batch.loc[idx_list, numeric + categorical]), dtype=float),
                )
            append_predictions(rows, batch, preds, actual_lookup, model_name, window_start)
            for row, pred in zip(batch.itertuples(), preds):
                history[(row.store_id, row.product_id)].append(float(pred))
    return pd.DataFrame(rows)


def combine_predictions(
    left: pd.DataFrame,
    right: pd.DataFrame,
    model_name: str,
    combine_func: Callable[[pd.Series, pd.Series, pd.DataFrame], pd.Series],
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    keys = ["window_start", "date", "store_id", "product_id"]
    merged = left.merge(
        right[keys + ["prediction"]],
        on=keys,
        suffixes=("_left", "_right"),
    )
    merged["prediction"] = combine_func(merged["prediction_left"], merged["prediction_right"], merged)
    spec = SPEC_BY_MODEL[model_name]
    out = merged[
        [
            "window_start",
            "window_end",
            "date",
            "horizon",
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "category",
            "actual",
            "prediction",
        ]
    ].copy()
    out["model"] = model_name
    out["model_label"] = spec.model_label
    out["family"] = spec.family
    out["feature_set"] = spec.feature_set
    return out


def make_hybrid_predictions(
    panel: pd.DataFrame,
    q1: pd.DataFrame,
    ridge: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    keys = ["window_start", "date", "store_id", "product_id"]
    merged = q1.merge(
        ridge[keys + ["prediction"]],
        on=keys,
        suffixes=("_q1", "_ridge"),
    )
    choice_records = []
    for window_start in sorted(merged["window_start"].unique()):
        hist = panel[pd.to_datetime(panel["date"]) < pd.Timestamp(window_start)]
        combo_mean = (
            hist.groupby(["store_id", "product_id"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "train_mean_sales"})
        )
        threshold = float(combo_mean["train_mean_sales"].median())
        combo_mean["use_ridge"] = combo_mean["train_mean_sales"] >= threshold
        combo_mean["window_start"] = pd.Timestamp(window_start)
        choice_records.append(combo_mean)
    choices = pd.concat(choice_records, ignore_index=True)
    merged = merged.merge(choices, on=["window_start", "store_id", "product_id"], how="left")
    merged["prediction"] = np.where(
        merged["use_ridge"].fillna(False),
        merged["prediction_ridge"],
        merged["prediction_q1"],
    )
    spec = SPEC_BY_MODEL[model_name]
    out = merged[
        [
            "window_start",
            "window_end",
            "date",
            "horizon",
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "category",
            "actual",
            "prediction",
        ]
    ].copy()
    out["model"] = model_name
    out["model_label"] = spec.model_label
    out["family"] = spec.family
    out["feature_set"] = spec.feature_set
    return out


def aggregate_for_level(preds: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "store_product":
        return preds.copy()
    if level == "store":
        group_cols = ["window_start", "date", "model", "model_label", "store_id", "store_name"]
    elif level == "product":
        group_cols = [
            "window_start",
            "date",
            "model",
            "model_label",
            "product_id",
            "product_name",
            "category",
        ]
    elif level == "category":
        group_cols = ["window_start", "date", "model", "model_label", "category"]
    else:
        raise ValueError(level)
    return (
        preds.groupby(group_cols, as_index=False, dropna=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(group_cols)
    )


def level_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in ["store_product", "store", "product", "category"]:
        agg = aggregate_for_level(preds, level)
        for (model, label), group in agg.groupby(["model", "model_label"], dropna=False):
            metrics = metric_dict(group["actual"], group["prediction"])
            spec = SPEC_BY_MODEL[model]
            rows.append(
                {
                    "level": level,
                    "model": model,
                    "model_label": label,
                    "family": spec.family,
                    **metrics,
                    "WAPE_pct": metrics["WAPE"] * 100,
                    "actual_sum": float(group["actual"].sum()),
                    "prediction_sum": float(group["prediction"].sum()),
                    "n": int(len(group)),
                    "complexity": spec.complexity,
                    "interpretability": spec.interpretability,
                    "paper_writability": spec.paper_writability,
                    "uses_future_weather_scenario": spec.uses_future_weather_scenario,
                    "notes": spec.notes,
                }
            )
    return pd.DataFrame(rows).sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def group_error_tables(preds: pd.DataFrame, best_model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best = preds[preds["model"] == best_model].copy()
    store_rows = []
    for (store_id, store_name), group in best.groupby(["store_id", "store_name"], dropna=False):
        m = metric_dict(group["actual"], group["prediction"])
        store_rows.append({"store_id": store_id, "store_name": store_name, **m, "WAPE_pct": m["WAPE"] * 100, "n": len(group)})
    product_rows = []
    for (product_id, product_name, category), group in best.groupby(["product_id", "product_name", "category"], dropna=False):
        m = metric_dict(group["actual"], group["prediction"])
        product_rows.append({"product_id": product_id, "product_name": product_name, "category": category, **m, "WAPE_pct": m["WAPE"] * 100, "n": len(group)})
    category_rows = []
    cat_agg = aggregate_for_level(best, "category")
    for category, group in cat_agg.groupby("category", dropna=False):
        m = metric_dict(group["actual"], group["prediction"])
        category_rows.append({"category": category, **m, "WAPE_pct": m["WAPE"] * 100, "n": len(group)})
    return (
        pd.DataFrame(store_rows).sort_values("WAPE", ascending=True),
        pd.DataFrame(product_rows).sort_values("WAPE", ascending=True),
        pd.DataFrame(category_rows).sort_values("WAPE", ascending=True),
    )


def daily_error_series(preds: pd.DataFrame, model: str, level: str = "store_product") -> pd.Series:
    agg = aggregate_for_level(preds[preds["model"] == model], level)
    daily = (
        agg.assign(abs_error=lambda x: (x["actual"] - x["prediction"]).abs())
        .groupby("date", as_index=True)["abs_error"]
        .sum()
        .sort_index()
    )
    return daily


def paired_tests(base_errors: pd.Series, cand_errors: pd.Series, label: str) -> dict:
    base, cand = base_errors.align(cand_errors, join="inner")
    diff = base - cand
    row = {
        "comparison": label,
        "n_pairs": int(len(diff)),
        "mean_baseline_error": float(base.mean()) if len(base) else np.nan,
        "mean_candidate_error": float(cand.mean()) if len(cand) else np.nan,
        "mean_error_reduction": float(diff.mean()) if len(diff) else np.nan,
        "t_stat": np.nan,
        "t_p_value_less": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p_value_less": np.nan,
        "dm_stat": np.nan,
        "dm_p_value_less": np.nan,
        "note": "探索性检验；模型选择与检验使用同一验证期，结论需谨慎。",
    }
    if len(diff) < 8:
        row["note"] += " 配对天数不足，不建议写显著改进。"
        return row
    try:
        t_res = stats.ttest_rel(base, cand, alternative="greater")
        row["t_stat"] = float(t_res.statistic)
        row["t_p_value_less"] = float(t_res.pvalue)
    except Exception as exc:
        row["note"] += f" 配对t检验失败: {exc}"
    try:
        if np.allclose(diff, 0):
            row["note"] += " Wilcoxon差值全为0，未计算。"
        else:
            w_res = stats.wilcoxon(base, cand, alternative="greater", zero_method="wilcox")
            row["wilcoxon_stat"] = float(w_res.statistic)
            row["wilcoxon_p_value_less"] = float(w_res.pvalue)
    except Exception as exc:
        row["note"] += f" Wilcoxon检验失败: {exc}"
    denom = diff.std(ddof=1) / np.sqrt(len(diff))
    if denom > 0:
        dm_stat = diff.mean() / denom
        row["dm_stat"] = float(dm_stat)
        row["dm_p_value_less"] = float(1.0 - stats.t.cdf(dm_stat, df=len(diff) - 1))
    else:
        row["note"] += " DM差值方差为0，未计算。"
    return row


def not_applicable_test_row(label: str, note: str) -> dict:
    return {
        "comparison": label,
        "n_pairs": 0,
        "mean_baseline_error": np.nan,
        "mean_candidate_error": np.nan,
        "mean_error_reduction": np.nan,
        "t_stat": np.nan,
        "t_p_value_less": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p_value_less": np.nan,
        "dm_stat": np.nan,
        "dm_p_value_less": np.nan,
        "note": note,
    }


def build_future_external_scenario(history: pd.DataFrame) -> pd.DataFrame:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month_day"] = data["date"].dt.strftime("%m-%d")
    rows = []
    for date in FUTURE_DATES:
        same = data[(data["month_day"] == date.strftime("%m-%d")) & (data["has_external_data"] == 1)]
        if same.empty:
            same = data[data["has_external_data"] == 1]
        weather = same["weather"].dropna().mode().iloc[0] if not same["weather"].dropna().empty else "未知"
        rows.append(
            {
                "date": pd.Timestamp(date),
                "weather": weather,
                "max_temperature": float(same["max_temperature"].mean()),
                "min_temperature": float(same["min_temperature"].mean()),
                "wind_power": float(same["wind_power"].mean()),
                "is_activity_day": int(round(float(same["is_activity_day"].mean()))),
                "is_holiday": int(date in pd.to_datetime(["2022-04-03", "2022-04-04", "2022-04-05"])),
                "external_scenario_note": "天气、温度、风力、活动日为历史同月日参考；节假日按2022清明日历设定",
            }
        )
    return pd.DataFrame(rows)


def recursive_future_candidate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    combos: pd.DataFrame,
    raw: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    future_external = build_future_external_scenario(raw)
    future_feature = future_external.copy()
    future_feature["has_external_data"] = 1
    external_lookup = future_feature.set_index("date").to_dict("index")

    # Reuse all rows available before future. External-feature models cannot use
    # 2022-03-31 for fitting because external variables are missing, but the
    # observed 2022-03-31 sales are still available as lag history for 2022-04-01.
    spec = SPEC_BY_MODEL[model_name]
    train = feature_df[feature_df["date"] <= pd.Timestamp("2022-03-31")].copy()
    if spec.uses_future_weather_scenario:
        train = train[train["has_external_data"] == 1].copy()

    q1_alpha = {}
    if model_name in {"exp_smoothing", "ensemble_q1_ridge_context_50", "hybrid_scale_q1_ridge_context"}:
        for row in combos.itertuples():
            vals = panel[(panel["store_id"] == row.store_id) & (panel["product_id"] == row.product_id)][TARGET].astype(float).tolist()
            q1_alpha[(row.store_id, row.product_id)] = optimize_alpha(vals)

    ridge_model = None
    numeric = categorical = None
    if model_name in {"ridge_context", "ensemble_q1_ridge_context_50", "hybrid_scale_q1_ridge_context"}:
        numeric, categorical = feature_columns("context_external")
        ridge_model = make_estimator("ridge_context", numeric, categorical)
        ridge_model.fit(train[numeric + categorical], train[TARGET])

    if model_name not in {
        "exp_smoothing",
        "ridge_context",
        "ensemble_q1_ridge_context_50",
        "hybrid_scale_q1_ridge_context",
    }:
        if spec.feature_set == "history_only":
            pass
        else:
            numeric, categorical = feature_columns(spec.feature_set)
            ridge_model = make_estimator(model_name, numeric, categorical)
            ridge_model.fit(train[numeric + categorical], train[TARGET])

    history, history_dates = init_history(panel, combos, pd.Timestamp("2022-04-01"))
    rows = []
    for date in FUTURE_DATES:
        batch = build_batch_features(combos, history, date, external_lookup)
        if model_name == "exp_smoothing":
            preds = []
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                preds.append(simple_prediction("exp_smoothing", history[key], history_dates, date, q1_alpha[key]))
            preds = np.asarray(preds, dtype=float)
        elif model_name == "ensemble_q1_ridge_context_50":
            ridge_preds = np.maximum(0.0, np.asarray(ridge_model.predict(batch[numeric + categorical]), dtype=float))
            q1_preds = []
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                q1_preds.append(simple_prediction("exp_smoothing", history[key], history_dates, date, q1_alpha[key]))
            preds = 0.5 * np.asarray(q1_preds, dtype=float) + 0.5 * ridge_preds
        elif model_name == "hybrid_scale_q1_ridge_context":
            ridge_preds = np.maximum(0.0, np.asarray(ridge_model.predict(batch[numeric + categorical]), dtype=float))
            q1_preds = []
            means = []
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                q1_preds.append(simple_prediction("exp_smoothing", history[key], history_dates, date, q1_alpha[key]))
                means.append(float(np.mean(history[key])) if history[key] else 0.0)
            threshold = float(np.median(means))
            preds = np.where(np.asarray(means) >= threshold, ridge_preds, np.asarray(q1_preds, dtype=float))
        elif spec.feature_set == "history_only":
            preds = []
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                preds.append(simple_prediction(model_name, history[key], history_dates, date))
            preds = np.asarray(preds, dtype=float)
        else:
            preds = np.maximum(0.0, np.asarray(ridge_model.predict(batch[numeric + categorical]), dtype=float))

        spec = SPEC_BY_MODEL[model_name]
        for row, pred in zip(batch.itertuples(), preds):
            ext = external_lookup[pd.Timestamp(date)]
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "store_id": row.store_id,
                    "store_name": row.store_name,
                    "product_id": row.product_id,
                    "product_name": row.product_name,
                    "category": row.category,
                    "weather": ext.get("weather", "未知"),
                    "is_holiday": int(ext.get("is_holiday", 0)),
                    "is_weekend": int(pd.Timestamp(date).isocalendar().weekday in [6, 7]),
                    "is_activity_day": int(ext.get("is_activity_day", 0)),
                    "predicted_sales": max(0.0, float(pred)),
                    "model": model_name,
                    "model_label": spec.model_label,
                    "external_scenario_note": ext.get("external_scenario_note", ""),
                }
            )
        for row, pred in zip(batch.itertuples(), preds):
            history[(row.store_id, row.product_id)].append(max(0.0, float(pred)))
        history_dates.append(pd.Timestamp(date))
    return pd.DataFrame(rows)


def plot_leaderboard(leaderboard: pd.DataFrame) -> None:
    plot_df = leaderboard.sort_values("WAPE_pct", ascending=True).head(14).sort_values("WAPE_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plot_df["family"].map(
        {
            "强baseline": "#4C78A8",
            "可解释回归": "#59A14F",
            "树模型": "#F28E2B",
            "分组模型": "#B07AA1",
            "集成模型": "#E15759",
            "分层集成": "#E15759",
        }
    ).fillna("#777777")
    ax.barh(plot_df["model_label"], plot_df["WAPE_pct"], color=colors)
    ax.set_xlabel("严格 7 日递推 WAPE (%)")
    for idx, row in enumerate(plot_df.itertuples()):
        ax.text(row.WAPE_pct + 0.2, idx, f"{row.WAPE_pct:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q4_model_leaderboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_notebook(summary: dict) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Q4 综合模型方法优化探索\n",
                "\n",
                "本 Notebook 对应 `src/experimental/q4_method_search.py` 的可复现实验入口。验证口径为最后 4 个完整 7 日窗口的严格递推预测。窗口内第 2 至第 7 天只使用前面日期的预测值更新 lag/rolling 特征，不使用窗口内真实销量。"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "ROOT = Path.cwd().parents[1] if Path.cwd().name == 'method_search' else Path.cwd()\n",
                "leaderboard = pd.read_csv(ROOT / 'tables/method_search/q4_model_leaderboard.csv')\n",
                "leaderboard[['rank','model_label','family','MAE','RMSE','WAPE_pct','complexity','interpretability','paper_writability']].head(15)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 分层误差\n",
                "\n",
                "下面查看同一候选模型在门店、商品、类别聚合层面的误差，避免只看门店-商品细粒度。"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "level_metrics = pd.read_csv(ROOT / 'tables/method_search/q4_level_metrics.csv')\n",
                "level_metrics[level_metrics['model'].isin(leaderboard.head(5)['model'])][['level','model_label','MAE','RMSE','WAPE_pct']]\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 显著性检验\n",
                "\n",
                "显著性检验使用每日绝对误差总和配对比较。由于模型选择也使用了同一验证期，该结果只作为探索性证据。"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tests = pd.read_csv(ROOT / 'tables/method_search/q4_significance_test.csv')\n",
                "tests\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## 当前实验摘要\n\n最佳模型：`{summary.get('best_model_label')}`；是否建议替换原综合模型：`{summary.get('recommend_replace_original_model')}`。"
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / "q4_method_search.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def current_model_review() -> str:
    stage5 = pd.read_csv(ROOT / "tables" / "q4_store_product_model_metrics.csv")
    recursive = pd.read_csv(ROOT / "tables" / "q4_recursive_7day_metrics.csv")
    comparison = pd.read_csv(ROOT / "tables" / "q4_comparison_with_previous_models.csv")
    feature_dict = pd.read_csv(ROOT / "tables" / "q4_feature_dictionary.csv")
    recursive_sp = recursive[recursive["level"] == "store_product"].sort_values("WAPE")
    return f"""# 问题四目标与 baseline 复核

## 1. 现有综合模型摘要

原阶段 5 综合模型以 `store_id + product_id + date` 为预测粒度，目标变量为 `positive_sales`。主要特征包括：

{df_to_md(feature_dict, 20)}

原报告候选模型包括 7 日移动平均、14 日滚动均值、问题一门店-商品指数平滑、Ridge 回归和随机森林。日滚动一步验证中，Ridge 回归 WAPE 最低：

{df_to_md(stage5[['model_label', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'n']], 10)}

后来追加的严格 7 日递推验证显示，窗口内不使用真实销量更新 lag/rolling 后，门店-商品层级排序发生变化：

{df_to_md(recursive_sp[['model_label', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'n']], 10)}

## 2. 预测目标粒度判断

题目文字是“预测各门店各种零食未来 7 天的总销量”。最核心提交粒度应为 `门店-商品` 的未来 7 天总销量。由于题目还要求结合前三问并与问题一、二比较，报告中还应提供门店汇总、商品汇总、类别汇总作为解释和一致性检查，但不应替代门店-商品主表。

## 3. 当前方案是否满足输出要求

当前 `outputs/final_7day_forecast.csv` 是门店-商品-日期预测表，`tables/q4_store_product_forecast_7day_total.csv` 是门店-商品 7 天汇总表，能够覆盖核心提交粒度。同时已有门店、商品、类别汇总表，便于与前三问对照。

## 4. 是否真正结合前三问

当前方案在特征层面有结合：`lag` 与 `rolling` 对应问题一的时间规律；`category` 对应问题二的类别聚合信息；天气、温度、节假日、活动日对应问题三的外部因素统计关联。但原 Q4 对商品关联的使用较弱，主要只用了附件类别字段，没有引入门店-类别历史销量、商品全店历史销量等上下文特征。

## 5. 与问题一、二比较是否公平

原日滚动比较把问题一、二结果截取到同一验证日期，指标口径基本一致：

{df_to_md(comparison, 10)}

但最终任务是未来 7 天一次性预测，因此更公平的主比较应采用严格 7 日递推验证。日滚动一步验证可以作为辅助结果，不能直接作为“未来 7 天显著改进”的主要证据。
"""


def leakage_audit_report() -> str:
    return """# 问题四信息泄露专项审查

## 总结

本次方法优化采用严格 7 日递推验证作为主比较口径。该口径下，每个验证窗口只使用 `date < window_start` 的真实销量初始化历史序列；窗口内第 2 至第 7 天的 `lag_1`、`rolling_mean_7`、`rolling_mean_14` 均由前面日期的预测值递推得到，不再使用窗口内真实销量。

## 逐项检查

| 检查项 | 审查结果 | 处理方式 |
|---|---|---|
| 滞后特征是否只使用过去销量 | 原公式 `shift` 不含当天销量，但原日滚动验证会在窗口内使用前几天真实销量 | 本次统一改为严格递推，窗口内使用预测值更新历史 |
| rolling 是否先 shift 再 rolling | 现有 `src/features.py` 已先 `shift(1)` 再 rolling | 本次递推验证中按预测历史重新计算 rolling |
| 目标编码是否泄露验证集 | 当前方案未使用目标编码 | 不新增目标编码 |
| 标准化是否只在训练集拟合 | sklearn Pipeline 在每个窗口的训练集上 fit | 保持 Pipeline 写法 |
| 类别聚合是否使用未来数据 | 类别为附件静态字段，预测后聚合只用于评价 | 不根据未来销量生成类别 |
| 未来 7 天天气是否真实已知 | 附件没有未来真实天气 | 含天气模型只作为情景模型；另设不含天气/活动模型 |
| 促销活动未来信息是否真实已知 | 附件没有未来真实活动安排 | 活动日为历史同期情景；不含活动模型作为稳健对照 |
| 训练集和验证集是否按时间切分 | 每个窗口训练集均为窗口开始日前数据 | 不使用随机切分 |
| 是否错误使用随机切分 | 未使用随机切分 | 保持时间切分 |
| 模型比较是否使用同一验证窗口 | 所有候选模型均使用 2022-03-01 至 2022-03-28 的 4 个完整 7 日窗口 | 指标可直接横向比较 |

## 仍需谨慎

含天气和活动日的模型在验证期使用真实外部变量，在未来预测时只能使用情景假设。因此如果不想承担未来外部变量假设，应优先考虑不含天气/活动的稳健模型，或在论文中明确“给定外部变量情景下”的预测条件。
"""


def build_reports(
    leaderboard: pd.DataFrame,
    level_df: pd.DataFrame,
    significance: pd.DataFrame,
    best_model: str,
    current_model: str,
    generated_forecast: bool,
) -> str:
    best = leaderboard.iloc[0]
    current = leaderboard[leaderboard["model"] == current_model].iloc[0]
    q1 = leaderboard[leaderboard["model"] == "exp_smoothing"].iloc[0]
    improvement_vs_current = current["WAPE_pct"] - best["WAPE_pct"]
    improvement_vs_q1 = q1["WAPE_pct"] - best["WAPE_pct"]
    best_spec = SPEC_BY_MODEL[str(best["model"])]
    replace = bool(
        best_spec.family != "强baseline"
        and improvement_vs_current >= 1.0
        and best["paper_writability"] in {"高", "较高"}
    )
    forecast_text = (
        "已生成 `outputs/method_search/q4_final_7day_forecast_candidate.csv` 作为对照候选，但仍需人工检查格式、负值和四舍五入规则；若最佳模型只是问题一 baseline，则不应直接作为问题四综合模型替换。"
        if generated_forecast
        else "未生成新的最终预测候选表，因为本次没有发现相对原综合模型达到明显、稳健改进且适合直接替换的方案。"
    )
    significance_brief = significance[
        ["comparison", "n_pairs", "mean_error_reduction", "t_p_value_less", "wilcoxon_p_value_less", "dm_p_value_less", "note"]
    ]
    top_table = leaderboard[
        [
            "rank",
            "model_label",
            "family",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "complexity",
            "interpretability",
            "paper_writability",
            "uses_future_weather_scenario",
        ]
    ].head(12)
    level_top = level_df[level_df["model"].isin(leaderboard.head(5)["model"])][
        ["level", "model_label", "MAE", "RMSE", "WAPE_pct"]
    ]
    replacement_text = "建议替换原综合模型" if replace else "不建议直接替换原综合模型"
    final_replacement_text = "建议先人工复核候选预测表后再考虑替换" if generated_forecast and replace else "不建议替换原最终预测文件"

    return f"""# 问题四综合模型方法优化探索报告

## 1. 当前综合模型的不足

原 Q4 Ridge 在日滚动一步验证中表现较好，但严格 7 日递推验证显示其门店-商品 WAPE 上升，说明原验证口径会受窗口内真实销量更新 lag/rolling 特征影响。原方案还主要把问题二信息简化为 `category` 静态字段，对门店-类别历史销量、门店总量、商品全店总量等上下文利用不足。

## 2. 信息泄露检查结果

未发现当天目标直接进入特征、随机切分、全量标准化等硬泄露。本次已把主验证统一为严格 7 日递推窗口，修正了原日滚动验证不能代表未来 7 天一次性预测的问题。未来天气和活动日仍不是附件真实观测，只能作为情景变量。

## 3. 候选模型对比

{df_to_md(top_table, 20)}

分层误差中，前 5 个候选模型在不同聚合层级表现如下：

{df_to_md(level_top, 40)}

## 4. 最优模型选择理由

本次严格递推验证的最低 WAPE 模型为 `{best['model_label']}`，门店-商品 WAPE 为 {best['WAPE_pct']:.2f}%。相对原 Q4 Ridge 递推口径的 WAPE 变化为 {improvement_vs_current:.2f} 个百分点；相对问题一门店-商品指数平滑的变化为 {improvement_vs_q1:.2f} 个百分点。

选择模型时不只看 WAPE：若最佳模型只是微弱领先，或需要较复杂集成而论文表达收益有限，应优先保留更稳定、可解释的方案。树模型虽然能表达非线性，但本次递推误差和论文可写性均不占优。

## 5. 是否建议替换原综合模型

结论：**{replacement_text}**。

判断依据是：替换不仅要误差更低，还要稳定、可解释、适合论文，并且不能依赖不可获得未来信息。本次若最佳模型相对原 Ridge 改进不足 1 个 WAPE 百分点，或显著性证据不足，则更适合把它作为稳健性/消融比较，而不是直接替换主方案。

## 6. 是否建议替换最终预测结果

结论：**{final_replacement_text}**。{forecast_text}

无论是否替换，替换前都应人工检查：预测表是否仍为门店-商品-日期粒度；7 天汇总是否完整；是否存在负值；是否按竞赛提交要求保留小数或四舍五入；未来天气/活动日情景是否需要在正文说明。

## 7. 与问题一、问题二模型比较是否公平

本次比较使用同一组严格 7 日递推窗口、同一门店-商品预测粒度、同一目标变量 `positive_sales` 和同一 MAE/RMSE/WAPE 口径。与问题二的类别比较通过预测后聚合到类别层级完成，避免把细粒度模型和类别模型放在不同验证日期上比较。

## 8. 误差改进是否稳定或显著

{df_to_md(significance_brief, 10)}

显著性检验以每日绝对误差总和为配对序列。由于候选模型较多，且最优模型是在同一验证期中选出的，检验只能作为探索性证据。若 p 值未稳定低于 0.05，不应写“显著改进”。

## 9. 论文中应该如何表述模型改进

推荐表述：“在严格 7 日递推验证下，综合模型与强 baseline 的误差比较如表所示。部分综合特征模型在聚合层级上有数值改进，但门店-商品层级相对问题一指数平滑的优势有限，因此本文将综合模型定位为融合解释框架，并保留简单模型作为强基准。”

避免表述：“综合模型显著提高了未来 7 天预测精度。”除非最终选定模型同时在严格递推验证、显著性检验和论文可解释性上都具备充分证据。

## 10. 若不建议使用最复杂模型，原因

RandomForest、ExtraTrees、LightGBM、XGBoost、CatBoost 等树模型或增强模型并不天然更适合本题。当前数据只有 7 家门店、约 12 个商品编码和较短未来 7 天预测任务，复杂模型容易在日滚动验证中看似有效，但递推后误差可能放大；同时论文中解释变量作用、写出数学表达和说明稳定性都会更困难。

## 11. 进入论文写作前需要人工确认

1. 最终提交是否要求每日预测表，还是只要求 7 天总销量表。
2. `positive_sales` 是否继续作为主目标，负销量是否只作为损耗/冲销审计字段。
3. 未来天气和活动日是否允许使用历史同期情景；若不允许，应采用不含天气/活动的模型。
4. 最终表是否需要整数销量；若需要，四舍五入后要重新检查汇总一致性。
5. 问题四正文应以严格 7 日递推验证为主，日滚动一步验证只能作为辅助结果。
"""


def main() -> None:
    start = time.perf_counter()
    for path in [TABLE_DIR, FIGURE_DIR, OUTPUT_DIR, NOTEBOOK_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "processed" / "modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_panel(raw)
    feature_df = prepare_feature_table(panel)
    combos = (
        panel[["store_id", "store_name", "product_id", "product_name", "category"]]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )
    actual_lookup = actual_lookup_from_panel(panel)

    predictions = {}
    for model_name in ["same_weekday_mean_8", "moving_average_7", "moving_average_14", "exp_smoothing"]:
        predictions[model_name] = recursive_simple_validate(panel, combos, actual_lookup, model_name)

    for model_name in [
        "linear_basic",
        "ridge_basic",
        "ridge_no_weather_activity",
        "ridge_context",
        "lasso_context",
        "random_forest_context",
        "extra_trees_context",
    ]:
        predictions[model_name] = recursive_ml_validate(panel, feature_df, combos, actual_lookup, model_name)

    predictions["ridge_by_store_context"] = recursive_grouped_ridge_validate(
        panel, feature_df, combos, actual_lookup, "ridge_by_store_context", "store_id"
    )
    predictions["ridge_by_category_context"] = recursive_grouped_ridge_validate(
        panel, feature_df, combos, actual_lookup, "ridge_by_category_context", "category"
    )
    predictions["ensemble_q1_ridge_context_50"] = combine_predictions(
        predictions["exp_smoothing"],
        predictions["ridge_context"],
        "ensemble_q1_ridge_context_50",
        lambda left, right, _: 0.5 * left + 0.5 * right,
    )
    predictions["hybrid_scale_q1_ridge_context"] = make_hybrid_predictions(
        panel,
        predictions["exp_smoothing"],
        predictions["ridge_context"],
        "hybrid_scale_q1_ridge_context",
    )

    all_preds = pd.concat(predictions.values(), ignore_index=True)
    all_preds["abs_error"] = (all_preds["actual"] - all_preds["prediction"]).abs()
    save_csv(all_preds, TABLE_DIR / "q4_validation_predictions.csv")

    level_df = level_metrics(all_preds)
    save_csv(level_df, TABLE_DIR / "q4_level_metrics.csv")

    sp = level_df[level_df["level"] == "store_product"].copy().sort_values(["WAPE", "MAE", "RMSE"]).reset_index(drop=True)
    sp.insert(0, "rank", np.arange(1, len(sp) + 1))
    leaderboard = sp[
        [
            "rank",
            "model",
            "model_label",
            "family",
            "MAE",
            "RMSE",
            "WAPE",
            "WAPE_pct",
            "actual_sum",
            "prediction_sum",
            "n",
            "complexity",
            "interpretability",
            "paper_writability",
            "uses_future_weather_scenario",
            "notes",
        ]
    ].copy()
    save_csv(leaderboard, TABLE_DIR / "q4_model_leaderboard.csv")
    plot_leaderboard(leaderboard)

    best_model = str(leaderboard.iloc[0]["model"])
    current_model = "ridge_basic"
    store_errors, product_errors, category_errors = group_error_tables(all_preds, best_model)
    save_csv(store_errors, TABLE_DIR / "q4_error_by_store.csv")
    save_csv(product_errors, TABLE_DIR / "q4_error_by_product.csv")
    save_csv(category_errors, TABLE_DIR / "q4_error_by_category.csv")

    env_rows = []
    for pkg, model_name in [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
    ]:
        try:
            __import__(pkg)
            available = True
        except Exception:
            available = False
        env_rows.append(
            {
                "model_family": model_name,
                "package": pkg,
                "available": available,
                "action": "本次不强行安装；若环境已具备可纳入同一验证框架",
            }
        )
    save_csv(pd.DataFrame(env_rows), TABLE_DIR / "q4_tree_model_environment.csv")

    baseline_candidates = ["exp_smoothing", "moving_average_7", "moving_average_14", "same_weekday_mean_8"]
    best_baseline = (
        leaderboard[leaderboard["model"].isin(baseline_candidates)]
        .sort_values("WAPE")
        .iloc[0]["model"]
    )
    tests = []
    for base_model, label in [
        (current_model, "最佳候选 vs 原Q4 Ridge严格递推口径"),
        ("exp_smoothing", "最佳候选 vs 问题一门店-商品指数平滑"),
        (str(best_baseline), "最佳候选 vs 最优强baseline"),
    ]:
        if base_model == best_model:
            tests.append(
                not_applicable_test_row(
                    label,
                    "最佳候选与基准为同一模型，误差序列完全相同，不做配对检验。",
                )
            )
            continue
        tests.append(
            paired_tests(
                daily_error_series(all_preds, base_model),
                daily_error_series(all_preds, best_model),
                label,
            )
        )
    tests.append(
        paired_tests(
            daily_error_series(all_preds, "exp_smoothing"),
            daily_error_series(all_preds, current_model),
            "原Q4 Ridge严格递推口径 vs 问题一门店-商品指数平滑",
        )
    )
    tests.append(
        paired_tests(
            daily_error_series(all_preds, "exp_smoothing"),
            daily_error_series(all_preds, "ensemble_q1_ridge_context_50"),
            "等权集成 vs 问题一门店-商品指数平滑",
        )
    )
    tests.append(
        not_applicable_test_row(
            "综合模型聚合到类别 vs 问题二类别模型",
            "问题二现有验证表是日滚动一步预测，当前主验证是严格7日递推；两者递推口径不同，不强行做显著性检验。可在论文中只做同日期数值对照或补做问题二严格递推。",
        )
    )
    tests.append(
        not_applicable_test_row(
            "综合模型聚合到门店类别 vs 问题二门店类别模型",
            "问题二现有门店类别结果不是本次统一递推框架生成，若直接检验会混合验证口径；因此本次不写显著改进。",
        )
    )
    significance = pd.DataFrame(tests)
    save_csv(significance, TABLE_DIR / "q4_significance_test.csv")

    current = leaderboard[leaderboard["model"] == current_model].iloc[0]
    best = leaderboard.iloc[0]
    recommend_replace = bool(
        SPEC_BY_MODEL[str(best["model"])].family != "强baseline"
        and (float(current["WAPE_pct"]) - float(best["WAPE_pct"])) >= 1.0
        and str(best["paper_writability"]) in {"高", "较高"}
    )
    generated_forecast = False
    if (
        best_model != current_model
        and (float(current["WAPE_pct"]) - float(best["WAPE_pct"])) > 0.0
    ):
        forecast = recursive_future_candidate(panel, feature_df, combos, raw, best_model)
        save_csv(forecast, OUTPUT_DIR / "q4_final_7day_forecast_candidate.csv")
        total = (
            forecast.groupby(["store_id", "store_name", "product_id", "product_name", "category", "model", "model_label"], as_index=False)[
                "predicted_sales"
            ]
            .sum()
            .rename(columns={"predicted_sales": "predicted_7day_sales"})
            .sort_values(["store_id", "predicted_7day_sales"], ascending=[True, False])
        )
        save_csv(total, TABLE_DIR / "q4_final_7day_forecast_candidate_total.csv")
        generated_forecast = True

    save_csv(
        pd.DataFrame(
            [
                {
                    "validation_windows": "; ".join(
                        f"{s.date()} to {e.date()}" for s, e in VALIDATION_WINDOWS
                    ),
                    "target": TARGET,
                    "grain": "store_id + product_id + date",
                    "n_store_product_combinations": int(combos[["store_id", "product_id"]].drop_duplicates().shape[0]),
                    "n_validation_rows_per_model": int(len(all_preds[all_preds["model"] == best_model])),
                    "best_model": best_model,
                    "best_model_label": str(best["model_label"]),
                    "best_wape_pct": float(best["WAPE_pct"]),
                    "current_ridge_wape_pct": float(current["WAPE_pct"]),
                    "forecast_candidate_generated": generated_forecast,
                    "recommend_replace_original_model": recommend_replace,
                }
            ]
        ),
        TABLE_DIR / "q4_method_search_summary.csv",
    )

    (OUTPUT_DIR / "q4_target_and_baseline_review.md").write_text(current_model_review(), encoding="utf-8")
    (OUTPUT_DIR / "q4_leakage_audit.md").write_text(leakage_audit_report(), encoding="utf-8")
    final_report = build_reports(
        leaderboard,
        level_df,
        significance,
        best_model,
        current_model,
        generated_forecast,
    )
    (OUTPUT_DIR / "q4_method_search_report.md").write_text(final_report, encoding="utf-8")
    summary = {
        "best_model": best_model,
        "best_model_label": str(best["model_label"]),
        "best_wape_pct": float(best["WAPE_pct"]),
        "current_ridge_wape_pct": float(current["WAPE_pct"]),
        "recommend_replace_original_model": recommend_replace,
        "forecast_candidate_generated": generated_forecast,
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    (OUTPUT_DIR / "q4_method_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_notebook(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
