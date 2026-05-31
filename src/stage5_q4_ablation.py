from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics
from src.features import add_leakage_safe_sales_features, build_store_product_panel
from src.models import (
    build_ridge_forecaster,
    clipped_predict,
    exp_smoothing_prediction,
    optimize_exp_smoothing_alpha,
)

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

TARGET = "positive_sales"
WINDOW_LENGTH = 7
VALIDATION_STARTS = pd.to_datetime(
    ["2022-03-01", "2022-03-08", "2022-03-15", "2022-03-22"]
)

HISTORY_NUMERIC = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]
CALENDAR_NUMERIC = ["weekday", "month", "is_weekend"]
EVENT_NUMERIC = ["is_holiday", "is_activity_day"]
WEATHER_NUMERIC = ["max_temperature", "min_temperature", "wind_power"]
ID_CATEGORICAL = ["store_id_str", "product_id_str"]
CATEGORY_CATEGORICAL = ["category"]
WEATHER_CATEGORICAL = ["weather"]

ALL_NUMERIC = HISTORY_NUMERIC + CALENDAR_NUMERIC + EVENT_NUMERIC + WEATHER_NUMERIC
ALL_CATEGORICAL = ID_CATEGORICAL + CATEGORY_CATEGORICAL + WEATHER_CATEGORICAL


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    model_label: str
    feature_scope: str
    numeric_features: list[str]
    categorical_features: list[str]


SPECS = [
    AblationSpec(
        variant="ridge_full_external",
        model_label="完整综合Ridge",
        feature_scope="历史滞后 + 日历 + 节假日/活动日 + 天气 + 门店/商品/类别",
        numeric_features=ALL_NUMERIC,
        categorical_features=ALL_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_no_weather",
        model_label="去天气Ridge",
        feature_scope="去除天气、温度、风力，保留历史、日历、节假日/活动日和门店/商品/类别",
        numeric_features=HISTORY_NUMERIC + CALENDAR_NUMERIC + EVENT_NUMERIC,
        categorical_features=ID_CATEGORICAL + CATEGORY_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_no_activity",
        model_label="去活动日Ridge",
        feature_scope="去除活动日，保留历史、日历、节假日、天气和门店/商品/类别",
        numeric_features=HISTORY_NUMERIC
        + CALENDAR_NUMERIC
        + ["is_holiday"]
        + WEATHER_NUMERIC,
        categorical_features=ALL_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_no_event",
        model_label="去节假日活动Ridge",
        feature_scope="去除节假日和活动日，保留历史、日历、天气和门店/商品/类别",
        numeric_features=HISTORY_NUMERIC + CALENDAR_NUMERIC + WEATHER_NUMERIC,
        categorical_features=ALL_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_history_id_category",
        model_label="仅历史与层级Ridge",
        feature_scope="仅使用历史滞后/滚动特征和门店/商品/类别",
        numeric_features=HISTORY_NUMERIC,
        categorical_features=ID_CATEGORICAL + CATEGORY_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_history_calendar",
        model_label="历史日历Ridge",
        feature_scope="历史滞后/滚动 + 星期/月/周末 + 门店/商品/类别",
        numeric_features=HISTORY_NUMERIC + CALENDAR_NUMERIC,
        categorical_features=ID_CATEGORICAL + CATEGORY_CATEGORICAL,
    ),
    AblationSpec(
        variant="ridge_no_history",
        model_label="无销量历史Ridge",
        feature_scope="不使用 lag/rolling，仅使用日历、外部变量和门店/商品/类别",
        numeric_features=CALENDAR_NUMERIC + EVENT_NUMERIC + WEATHER_NUMERIC,
        categorical_features=ALL_CATEGORICAL,
    ),
]


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


def prepend_result_log(row: str) -> None:
    path = ROOT / "RESULT_LOG.md"
    text = path.read_text(encoding="utf-8")
    if row in text:
        return
    lines = text.splitlines()
    insert_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("|------"):
            insert_idx = idx + 1
            break
    if insert_idx is not None:
        lines.insert(insert_idx, row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUTS / name, index=False, encoding="utf-8-sig")


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_leakage_safe_sales_features(panel, TARGET)
    out["store_id_str"] = out["store_id"].astype(str)
    out["product_id_str"] = out["product_id"].astype(str)
    out["category"] = out["category"].astype(str)
    out["weather"] = out["weather"].fillna("未知").astype(str)
    for col in ALL_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["has_external_data"] = out["has_external_data"].fillna(0).astype(int)
    return out.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def validation_windows(feature_df: pd.DataFrame) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[dict]]:
    external_dates = set(
        pd.to_datetime(
            feature_df.loc[feature_df["has_external_data"] == 1, "date"]
        ).dt.normalize()
    )
    data_dates = set(pd.to_datetime(feature_df["date"]).dt.normalize())
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    excluded: list[dict] = []
    for start in VALIDATION_STARTS:
        end = start + pd.Timedelta(days=WINDOW_LENGTH - 1)
        dates = pd.date_range(start, end, freq="D")
        has_all_sales = all(d in data_dates for d in dates)
        has_all_external = all(d in external_dates for d in dates)
        if has_all_sales and has_all_external:
            windows.append((start, end))
        else:
            excluded.append(
                {
                    "window_start": start.date().isoformat(),
                    "window_end": end.date().isoformat(),
                    "reason": "不足完整 7 日销售或外部变量覆盖，未纳入消融检验",
                }
            )
    return windows, excluded


def history_feature_row(values: list[float]) -> dict[str, float]:
    values = [float(v) for v in values if pd.notna(v)]
    recent7 = values[-7:] if values else [0.0]
    recent14 = values[-14:] if values else [0.0]
    return {
        "lag_1": float(values[-1]) if len(values) >= 1 else 0.0,
        "lag_7": float(values[-7]) if len(values) >= 7 else 0.0,
        "lag_14": float(values[-14]) if len(values) >= 14 else 0.0,
        "rolling_mean_7": float(np.mean(recent7)),
        "rolling_mean_14": float(np.mean(recent14)),
        "rolling_std_7": float(np.std(recent7, ddof=1)) if len(recent7) >= 2 else 0.0,
    }


def build_lookup_tables(panel: pd.DataFrame, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    combos = (
        panel[["store_id", "store_name", "product_id", "product_name", "category"]]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )
    actual_lookup = (
        panel.set_index(["date", "store_id", "product_id"])[TARGET].astype(float).to_dict()
    )
    external_cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_holiday",
        "is_activity_day",
        "weekday",
        "month",
        "is_weekend",
        "has_external_data",
    ]
    external_lookup = (
        feature_df[external_cols].drop_duplicates("date").set_index("date").to_dict("index")
    )
    return combos, actual_lookup, external_lookup


def initial_history(panel: pd.DataFrame, combos: pd.DataFrame, window_start: pd.Timestamp) -> dict:
    before = panel[panel["date"] < window_start]
    history = {}
    for row in combos.itertuples():
        key = (int(row.store_id), int(row.product_id))
        history[key] = (
            before[
                (before["store_id"] == row.store_id)
                & (before["product_id"] == row.product_id)
            ]
            .sort_values("date")[TARGET]
            .astype(float)
            .tolist()
        )
    return history


def batch_rows_for_date(
    combos: pd.DataFrame,
    history: dict,
    target_date: pd.Timestamp,
    ext: dict,
) -> pd.DataFrame:
    rows = []
    for row in combos.itertuples():
        rows.append(
            {
                "date": pd.Timestamp(target_date),
                "store_id": int(row.store_id),
                "store_name": row.store_name,
                "product_id": int(row.product_id),
                "product_name": row.product_name,
                "category": str(row.category),
                **history_feature_row(history[(int(row.store_id), int(row.product_id))]),
                "weekday": int(ext["weekday"]),
                "month": int(ext["month"]),
                "is_weekend": int(ext["is_weekend"]),
                "is_holiday": int(ext["is_holiday"]),
                "is_activity_day": int(ext["is_activity_day"]),
                "max_temperature": float(ext.get("max_temperature", 0.0)),
                "min_temperature": float(ext.get("min_temperature", 0.0)),
                "wind_power": float(ext.get("wind_power", 0.0)),
                "weather": str(ext.get("weather", "未知")),
                "store_id_str": str(row.store_id),
                "product_id_str": str(row.product_id),
            }
        )
    return pd.DataFrame(rows)


def append_prediction_rows(
    rows: list[dict],
    batch: pd.DataFrame,
    actual_lookup: dict,
    variant: str,
    model_label: str,
    feature_scope: str,
    window_start: pd.Timestamp,
) -> None:
    for pred_row in batch.itertuples():
        key = (pd.Timestamp(pred_row.date), pred_row.store_id, pred_row.product_id)
        actual = float(actual_lookup.get(key, 0.0))
        rows.append(
            {
                "validation_mode": "strict_recursive_7day",
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(days=WINDOW_LENGTH - 1),
                "date": pd.Timestamp(pred_row.date),
                "horizon": int((pd.Timestamp(pred_row.date) - window_start).days + 1),
                "store_id": pred_row.store_id,
                "store_name": pred_row.store_name,
                "product_id": pred_row.product_id,
                "product_name": pred_row.product_name,
                "category": pred_row.category,
                "actual": actual,
                "prediction": float(pred_row.prediction),
                "model": variant,
                "model_label": model_label,
                "feature_scope": feature_scope,
                "feature_source": "验证窗口内 lag/rolling 使用前序预测值递推生成",
            }
        )


def validate_exp_smoothing(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    combos, actual_lookup, external_lookup = build_lookup_tables(panel, feature_df)
    rows: list[dict] = []
    variant = "q1_store_product_exp_smoothing"
    label = "问题一门店-商品简单指数平滑"
    for window_start, window_end in windows:
        history = initial_history(panel, combos, window_start)
        alphas = {
            (int(row.store_id), int(row.product_id)): optimize_exp_smoothing_alpha(
                pd.Series(history[(int(row.store_id), int(row.product_id))], dtype=float)
            )
            for row in combos.itertuples()
        }
        for target_date in pd.date_range(window_start, window_end, freq="D"):
            batch_rows = []
            ext = external_lookup[pd.Timestamp(target_date)]
            for row in combos.itertuples():
                key = (int(row.store_id), int(row.product_id))
                pred = max(0.0, float(exp_smoothing_prediction(history[key], alphas[key])))
                batch_rows.append(
                    {
                        "date": pd.Timestamp(target_date),
                        "store_id": int(row.store_id),
                        "store_name": row.store_name,
                        "product_id": int(row.product_id),
                        "product_name": row.product_name,
                        "category": str(row.category),
                        "weather": str(ext.get("weather", "未知")),
                        "prediction": pred,
                    }
                )
            batch = pd.DataFrame(batch_rows)
            append_prediction_rows(
                rows,
                batch,
                actual_lookup,
                variant,
                label,
                "仅使用该门店-商品自身历史销量，不使用外部变量",
                window_start,
            )
            for pred_row in batch.itertuples():
                history[(int(pred_row.store_id), int(pred_row.product_id))].append(
                    float(pred_row.prediction)
                )
    return pd.DataFrame(rows)


def validate_ridge_spec(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    spec: AblationSpec,
) -> pd.DataFrame:
    combos, actual_lookup, external_lookup = build_lookup_tables(panel, feature_df)
    rows: list[dict] = []
    features = spec.numeric_features + spec.categorical_features
    for window_start, window_end in windows:
        train = feature_df[
            (feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)
        ].copy()
        model = build_ridge_forecaster(
            spec.numeric_features,
            spec.categorical_features,
            alpha=10.0,
        )
        model.fit(train[features], train[TARGET])

        history = initial_history(panel, combos, window_start)
        for target_date in pd.date_range(window_start, window_end, freq="D"):
            ext = external_lookup[pd.Timestamp(target_date)]
            batch = batch_rows_for_date(combos, history, pd.Timestamp(target_date), ext)
            for col in spec.numeric_features:
                batch[col] = pd.to_numeric(batch[col], errors="coerce").fillna(0.0)
            batch["prediction"] = clipped_predict(model, batch[features])
            append_prediction_rows(
                rows,
                batch,
                actual_lookup,
                spec.variant,
                spec.model_label,
                spec.feature_scope,
                window_start,
            )
            for pred_row in batch.itertuples():
                history[(int(pred_row.store_id), int(pred_row.product_id))].append(
                    float(pred_row.prediction)
                )
    return pd.DataFrame(rows)


def aggregate_predictions(preds: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "store_product":
        return preds.copy()
    if level == "store":
        group_cols = ["date", "window_start", "model", "model_label", "store_id", "store_name"]
    elif level == "product":
        group_cols = [
            "date",
            "window_start",
            "model",
            "model_label",
            "product_id",
            "product_name",
            "category",
        ]
    elif level == "category":
        group_cols = ["date", "window_start", "model", "model_label", "category"]
    else:
        raise ValueError(level)
    return (
        preds.groupby(group_cols, dropna=False, as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(group_cols)
    )


def metric_table(preds: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    validation_start = min(start for start, _ in windows).date().isoformat()
    validation_end = max(end for _, end in windows).date().isoformat()
    rows = []
    for level in ["store_product", "store", "product", "category"]:
        metric_input = aggregate_predictions(preds, level)
        for (model, label), group in metric_input.groupby(["model", "model_label"], dropna=False):
            metrics = calculate_metrics(group["actual"], group["prediction"])
            feature_scope = (
                preds.loc[preds["model"] == model, "feature_scope"].dropna().iloc[0]
            )
            rows.append(
                {
                    "validation_mode": "strict_recursive_7day",
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "level": level,
                    "model": model,
                    "model_label": label,
                    "feature_scope": feature_scope,
                    **metrics,
                    "WAPE_pct": metrics["WAPE"] * 100,
                    "actual_sum": float(group["actual"].sum()),
                    "prediction_sum": float(group["prediction"].sum()),
                    "bias": float(group["prediction"].sum() - group["actual"].sum()),
                    "bias_pct_of_actual": float(
                        (group["prediction"].sum() - group["actual"].sum())
                        / group["actual"].sum()
                    )
                    if group["actual"].sum() != 0
                    else np.nan,
                    "n": int(len(group)),
                }
            )
    out = pd.DataFrame(rows)
    sp = out[out["level"] == "store_product"].set_index("model")
    full_wape = float(sp.loc["ridge_full_external", "WAPE_pct"])
    q1_wape = float(sp.loc["q1_store_product_exp_smoothing", "WAPE_pct"])
    out["WAPE_pct_point_change_vs_full"] = out["WAPE_pct"] - full_wape
    out["WAPE_pct_point_change_vs_q1"] = out["WAPE_pct"] - q1_wape
    return out.sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def plot_store_product_wape(metrics: pd.DataFrame) -> None:
    sp = metrics[metrics["level"] == "store_product"].sort_values("WAPE_pct")
    label_map = {
        "q1_store_product_exp_smoothing": "Q1 exp smoothing",
        "ridge_full_external": "Full Ridge",
        "ridge_no_weather": "No weather",
        "ridge_no_activity": "No activity",
        "ridge_no_event": "No holiday/activity",
        "ridge_history_id_category": "History + hierarchy",
        "ridge_history_calendar": "History + calendar",
        "ridge_no_history": "No sales history",
    }
    labels = [label_map.get(model, model) for model in sp["model"]]
    values = sp["WAPE_pct"].tolist()
    plt.figure(figsize=(10, 5.8))
    colors = ["#2F6B55" if m == "q1_store_product_exp_smoothing" else "#6B7280" for m in sp["model"]]
    plt.barh(labels, values, color=colors)
    plt.xlabel("WAPE (%)")
    plt.title("Q4 strict recursive 7-day ablation at store-product level")
    for idx, value in enumerate(values):
        plt.text(value + 0.25, idx, f"{value:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "q4_ablation_store_product_wape.png", dpi=180)
    plt.close()


def build_report(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    excluded: list[dict],
    metrics: pd.DataFrame,
) -> str:
    window_df = pd.DataFrame(
        [
            {
                "window_start": start.date().isoformat(),
                "window_end": end.date().isoformat(),
                "length": int((end - start).days + 1),
            }
            for start, end in windows
        ]
    )
    spec_df = pd.DataFrame(
        [
            {
                "model": "q1_store_product_exp_smoothing",
                "model_label": "问题一门店-商品简单指数平滑",
                "feature_scope": "仅使用该门店-商品自身历史销量，不使用外部变量",
            }
        ]
        + [
            {
                "model": spec.variant,
                "model_label": spec.model_label,
                "feature_scope": spec.feature_scope,
            }
            for spec in SPECS
        ]
    )
    sp = metrics[metrics["level"] == "store_product"].copy()
    sp_table = sp[
        [
            "model_label",
            "feature_scope",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "WAPE_pct_point_change_vs_full",
            "WAPE_pct_point_change_vs_q1",
            "actual_sum",
            "prediction_sum",
            "bias_pct_of_actual",
            "n",
        ]
    ].copy()
    level_table = metrics[
        ["level", "model_label", "MAE", "RMSE", "WAPE_pct", "bias_pct_of_actual", "n"]
    ].copy()
    best = sp.sort_values(["WAPE", "MAE", "RMSE"]).iloc[0]
    full = sp[sp["model"] == "ridge_full_external"].iloc[0]
    no_weather = sp[sp["model"] == "ridge_no_weather"].iloc[0]
    no_activity = sp[sp["model"] == "ridge_no_activity"].iloc[0]
    history_calendar = sp[sp["model"] == "ridge_history_calendar"].iloc[0]
    no_history = sp[sp["model"] == "ridge_no_history"].iloc[0]
    q1 = sp[sp["model"] == "q1_store_product_exp_smoothing"].iloc[0]

    weather_gain = no_weather["WAPE_pct"] - full["WAPE_pct"]
    activity_gain = no_activity["WAPE_pct"] - full["WAPE_pct"]
    history_penalty = no_history["WAPE_pct"] - full["WAPE_pct"]
    calendar_gap = history_calendar["WAPE_pct"] - full["WAPE_pct"]
    ridge_vs_q1 = full["WAPE_pct"] - q1["WAPE_pct"]

    return f"""# 问题四严格 7 日递推消融实验报告

执行日期：{date.today().isoformat()}

## 1. 实验目的

本实验不继续盲目调参，而是在与问题四主验证一致的严格 7 日递推口径下，检查综合 Ridge 模型中历史销量、日历、节假日/活动日、天气、门店/商品/类别等信息分别提供了多少可复现增益。每个验证窗口开始时只使用窗口开始日前真实销量，窗口内第 2 天及以后 `lag` 和 `rolling` 均由前序预测值递推生成。

## 2. 验证窗口

{df_to_md(window_df)}

未纳入窗口：

{df_to_md(pd.DataFrame(excluded))}

## 3. 消融模型设置

{df_to_md(spec_df, 20)}

## 4. 门店-商品主粒度结果

{df_to_md(sp_table, 20)}

完整分层结果见 `tables/q4_ablation_metrics.csv`，逐日预测明细见 `tables/q4_ablation_predictions.csv`。图表已保存至 `figures/q4_ablation_store_product_wape.png`。

## 5. 关键判断

1. 严格递推门店-商品粒度下，当前最佳模型为 **{best['model_label']}**，WAPE 为 {best['WAPE_pct']:.2f}%。
2. 完整综合 Ridge 的 WAPE 为 {full['WAPE_pct']:.2f}%，相对问题一简单指数平滑高 {ridge_vs_q1:.2f} 个百分点。因此论文不能写“综合模型在严格 7 日预测中整体优于问题一模型”。
3. 去天气 Ridge 相比完整 Ridge 的 WAPE 变化为 {weather_gain:.2f} 个百分点。天气变量可写作情景补充特征，但不能写成确定因果作用。
4. 去活动日 Ridge 相比完整 Ridge 的 WAPE 变化为 {activity_gain:.2f} 个百分点。活动日变量可能包含经营安排内生性，论文中应表述为“与活动日标记相关的销量差异”。
5. 无销量历史 Ridge 相比完整 Ridge 的 WAPE 上升 {history_penalty:.2f} 个百分点，说明滞后和滚动销量仍是最核心预测信息。
6. 历史日历 Ridge 相比完整 Ridge 的 WAPE 变化为 {calendar_gap:.2f} 个百分点，说明外部变量的边际收益需要结合严格递推和未来可得性谨慎解释。

## 6. 论文取舍建议

问题四正文建议以“完整综合 Ridge + 严格递推消融”说明综合模型确实整合了前三问信息；但最终模型优劣表述应降调为：综合 Ridge 在门店、商品等聚合层面有较好解释价值，在门店-商品最细粒度上未超过问题一简单指数平滑。消融实验适合放在模型评价或附录，用来证明团队没有简单堆特征，而是做了可复现的方法取舍。
"""


def main() -> None:
    for path in [TABLES, OUTPUTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "processed" / "modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    panel["date"] = pd.to_datetime(panel["date"])
    feature_df = prepare_model_frame(panel)

    windows, excluded = validation_windows(feature_df)
    if not windows:
        raise RuntimeError("没有可用于消融实验的完整 7 日验证窗口。")

    predictions = [validate_exp_smoothing(panel, feature_df, windows)]
    predictions.extend(validate_ridge_spec(panel, feature_df, windows, spec) for spec in SPECS)
    preds = pd.concat(predictions, ignore_index=True)
    preds["abs_error"] = (preds["actual"] - preds["prediction"]).abs()
    save_csv(preds, "q4_ablation_predictions.csv")

    metrics = metric_table(preds, windows)
    save_csv(metrics, "q4_ablation_metrics.csv")
    plot_store_product_wape(metrics)

    report = build_report(windows, excluded, metrics)
    (OUTPUTS / "q4_ablation_report.md").write_text(report, encoding="utf-8")

    sp = metrics[metrics["level"] == "store_product"].set_index("model")
    best = metrics[metrics["level"] == "store_product"].sort_values("WAPE").iloc[0]
    row = (
        f"| {date.today().isoformat()} | 问题四严格 7 日递推消融实验 | "
        "processed: modeling_base_table.csv；未修改 data/raw | "
        "问题一指数平滑、完整Ridge、去天气/去活动/去事件/历史日历/无历史等 Ridge 消融 | "
        "lag/rolling、门店、商品、类别、日历、节假日、活动日、天气按消融组合控制 | "
        f"门店-商品 WAPE：最佳={best['model_label']} {best['WAPE_pct']:.2f}%；"
        f"完整Ridge={sp.loc['ridge_full_external','WAPE_pct']:.2f}%；"
        f"问题一指数平滑={sp.loc['q1_store_product_exp_smoothing','WAPE_pct']:.2f}%；"
        f"去天气={sp.loc['ridge_no_weather','WAPE_pct']:.2f}% | "
        "历史销量是核心信息；完整Ridge可作为综合解释模型，但严格递推细粒度未超过问题一指数平滑 | "
        "消融使用同一验证窗口，仍不可把天气/活动日关联写成因果；不据此强行替换最终预测文件 | "
        "将消融表和 WAPE 图写入模型评价或附录 |"
    )
    prepend_result_log(row)

    summary = {
        "validation_mode": "strict_recursive_7day",
        "windows": [
            {"window_start": str(start.date()), "window_end": str(end.date())}
            for start, end in windows
        ],
        "store_product_metrics": metrics[metrics["level"] == "store_product"][
            ["model", "model_label", "MAE", "RMSE", "WAPE", "WAPE_pct"]
        ].to_dict("records"),
        "best_store_product_model": str(best["model"]),
        "best_store_product_wape_pct": float(best["WAPE_pct"]),
    }
    (OUTPUTS / "q4_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
