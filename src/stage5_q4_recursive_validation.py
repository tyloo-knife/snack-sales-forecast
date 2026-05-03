from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics
from src.features import (
    add_leakage_safe_sales_features,
    build_store_product_panel,
)
from src.models import (
    build_random_forest_forecaster,
    build_ridge_forecaster,
    clipped_predict,
    exp_smoothing_prediction,
    optimize_exp_smoothing_alpha,
)

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")
VALIDATION_END = pd.Timestamp("2022-03-30")
WINDOW_LENGTH = 7

NUMERIC_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "weekday",
    "month",
    "is_weekend",
    "max_temperature",
    "min_temperature",
    "wind_power",
    "is_holiday",
    "is_activity_day",
]
CATEGORICAL_FEATURES = ["store_id_str", "product_id_str", "category", "weather"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_LABELS = {
    "q1_store_product_exp_smoothing": "问题一门店-商品简单指数平滑",
    "baseline_moving_average_7": "Baseline移动平均(7日)",
    "baseline_rolling_mean_14": "Baseline滚动均值(14日)",
    "ridge": "综合Ridge回归",
    "random_forest": "综合随机森林",
}


def df_to_md(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    if df is None or df.empty:
        return "无。"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}"
            )
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


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUTS / name, index=False, encoding="utf-8-sig")


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_leakage_safe_sales_features(panel, TARGET)
    out["store_id_str"] = out["store_id"].astype(str)
    out["product_id_str"] = out["product_id"].astype(str)
    out["category"] = out["category"].astype(str)
    out["weather"] = out["weather"].astype(str).fillna("未知")
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def make_model(model_name: str):
    if model_name == "ridge":
        return build_ridge_forecaster(NUMERIC_FEATURES, CATEGORICAL_FEATURES, alpha=10.0)
    if model_name == "random_forest":
        return build_random_forest_forecaster(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    raise ValueError(model_name)


def strict_7day_windows(feature_df: pd.DataFrame) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[dict]]:
    starts = pd.to_datetime(
        ["2022-03-01", "2022-03-08", "2022-03-15", "2022-03-22", "2022-03-29"]
    )
    external_dates = set(
        feature_df.loc[feature_df["has_external_data"] == 1, "date"].dt.normalize().unique()
    )
    data_dates = set(feature_df["date"].dt.normalize().unique())
    windows = []
    excluded = []
    for start in starts:
        end = start + pd.Timedelta(days=WINDOW_LENGTH - 1)
        dates = pd.date_range(start, end, freq="D")
        has_all_sales = all(date in data_dates for date in dates)
        has_all_external = all(date in external_dates for date in dates)
        if end <= VALIDATION_END and has_all_sales and has_all_external:
            windows.append((start, end))
        else:
            excluded.append(
                {
                    "window_start": start.date().isoformat(),
                    "window_end": end.date().isoformat(),
                    "reason": "不足完整 7 日销售或外部变量覆盖，不能作为严格 7 日窗口",
                }
            )
    return windows, excluded


def history_feature_row(values: list[float]) -> dict:
    def lag(k: int) -> float:
        return float(values[-k]) if len(values) >= k else 0.0

    recent7 = values[-7:] if values else [0.0]
    recent14 = values[-14:] if values else [0.0]
    return {
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
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
    external = (
        feature_df[external_cols]
        .drop_duplicates("date")
        .set_index("date")
        .to_dict("index")
    )
    return combos, actual_lookup, external


def initial_history(panel: pd.DataFrame, combos: pd.DataFrame, window_start: pd.Timestamp) -> dict:
    history = {}
    before_window = panel[panel["date"] < window_start]
    for row in combos.itertuples():
        values = (
            before_window[
                (before_window["store_id"] == row.store_id)
                & (before_window["product_id"] == row.product_id)
            ]
            .sort_values("date")[TARGET]
            .astype(float)
            .tolist()
        )
        history[(row.store_id, row.product_id)] = values
    return history


def append_prediction_rows(
    rows: list[dict],
    batch: pd.DataFrame,
    actual_lookup: dict,
    model_name: str,
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
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature_source": "lag/rolling 使用窗口内已预测销量递推生成",
            }
        )


def recursive_baseline_validate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    model_name: str,
) -> pd.DataFrame:
    combos, actual_lookup, external_lookup = build_lookup_tables(panel, feature_df)
    rows = []
    window_size = 7 if model_name == "baseline_moving_average_7" else 14
    for window_start, window_end in windows:
        history = initial_history(panel, combos, window_start)
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch_rows = []
            ext = external_lookup[pd.Timestamp(date)]
            for row in combos.itertuples():
                values = history[(row.store_id, row.product_id)]
                recent = values[-window_size:] if values else [0.0]
                pred = max(0.0, float(np.mean(recent)))
                batch_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "store_id": row.store_id,
                        "store_name": row.store_name,
                        "product_id": row.product_id,
                        "product_name": row.product_name,
                        "category": row.category,
                        "prediction": pred,
                        "weather": ext["weather"],
                    }
                )
            batch = pd.DataFrame(batch_rows)
            append_prediction_rows(rows, batch, actual_lookup, model_name, window_start)
            # The strict protocol appends predictions, not validation truth.
            for pred_row in batch.itertuples():
                history[(pred_row.store_id, pred_row.product_id)].append(float(pred_row.prediction))
    return pd.DataFrame(rows)


def recursive_exp_smoothing_validate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    combos, actual_lookup, external_lookup = build_lookup_tables(panel, feature_df)
    model_name = "q1_store_product_exp_smoothing"
    rows = []
    for window_start, window_end in windows:
        history = initial_history(panel, combos, window_start)
        alphas = {
            (row.store_id, row.product_id): optimize_exp_smoothing_alpha(
                pd.Series(history[(row.store_id, row.product_id)], dtype=float)
            )
            for row in combos.itertuples()
        }
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch_rows = []
            ext = external_lookup[pd.Timestamp(date)]
            for row in combos.itertuples():
                key = (row.store_id, row.product_id)
                pred = max(0.0, float(exp_smoothing_prediction(history[key], alphas[key])))
                batch_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "store_id": row.store_id,
                        "store_name": row.store_name,
                        "product_id": row.product_id,
                        "product_name": row.product_name,
                        "category": row.category,
                        "prediction": pred,
                        "weather": ext["weather"],
                    }
                )
            batch = pd.DataFrame(batch_rows)
            append_prediction_rows(rows, batch, actual_lookup, model_name, window_start)
            for pred_row in batch.itertuples():
                history[(pred_row.store_id, pred_row.product_id)].append(float(pred_row.prediction))
    return pd.DataFrame(rows)


def recursive_ml_validate(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    model_name: str,
) -> pd.DataFrame:
    combos, actual_lookup, external_lookup = build_lookup_tables(panel, feature_df)
    rows = []
    for window_start, window_end in windows:
        train = feature_df[
            (feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)
        ].copy()
        model = make_model(model_name)
        model.fit(train[FEATURES], train[TARGET])

        history = initial_history(panel, combos, window_start)
        for date in pd.date_range(window_start, window_end, freq="D"):
            batch_rows = []
            ext = external_lookup[pd.Timestamp(date)]
            for row in combos.itertuples():
                dynamic = history_feature_row(history[(row.store_id, row.product_id)])
                batch_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "store_id": row.store_id,
                        "store_name": row.store_name,
                        "product_id": row.product_id,
                        "product_name": row.product_name,
                        "category": row.category,
                        **dynamic,
                        "weekday": int(ext["weekday"]),
                        "month": int(ext["month"]),
                        "is_weekend": int(ext["is_weekend"]),
                        "max_temperature": float(ext["max_temperature"]),
                        "min_temperature": float(ext["min_temperature"]),
                        "wind_power": float(ext["wind_power"]),
                        "is_holiday": int(ext["is_holiday"]),
                        "is_activity_day": int(ext["is_activity_day"]),
                        "weather": str(ext["weather"]),
                        "store_id_str": str(row.store_id),
                        "product_id_str": str(row.product_id),
                    }
                )
            batch = pd.DataFrame(batch_rows)
            for col in NUMERIC_FEATURES:
                batch[col] = pd.to_numeric(batch[col], errors="coerce").fillna(0.0)
            batch["prediction"] = clipped_predict(model, batch[FEATURES])
            append_prediction_rows(rows, batch, actual_lookup, model_name, window_start)
            for pred_row in batch.itertuples():
                history[(pred_row.store_id, pred_row.product_id)].append(float(pred_row.prediction))
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
    else:
        raise ValueError(level)
    return (
        preds.groupby(group_cols, dropna=False, as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(group_cols)
    )


def metrics_from_predictions(preds: pd.DataFrame, level: str, validation_mode: str) -> pd.DataFrame:
    metric_input = aggregate_predictions(preds, level)
    rows = []
    for (model, label), group in metric_input.groupby(["model", "model_label"], dropna=False):
        metrics = calculate_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                "validation_mode": validation_mode,
                "level": level,
                "model": model,
                "model_label": label,
                **metrics,
                "WAPE_pct": metrics["WAPE"] * 100,
                "actual_sum": float(group["actual"].sum()),
                "prediction_sum": float(group["prediction"].sum()),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def all_level_metrics(preds: pd.DataFrame, validation_mode: str) -> pd.DataFrame:
    return pd.concat(
        [
            metrics_from_predictions(preds, "store_product", validation_mode),
            metrics_from_predictions(preds, "store", validation_mode),
            metrics_from_predictions(preds, "product", validation_mode),
        ],
        ignore_index=True,
    )


def current_daily_rolling_metrics(strict_preds: pd.DataFrame) -> pd.DataFrame:
    current_path = TABLES / "q4_validation_predictions_store_product.csv"
    current = pd.read_csv(current_path)
    current["date"] = pd.to_datetime(current["date"])
    current["window_start"] = pd.to_datetime(current["window_start"], errors="coerce")
    model_order = strict_preds["model"].drop_duplicates().tolist()
    validation_dates = set(pd.to_datetime(strict_preds["date"]).dt.normalize().unique())
    current = current[
        current["model"].isin(model_order)
        & current["date"].dt.normalize().isin(validation_dates)
    ].copy()
    current["validation_mode"] = "daily_rolling_one_step"
    return all_level_metrics(current, "daily_rolling_one_step")


def comparison_table(daily_metrics: pd.DataFrame, recursive_metrics: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["level", "model", "model_label"]
    daily = daily_metrics.rename(
        columns={
            "MAE": "daily_MAE",
            "RMSE": "daily_RMSE",
            "WAPE": "daily_WAPE",
            "WAPE_pct": "daily_WAPE_pct",
            "actual_sum": "daily_actual_sum",
            "prediction_sum": "daily_prediction_sum",
            "n": "daily_n",
        }
    )
    rec = recursive_metrics.rename(
        columns={
            "MAE": "recursive_MAE",
            "RMSE": "recursive_RMSE",
            "WAPE": "recursive_WAPE",
            "WAPE_pct": "recursive_WAPE_pct",
            "actual_sum": "recursive_actual_sum",
            "prediction_sum": "recursive_prediction_sum",
            "n": "recursive_n",
        }
    )
    merged = daily[merge_cols + [c for c in daily.columns if c.startswith("daily_")]].merge(
        rec[merge_cols + [c for c in rec.columns if c.startswith("recursive_")]],
        on=merge_cols,
        how="outer",
    )
    merged["WAPE_change_pct_points"] = merged["recursive_WAPE_pct"] - merged["daily_WAPE_pct"]
    merged["WAPE_relative_change_pct"] = (
        (merged["recursive_WAPE"] - merged["daily_WAPE"]) / merged["daily_WAPE"] * 100
    )
    return merged.sort_values(["level", "recursive_WAPE", "daily_WAPE"]).reset_index(drop=True)


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


def build_report(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    excluded_windows: list[dict],
    recursive_metrics: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    store_product_recursive = recursive_metrics[recursive_metrics["level"] == "store_product"]
    best_current = (
        pd.read_csv(TABLES / "q4_store_product_model_metrics.csv")
        .query("model in ['ridge', 'random_forest']")
        .sort_values("WAPE")
        .iloc[0]
    )
    best_model = best_current["model"]
    best_label = best_current["model_label"]
    strict_best_row = store_product_recursive[store_product_recursive["model"] == best_model].iloc[0]
    strict_q1_row = store_product_recursive[
        store_product_recursive["model"] == "q1_store_product_exp_smoothing"
    ].iloc[0]
    strict_baseline_rows = store_product_recursive[
        store_product_recursive["model"].isin(
            ["baseline_moving_average_7", "baseline_rolling_mean_14"]
        )
    ].sort_values("WAPE")
    strict_best_baseline = strict_baseline_rows.iloc[0]
    still_better = strict_best_row["WAPE"] < strict_best_baseline["WAPE"]

    daily_best_row = daily_metrics[
        (daily_metrics["level"] == "store_product") & (daily_metrics["model"] == best_model)
    ].iloc[0]
    performance_drop = strict_best_row["WAPE"] - daily_best_row["WAPE"]

    window_df = pd.DataFrame(
        [
            {
                "window_start": start.date().isoformat(),
                "window_end": end.date().isoformat(),
                "length": (end - start).days + 1,
            }
            for start, end in windows
        ]
    )
    excluded_df = pd.DataFrame(excluded_windows)

    level_table = recursive_metrics[
        ["level", "model_label", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]
    ]
    compare_table = comparison[
        [
            "level",
            "model_label",
            "daily_WAPE_pct",
            "recursive_WAPE_pct",
            "WAPE_change_pct_points",
            "WAPE_relative_change_pct",
        ]
    ]

    still_better_text = "仍然优于" if still_better else "不再优于"
    drop_note = (
        "严格递推 WAPE 高于日滚动口径，说明原验证受益于窗口内真实销量参与滞后和滚动特征。"
        if performance_drop > 0
        else "严格递推 WAPE 未高于日滚动口径，说明该模型在本次窗口上未表现出递推误差扩大。"
    )

    return f"""# 问题四严格 7 日递推窗口验证报告

执行日期：2026-05-02

## 1. 验证口径修正

原问题四验证保留为“日滚动一步预测验证”：每个预测日的 `lag_1`、`rolling_mean_7`、`rolling_mean_14` 来自已经观测到的真实历史销量。专项检查显示，该口径没有当天销量直接进入当天特征的硬泄露，但在一个 7 日窗口内，后续日期会使用窗口前几天的真实销量，因此不能直接代表“窗口开始时一次性预测未来 7 天”。

本次新增“严格 7 日递推窗口验证”：每个窗口只用 `date < window_start` 的真实销量初始化历史缓存；窗口内第 1 天预测后，把预测值而不是真实值写回该门店-商品组合缓存；窗口内第 2 至第 7 天的滞后和滚动特征都从这个缓存生成。

## 2. 验证窗口

采用完整 7 日窗口如下：

{df_to_md(window_df)}

未纳入窗口如下：

{df_to_md(excluded_df)}

说明：`2022-03-29` 起始窗口无法形成完整 7 日验证窗口，并且后续日期外部变量覆盖不足，因此不用于严格 7 日递推验证。为公平比较，日滚动一步预测结果也截取到上述完整窗口对应日期。

## 3. 严格 7 日递推验证误差

{df_to_md(level_table, 40)}

## 4. 与日滚动一步预测结果对比

{df_to_md(compare_table, 60)}

## 5. 当前最优模型是否仍优于 baseline

阶段 5 日滚动口径下的最优综合模型为 `{best_label}`。在严格 7 日递推窗口验证下，该模型门店-商品层级 WAPE 为 {strict_best_row['WAPE_pct']:.2f}%；严格递推下最优 baseline 为 `{strict_best_baseline['model_label']}`，WAPE 为 {strict_best_baseline['WAPE_pct']:.2f}%。因此，当前最优综合模型在严格 7 日递推口径下{still_better_text} baseline。

同时需要注意，若把问题一门店-商品简单指数平滑也作为候选模型比较，其严格递推 WAPE 为 {strict_q1_row['WAPE_pct']:.2f}%，略低于 `{best_label}`。因此论文中不能写“综合模型在严格递推口径下优于所有候选模型”，更严谨的表述是“综合 Ridge 在严格递推口径下仍优于 q4 baseline，但未能超过问题一简单指数平滑”。

## 6. 性能变化原因

`{best_label}` 在日滚动一步预测口径下门店-商品 WAPE 为 {daily_best_row['WAPE_pct']:.2f}%，在严格 7 日递推口径下为 {strict_best_row['WAPE_pct']:.2f}%，变化 {performance_drop * 100:.2f} 个百分点。{drop_note}

递推验证通常比日滚动验证更难，原因是窗口内第 2 至第 7 天的 `lag_1`、`rolling_mean_7`、`rolling_mean_14` 不再使用真实销量，而是使用前面日期的预测值。若第 1 天或第 2 天预测偏差较大，这个偏差会进入后续特征，造成误差累积。该结果更接近最终未来 7 天预测的实际使用方式。

## 7. 论文写法建议

原阶段 5 的误差表应标注为“日滚动一步预测验证结果”，适合说明模型在每天更新真实销量后的短期预测能力。本报告新增的误差表应作为“严格未来 7 天预测验证”的主要依据，适合支撑问题四最终 7 日预测结论。两种验证口径的结论不能混用。
"""


def main() -> None:
    for path in [TABLES, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data/processed/modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    feature_df = prepare_model_frame(panel)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    feature_df["has_external_data"] = feature_df["has_external_data"].fillna(0).astype(int)

    windows, excluded_windows = strict_7day_windows(feature_df)
    if not windows:
        raise RuntimeError("没有可用的完整 7 日验证窗口。")

    recursive_preds = pd.concat(
        [
            recursive_exp_smoothing_validate(panel, feature_df, windows),
            recursive_baseline_validate(panel, feature_df, windows, "baseline_moving_average_7"),
            recursive_baseline_validate(panel, feature_df, windows, "baseline_rolling_mean_14"),
            recursive_ml_validate(panel, feature_df, windows, "ridge"),
            recursive_ml_validate(panel, feature_df, windows, "random_forest"),
        ],
        ignore_index=True,
    )
    recursive_preds["abs_error"] = (
        recursive_preds["actual"] - recursive_preds["prediction"]
    ).abs()
    save_csv(recursive_preds, "q4_recursive_7day_validation.csv")

    recursive_metrics = all_level_metrics(recursive_preds, "strict_recursive_7day")
    daily_metrics = current_daily_rolling_metrics(recursive_preds)
    comparison = comparison_table(daily_metrics, recursive_metrics)
    save_csv(recursive_metrics, "q4_recursive_7day_metrics.csv")
    save_csv(comparison, "q4_recursive_7day_validation_comparison.csv")

    report = build_report(windows, excluded_windows, recursive_metrics, daily_metrics, comparison)
    (OUTPUTS / "q4_recursive_7day_model_comparison.md").write_text(report, encoding="utf-8")

    store_product_recursive = recursive_metrics[recursive_metrics["level"] == "store_product"]
    ridge_row = store_product_recursive[store_product_recursive["model"] == "ridge"].iloc[0]
    q1_row = store_product_recursive[
        store_product_recursive["model"] == "q1_store_product_exp_smoothing"
    ].iloc[0]
    best_baseline = store_product_recursive[
        store_product_recursive["model"].isin(
            ["baseline_moving_average_7", "baseline_rolling_mean_14"]
        )
    ].sort_values("WAPE").iloc[0]
    daily_ridge = daily_metrics[
        (daily_metrics["level"] == "store_product") & (daily_metrics["model"] == "ridge")
    ].iloc[0]
    relation = "仍优于" if ridge_row["WAPE"] < best_baseline["WAPE"] else "不再优于"
    row = (
        "| 2026-05-02 | 问题四严格 7 日递推窗口验证 | "
        "processed: modeling_base_table.csv；保留原 q4 日滚动验证结果 | "
        "问题一指数平滑、7日/14日baseline、Ridge、RandomForest | "
        "滞后销量、滚动均值、星期、月份、门店、商品、类别、天气、节假日、活动日；验证窗口内 lag/rolling 使用预测值递推 | "
        f"日滚动 Ridge WAPE={daily_ridge['WAPE_pct']:.2f}%；严格递推 Ridge WAPE={ridge_row['WAPE_pct']:.2f}%；最优baseline WAPE={best_baseline['WAPE_pct']:.2f}%；问题一指数平滑 WAPE={q1_row['WAPE_pct']:.2f}% | "
        f"已新增严格 7 日递推验证；Ridge 在该口径下{relation} q4 baseline，但门店-商品层级略差于问题一指数平滑 | "
        "严格递推通常更难，误差可能因预测值进入后续 lag/rolling 特征而累积；日滚动与递推验证适用场景不同，不能混用结论 | "
        "按严格 7 日递推口径修订问题四论文表述 |"
    )
    prepend_result_log(row)

    summary = {
        "windows": [
            {
                "window_start": str(start.date()),
                "window_end": str(end.date()),
                "length": int((end - start).days + 1),
            }
            for start, end in windows
        ],
        "excluded_windows": excluded_windows,
        "recursive_prediction_rows": int(len(recursive_preds)),
        "ridge_daily_rolling_store_product_wape": float(daily_ridge["WAPE"]),
        "ridge_recursive_store_product_wape": float(ridge_row["WAPE"]),
        "best_baseline_recursive_store_product_model": str(best_baseline["model"]),
        "best_baseline_recursive_store_product_wape": float(best_baseline["WAPE"]),
        "ridge_vs_baseline_recursive": relation,
    }
    (OUTPUTS / "q4_recursive_7day_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
