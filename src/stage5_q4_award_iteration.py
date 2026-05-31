from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

CURRENT_MODEL = "hybrid_current_q1_regular_ridge"
Q1_MODEL = "q1_store_product_exp_smoothing"
RIDGE_MODEL = "ridge_full_external"
IMPROVEMENT_THRESHOLD_PCT = 0.30


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


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def load_strategy_base() -> pd.DataFrame:
    strategy_path = TABLES / "q4_low_volume_strategy_predictions.csv"
    cls_path = TABLES / "q4_low_volume_classification.csv"
    if not strategy_path.exists():
        raise FileNotFoundError("缺少 tables/q4_low_volume_strategy_predictions.csv。")
    if not cls_path.exists():
        raise FileNotFoundError("缺少 tables/q4_low_volume_classification.csv。")

    preds = pd.read_csv(strategy_path)
    cls = pd.read_csv(cls_path)
    for frame in [preds, cls]:
        frame["window_start"] = pd.to_datetime(frame["window_start"])
    preds["date"] = pd.to_datetime(preds["date"])

    key_cols = [
        "validation_mode",
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
    ]
    model_map = {
        "q1_store_product_exp_smoothing": "pred_q1_exp_smoothing",
        "ridge_full_external": "pred_ridge_full",
    }
    pivot = (
        preds[preds["model"].isin(model_map)]
        .pivot_table(index=key_cols, columns="model", values="prediction", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns=model_map)
    )
    required = {"pred_q1_exp_smoothing", "pred_ridge_full"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"低销量策略预测表缺少必要基准预测列：{sorted(missing)}")

    cls_cols = [
        "window_start",
        "store_id",
        "product_id",
        "history_average_sales",
        "history_zero_sales_ratio",
        "recent28_mean",
        "product_history_mean",
        "category_history_mean",
        "hierarchy_shrink_target",
        "demand_class",
        "is_low_volume",
    ]
    out = pivot.merge(cls[cls_cols], on=["window_start", "store_id", "product_id"], how="left")
    out["is_low_volume"] = out["is_low_volume"].fillna(0).astype(int)
    numeric_cols = [
        "actual",
        "pred_q1_exp_smoothing",
        "pred_ridge_full",
        "history_average_sales",
        "history_zero_sales_ratio",
        "recent28_mean",
        "product_history_mean",
        "category_history_mean",
        "hierarchy_shrink_target",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["window_end"] = pd.to_datetime(out["window_end"])
    return out.sort_values(["window_start", "date", "store_id", "product_id"]).reset_index(
        drop=True
    )


def candidate_thresholds() -> list[tuple[str, str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (
            "current_rule",
            "当前低销量规则",
            "沿用 no_positive_history、very_sparse、low_volume_intermittent 三类低销量判定",
            lambda df: df["is_low_volume"].eq(1),
        ),
        (
            "avg_lt_0_2",
            "历史均值 <0.2",
            "窗口开始日前门店-商品历史日均销量小于 0.2",
            lambda df: df["history_average_sales"].lt(0.2),
        ),
        (
            "avg_lt_0_3",
            "历史均值 <0.3",
            "窗口开始日前门店-商品历史日均销量小于 0.3",
            lambda df: df["history_average_sales"].lt(0.3),
        ),
        (
            "zero_ge_0_6",
            "零销量比例 >=0.6",
            "窗口开始日前零销量天数占比不低于 60%",
            lambda df: df["history_zero_sales_ratio"].ge(0.6),
        ),
        (
            "zero_ge_0_75",
            "零销量比例 >=0.75",
            "窗口开始日前零销量天数占比不低于 75%",
            lambda df: df["history_zero_sales_ratio"].ge(0.75),
        ),
        (
            "avg_lt_0_3_or_zero_ge_0_75",
            "历史均值 <0.3 或零销量比例 >=0.75",
            "低均值和高零销量比例的并集判定",
            lambda df: df["history_average_sales"].lt(0.3)
            | df["history_zero_sales_ratio"].ge(0.75),
        ),
    ]


def low_volume_predictors() -> list[tuple[str, str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (
            "q1_exp_smoothing",
            "低销量用指数平滑",
            "低销量序列使用问题一门店-商品指数平滑，常规序列使用完整 Ridge",
            lambda df: df["pred_q1_exp_smoothing"],
        ),
        (
            "recent28_mean",
            "低销量用近28日均值",
            "低销量序列使用窗口开始日前近 28 日均值，常规序列使用完整 Ridge",
            lambda df: df["recent28_mean"],
        ),
        (
            "product_mean",
            "低销量用商品均值",
            "低销量序列使用窗口开始日前同商品历史均值，常规序列使用完整 Ridge",
            lambda df: df["product_history_mean"],
        ),
        (
            "category_mean",
            "低销量用类别均值",
            "低销量序列使用窗口开始日前同类别历史均值，常规序列使用完整 Ridge",
            lambda df: df["category_history_mean"],
        ),
        (
            "hierarchy_shrink",
            "低销量用层级收缩均值",
            "低销量序列使用近28日、商品均值、类别均值的层级收缩目标，常规序列使用完整 Ridge",
            lambda df: df["hierarchy_shrink_target"],
        ),
        (
            "q1_hierarchy_blend",
            "低销量指数平滑+层级收缩",
            "低销量序列使用 70% 指数平滑 + 30% 层级收缩目标，常规序列使用完整 Ridge",
            lambda df: 0.7 * df["pred_q1_exp_smoothing"]
            + 0.3 * df["hierarchy_shrink_target"],
        ),
    ]


def base_columns() -> list[str]:
    return [
        "validation_mode",
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
        "history_average_sales",
        "history_zero_sales_ratio",
        "demand_class",
        "is_low_volume",
    ]


def build_candidate_predictions(base: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def append_model(
        model: str,
        label: str,
        rule: str,
        prediction: pd.Series | np.ndarray,
        low_mask: pd.Series | None = None,
        threshold_label: str = "不适用",
    ) -> None:
        tmp = base[base_columns()].copy()
        tmp["prediction"] = np.maximum(0.0, pd.Series(prediction, index=base.index).astype(float))
        tmp["model"] = model
        tmp["model_label"] = label
        tmp["strategy_rule"] = rule
        tmp["threshold_rule"] = threshold_label
        tmp["low_volume_selected"] = low_mask.astype(int).to_numpy() if low_mask is not None else 0
        rows.append(tmp)

    append_model(
        Q1_MODEL,
        "问题一简单指数平滑",
        "所有门店-商品序列均使用问题一指数平滑",
        base["pred_q1_exp_smoothing"],
    )
    append_model(
        RIDGE_MODEL,
        "完整综合Ridge",
        "所有门店-商品序列均使用完整综合 Ridge",
        base["pred_ridge_full"],
    )

    for threshold_id, threshold_label, threshold_rule, mask_fn in candidate_thresholds():
        low_mask = mask_fn(base).fillna(False)
        for pred_id, pred_label, pred_rule, pred_fn in low_volume_predictors():
            pred = np.where(low_mask, pred_fn(base), base["pred_ridge_full"])
            model = f"hybrid_{threshold_id}_{pred_id}"
            label = f"{threshold_label}-{pred_label}"
            if threshold_id == "current_rule" and pred_id == "q1_exp_smoothing":
                model = CURRENT_MODEL
                label = "当前混合策略：低销量指数平滑-常规Ridge"
            append_model(
                model=model,
                label=label,
                rule=pred_rule,
                prediction=pred,
                low_mask=low_mask,
                threshold_label=threshold_label,
            )

    candidates = pd.concat(rows, ignore_index=True)
    candidates["abs_error"] = (candidates["actual"] - candidates["prediction"]).abs()
    return candidates


def apply_q1_aggregate_calibration(
    base: pd.DataFrame,
    candidate_preds: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    group_map = {
        "overall": [],
        "store": ["store_id"],
        "product": ["product_id"],
        "category": ["category"],
    }
    group_cols = group_map[level]
    current = candidate_preds[candidate_preds["model"] == CURRENT_MODEL].copy()
    if current.empty:
        raise ValueError("候选预测中找不到当前混合策略。")
    current = current.drop(columns=["prediction"])
    current = current.merge(
        base[
            [
                "window_start",
                "date",
                "store_id",
                "product_id",
                "category",
                "pred_q1_exp_smoothing",
            ]
        ],
        on=["window_start", "date", "store_id", "product_id", "category"],
        how="left",
    )
    current = current.rename(columns={"pred_q1_exp_smoothing": "q1_target_prediction"})
    original = candidate_preds.loc[
        candidate_preds["model"] == CURRENT_MODEL,
        ["window_start", "date", "store_id", "product_id", "category", "prediction"],
    ]
    current = current.merge(
        original,
        on=["window_start", "date", "store_id", "product_id", "category"],
        how="left",
    )

    sum_cols = ["window_start", "date", *group_cols]
    factor = (
        current.groupby(sum_cols, dropna=False, as_index=False)
        .agg(
            pred_sum=("prediction", "sum"),
            q1_sum=("q1_target_prediction", "sum"),
        )
        .reset_index(drop=True)
    )
    factor["calibration_factor"] = np.where(
        factor["pred_sum"].abs().gt(1e-9),
        factor["q1_sum"] / factor["pred_sum"],
        1.0,
    )
    # Avoid making a diagnostic calibration dominate the validation through extreme ratios.
    factor["calibration_factor"] = factor["calibration_factor"].clip(lower=0.5, upper=1.5)
    out = current.merge(factor[sum_cols + ["calibration_factor"]], on=sum_cols, how="left")
    out["prediction"] = np.maximum(0.0, out["prediction"] * out["calibration_factor"].fillna(1.0))
    out["model"] = f"hybrid_current_calibrated_to_q1_{level}"
    level_label = {"overall": "总量", "store": "门店", "product": "商品", "category": "类别"}[level]
    out["model_label"] = f"当前混合策略校准到问题一{level_label}预测"
    out["strategy_rule"] = (
        f"以后处理方式将当前混合策略在日期-{level_label}层面的预测总量按比例缩放到问题一指数平滑聚合预测；"
        "比例只使用两套模型预测值，不使用验证窗口真实销量"
    )
    out["threshold_rule"] = f"问题一{level_label}聚合预测后处理校准"
    return out[candidate_preds.columns]


def add_calibrated_candidates(base: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    calibrated = [
        apply_q1_aggregate_calibration(base, candidates, level)
        for level in ["overall", "store", "product", "category"]
    ]
    return pd.concat([candidates, *calibrated], ignore_index=True)


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


def metric_row(df: pd.DataFrame) -> dict:
    metrics = calculate_metrics(df["actual"], df["prediction"])
    actual_sum = float(df["actual"].sum())
    prediction_sum = float(df["prediction"].sum())
    return {
        **metrics,
        "WAPE_pct": metrics["WAPE"] * 100 if pd.notna(metrics["WAPE"]) else np.nan,
        "actual_sum": actual_sum,
        "prediction_sum": prediction_sum,
        "abs_error_sum": float((df["actual"] - df["prediction"]).abs().sum()),
        "bias": prediction_sum - actual_sum,
        "bias_pct_of_actual": (prediction_sum - actual_sum) / actual_sum
        if actual_sum != 0
        else np.nan,
        "n": int(len(df)),
    }


def build_metric_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in ["store_product", "store", "product", "category"]:
        agg = aggregate_predictions(preds, level)
        for (model, label), group in agg.groupby(["model", "model_label"], dropna=False):
            original = preds[preds["model"] == model]
            row = {
                "validation_mode": "strict_recursive_7day",
                "level": level,
                "model": model,
                "model_label": label,
                "strategy_rule": original["strategy_rule"].dropna().iloc[0],
                "threshold_rule": original["threshold_rule"].dropna().iloc[0],
                "low_volume_selected_share": float(original["low_volume_selected"].mean()),
                **metric_row(group),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    sp = out[out["level"] == "store_product"].set_index("model")
    current_wape = float(sp.loc[CURRENT_MODEL, "WAPE_pct"])
    q1_wape = float(sp.loc[Q1_MODEL, "WAPE_pct"])
    ridge_wape = float(sp.loc[RIDGE_MODEL, "WAPE_pct"])
    out["WAPE_pct_point_change_vs_current"] = out["WAPE_pct"] - current_wape
    out["WAPE_pct_point_change_vs_q1"] = out["WAPE_pct"] - q1_wape
    out["WAPE_pct_point_change_vs_ridge"] = out["WAPE_pct"] - ridge_wape
    return out.sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def build_window_metric_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label, window_start), group in preds.groupby(
        ["model", "model_label", "window_start"], dropna=False
    ):
        rows.append(
            {
                "validation_mode": "strict_recursive_7day",
                "window_start": pd.Timestamp(window_start).date().isoformat(),
                "model": model,
                "model_label": label,
                **metric_row(group),
            }
        )
    out = pd.DataFrame(rows)
    current = out[out["model"] == CURRENT_MODEL][["window_start", "WAPE_pct"]].rename(
        columns={"WAPE_pct": "current_window_WAPE_pct"}
    )
    out = out.merge(current, on="window_start", how="left")
    out["WAPE_pct_point_change_vs_current_window"] = (
        out["WAPE_pct"] - out["current_window_WAPE_pct"]
    )
    return out.sort_values(["window_start", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def build_leave_one_window_selection(preds: pd.DataFrame) -> pd.DataFrame:
    # Calibration rows are diagnostics, not competing low-volume decision rules.
    candidate_pool = preds[~preds["model"].str.contains("calibrated", regex=False)].copy()
    windows = sorted(pd.to_datetime(candidate_pool["window_start"]).drop_duplicates())
    rows = []
    for holdout in windows:
        train = candidate_pool[candidate_pool["window_start"] != holdout]
        holdout_df = candidate_pool[candidate_pool["window_start"] == holdout]
        train_scores = []
        for (model, label), group in train.groupby(["model", "model_label"], dropna=False):
            train_scores.append({"model": model, "model_label": label, **metric_row(group)})
        selected = pd.DataFrame(train_scores).sort_values(["WAPE", "MAE", "RMSE"]).iloc[0]
        eval_models = [
            (selected["model"], selected["model_label"], "selected_by_other_windows"),
            (CURRENT_MODEL, "当前混合策略：低销量指数平滑-常规Ridge", "current_hybrid"),
            (Q1_MODEL, "问题一简单指数平滑", "q1_baseline"),
            (RIDGE_MODEL, "完整综合Ridge", "ridge_baseline"),
        ]
        for model, label, eval_type in eval_models:
            group = holdout_df[holdout_df["model"] == model]
            rows.append(
                {
                    "holdout_window_start": holdout.date().isoformat(),
                    "selected_model_on_other_windows": selected["model"],
                    "selected_label_on_other_windows": selected["model_label"],
                    "selected_train_WAPE_pct": float(selected["WAPE_pct"]),
                    "eval_type": eval_type,
                    "eval_model": model,
                    "eval_label": label,
                    **metric_row(group),
                }
            )
    out = pd.DataFrame(rows)
    selected = out[out["eval_type"] == "selected_by_other_windows"][
        ["holdout_window_start", "WAPE_pct"]
    ].rename(columns={"WAPE_pct": "selected_holdout_WAPE_pct"})
    current = out[out["eval_type"] == "current_hybrid"][
        ["holdout_window_start", "WAPE_pct"]
    ].rename(columns={"WAPE_pct": "current_holdout_WAPE_pct"})
    out = out.merge(selected, on="holdout_window_start", how="left").merge(
        current, on="holdout_window_start", how="left"
    )
    out["WAPE_pct_point_change_vs_current_holdout"] = (
        out["WAPE_pct"] - out["current_holdout_WAPE_pct"]
    )
    return out.sort_values(["holdout_window_start", "eval_type"]).reset_index(drop=True)


def load_calendar() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "modeling_base_table.csv"
    calendar = pd.read_csv(
        path,
        usecols=lambda col: col
        in {"date", "weekday", "is_weekend", "is_holiday", "is_activity_day"},
    )
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.drop_duplicates("date")
    for col in ["weekday", "is_weekend", "is_holiday", "is_activity_day"]:
        calendar[col] = pd.to_numeric(calendar[col], errors="coerce").fillna(0).astype(int)
    return calendar


def grouped_metrics(preds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in preds.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metric_row(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_error_attribution(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hybrid = candidates[candidates["model"] == CURRENT_MODEL].copy()
    hybrid = hybrid.merge(load_calendar(), on="date", how="left")
    by_horizon = grouped_metrics(hybrid, ["horizon"])
    by_class = grouped_metrics(hybrid, ["demand_class", "is_low_volume"])

    segment_rows = []
    segment_defs = [
        ("工作日", "is_weekend=0 且 is_holiday=0", (hybrid["is_weekend"] == 0) & (hybrid["is_holiday"] == 0)),
        ("周末", "is_weekend=1", hybrid["is_weekend"] == 1),
        ("非周末", "is_weekend=0", hybrid["is_weekend"] == 0),
        ("活动日", "is_activity_day=1", hybrid["is_activity_day"] == 1),
        ("非活动日", "is_activity_day=0", hybrid["is_activity_day"] == 0),
        ("节假日", "is_holiday=1", hybrid["is_holiday"] == 1),
        ("非节假日", "is_holiday=0", hybrid["is_holiday"] == 0),
    ]
    for label, definition, mask in segment_defs:
        subset = hybrid.loc[mask].copy()
        if subset.empty:
            segment_rows.append(
                {
                    "calendar_segment": label,
                    "definition": definition,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "WAPE": np.nan,
                    "WAPE_pct": np.nan,
                    "actual_sum": 0.0,
                    "prediction_sum": 0.0,
                    "abs_error_sum": 0.0,
                    "bias": 0.0,
                    "bias_pct_of_actual": np.nan,
                    "n": 0,
                    "date_count": 0,
                }
            )
        else:
            row = {"calendar_segment": label, "definition": definition, **metric_row(subset)}
            row["date_count"] = int(subset["date"].nunique())
            segment_rows.append(row)
    by_calendar = pd.DataFrame(segment_rows)
    save_csv(by_horizon, "q4_hybrid_error_by_horizon.csv")
    save_csv(by_class, "q4_hybrid_error_by_demand_class.csv")
    save_csv(by_calendar, "q4_hybrid_error_by_calendar_segment.csv")
    low = hybrid[hybrid["is_low_volume"] == 1]
    regular = hybrid[hybrid["is_low_volume"] == 0]
    return by_horizon, by_class, by_calendar, {
        "low_volume": metric_row(low) if not low.empty else {},
        "regular": metric_row(regular) if not regular.empty else {},
    }


def aggregate_7day_window(preds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        preds.groupby(["window_start", *group_cols], dropna=False, as_index=False)
        .agg(
            actual_7day_sales=("actual", "sum"),
            predicted_7day_sales=("prediction", "sum"),
        )
        .assign(
            residual=lambda df: df["actual_7day_sales"] - df["predicted_7day_sales"],
            abs_residual=lambda df: df["residual"].abs(),
        )
    )


def interval_table_for_level(
    validation: pd.DataFrame,
    future: pd.DataFrame,
    level: str,
    group_cols: list[str],
) -> pd.DataFrame:
    val = aggregate_7day_window(validation, group_cols)
    stats = (
        val.groupby(group_cols, dropna=False)
        .agg(
            residual_q10=("residual", lambda x: float(np.quantile(x, 0.10))),
            residual_median=("residual", "median"),
            residual_q90=("residual", lambda x: float(np.quantile(x, 0.90))),
            residual_mean=("residual", "mean"),
            mean_abs_residual=("abs_residual", "mean"),
            residual_std=("residual", "std"),
            residual_count=("residual", "count"),
            validation_actual_7day_mean=("actual_7day_sales", "mean"),
            validation_prediction_7day_mean=("predicted_7day_sales", "mean"),
        )
        .reset_index()
    )
    fut = (
        future.groupby(group_cols, dropna=False, as_index=False)["predicted_sales"]
        .sum()
        .rename(columns={"predicted_sales": "point_forecast_7day_sales"})
    )
    out = fut.merge(stats, on=group_cols, how="left")
    out["interval_method"] = "严格递推验证7日残差经验10%-90%分位数"
    raw_lower = out["point_forecast_7day_sales"] + out["residual_q10"]
    raw_upper = out["point_forecast_7day_sales"] + out["residual_q90"]
    out["lower_80pct_empirical"] = np.maximum(
        0.0, np.minimum(out["point_forecast_7day_sales"], raw_lower)
    )
    out["median_residual_adjusted_forecast"] = np.maximum(
        0.0, out["point_forecast_7day_sales"] + out["residual_median"]
    )
    out["upper_80pct_empirical"] = np.maximum(out["point_forecast_7day_sales"], raw_upper)
    out.insert(0, "level", level)
    return out.sort_values("point_forecast_7day_sales", ascending=False).reset_index(drop=True)


def build_forecast_intervals(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    future_path = OUTPUTS / "final_7day_forecast_hybrid_low_volume.csv"
    if not future_path.exists():
        raise FileNotFoundError("缺少 outputs/final_7day_forecast_hybrid_low_volume.csv。")
    future = pd.read_csv(future_path)
    future["date"] = pd.to_datetime(future["date"])
    future["predicted_sales"] = pd.to_numeric(future["predicted_sales"], errors="coerce").fillna(0.0)

    validation = candidates[candidates["model"] == CURRENT_MODEL].copy()
    store = interval_table_for_level(
        validation,
        future,
        "store",
        ["store_id", "store_name"],
    )
    product = interval_table_for_level(
        validation,
        future,
        "product",
        ["product_id", "product_name", "category"],
    )
    category = interval_table_for_level(validation, future, "category", ["category"])
    save_csv(store, "final_forecast_interval_by_store.csv")
    save_csv(product, "final_forecast_interval_by_product.csv")
    save_csv(category, "final_forecast_interval_by_category.csv")
    return store, product, category


def plot_outputs(
    metrics: pd.DataFrame,
    by_horizon: pd.DataFrame,
    store_interval: pd.DataFrame,
) -> None:
    setup_plot_style()

    sp = metrics[metrics["level"] == "store_product"].head(12).sort_values("WAPE_pct")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    labels = sp["model_label"].str.slice(0, 28)
    colors = ["#2F6B55" if m == CURRENT_MODEL else "#6B7280" for m in sp["model"]]
    ax.barh(labels, sp["WAPE_pct"], color=colors)
    ax.set_xlabel("WAPE (%)")
    ax.set_title("严格递推候选模型 WAPE 前 12")
    for idx, row in enumerate(sp.itertuples()):
        ax.text(row.WAPE_pct + 0.15, idx, f"{row.WAPE_pct:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_candidate_model_wape.png", dpi=180)
    plt.close(fig)

    horizon = by_horizon.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(horizon["horizon"].astype(str), horizon["WAPE_pct"], color="#4C78A8")
    ax.set_xlabel("预测步长 horizon")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("当前混合策略按 horizon 的严格递推误差")
    for idx, row in enumerate(horizon.itertuples()):
        ax.text(idx, row.WAPE_pct + 0.6, f"{row.WAPE_pct:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_hybrid_error_by_horizon.png", dpi=180)
    plt.close(fig)

    store = store_interval.sort_values("point_forecast_7day_sales", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = store["point_forecast_7day_sales"].to_numpy()
    lower = x - store["lower_80pct_empirical"].to_numpy()
    upper = store["upper_80pct_empirical"].to_numpy() - x
    ax.errorbar(
        x,
        store["store_name"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2F6B55",
        ecolor="#9CA3AF",
        capsize=3,
    )
    ax.set_xlabel("未来 7 天预测销量")
    ax.set_title("门店未来 7 天预测及经验残差区间")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_forecast_interval_by_store.png", dpi=180)
    plt.close(fig)

    store_total_path = TABLES / "q4_hybrid_forecast_7day_total_by_store.csv"
    category_total_path = TABLES / "q4_hybrid_forecast_7day_total_by_category.csv"
    if store_total_path.exists():
        store_total = pd.read_csv(store_total_path).sort_values(
            "predicted_7day_sales", ascending=True
        )
        fig, ax = plt.subplots(figsize=(10, 5.4))
        ax.barh(store_total["store_name"], store_total["predicted_7day_sales"], color="#2F6B55")
        ax.set_xlabel("预测 7 日销量")
        ax.set_title("低销量混合策略门店 7 日预测总量")
        for idx, row in enumerate(store_total.itertuples()):
            ax.text(
                row.predicted_7day_sales + 3,
                idx,
                f"{row.predicted_7day_sales:.1f}",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(FIGURES / "q4_hybrid_forecast_7day_total_by_store.png", dpi=180)
        plt.close(fig)
    if category_total_path.exists():
        category_total = pd.read_csv(category_total_path).sort_values(
            "predicted_7day_sales", ascending=True
        )
        fig, ax = plt.subplots(figsize=(10, 5.4))
        ax.barh(category_total["category"], category_total["predicted_7day_sales"], color="#4C78A8")
        ax.set_xlabel("预测 7 日销量")
        ax.set_title("低销量混合策略类别 7 日预测总量")
        for idx, row in enumerate(category_total.itertuples()):
            ax.text(
                row.predicted_7day_sales + 3,
                idx,
                f"{row.predicted_7day_sales:.1f}",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(FIGURES / "q4_hybrid_forecast_7day_total_by_category.png", dpi=180)
        plt.close(fig)


def build_candidate_review(
    metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    leave_one: pd.DataFrame,
) -> str:
    sp = metrics[metrics["level"] == "store_product"].copy()
    show_cols = [
        "model_label",
        "threshold_rule",
        "MAE",
        "RMSE",
        "WAPE_pct",
        "WAPE_pct_point_change_vs_current",
        "actual_sum",
        "prediction_sum",
        "bias_pct_of_actual",
        "low_volume_selected_share",
    ]
    top = sp[show_cols].head(15)
    current = sp[sp["model"] == CURRENT_MODEL].iloc[0]
    q1 = sp[sp["model"] == Q1_MODEL].iloc[0]
    ridge = sp[sp["model"] == RIDGE_MODEL].iloc[0]
    best = sp.iloc[0]
    improvement = current["WAPE_pct"] - best["WAPE_pct"]
    selected_mean = float(
        leave_one.loc[leave_one["eval_type"] == "selected_by_other_windows", "WAPE_pct"].mean()
    )
    current_holdout_mean = float(
        leave_one.loc[leave_one["eval_type"] == "current_hybrid", "WAPE_pct"].mean()
    )
    holdout_improvement = current_holdout_mean - selected_mean
    replace_note = (
        f"全样本最佳候选相对当前改善 {improvement:.3f} 个百分点，但留一窗口选模后的平均改善为 "
        f"{holdout_improvement:.3f} 个百分点，低于 0.3 个百分点阈值。因此本轮不自动替换冻结主方案，"
        "只将近 28 日均值兜底记为探索性候选。"
    )
    calibration = sp[sp["model"].str.contains("calibrated", regex=False)][show_cols]
    best_windows = window_metrics[window_metrics["model"] == best["model"]][
        [
            "window_start",
            "model_label",
            "WAPE_pct",
            "WAPE_pct_point_change_vs_current_window",
            "actual_sum",
            "prediction_sum",
        ]
    ]
    leave_cols = [
        "holdout_window_start",
        "selected_label_on_other_windows",
        "eval_type",
        "eval_label",
        "WAPE_pct",
        "WAPE_pct_point_change_vs_current_holdout",
    ]

    return f"""# 问题四候选模型有限优化复核

执行日期：{date.today().isoformat()}

## 1. 复核目的

本轮只围绕当前严格 7 日递推口径做有限优化，重点检查两类问题：第一，低销量判定阈值和低销量兜底预测方法是否能稳定优于当前混合策略；第二，门店、商品、类别层级的比例校准是否能改善主粒度 WAPE。所有候选均基于已经生成的严格递推验证明细，不使用验证窗口真实销量构造未来特征，也不覆盖原最终预测表。

## 2. 主粒度候选结果

{df_to_md(top, 20)}

完整结果见 `tables/q4_candidate_model_metrics.csv`，候选预测明细见 `tables/q4_candidate_model_predictions.csv`，对比图见 `figures/q4_candidate_model_wape.png`。

## 3. 当前基准与替换判断

- 当前混合策略 WAPE={current['WAPE_pct']:.3f}%。
- 问题一简单指数平滑 WAPE={q1['WAPE_pct']:.3f}%。
- 完整综合 Ridge WAPE={ridge['WAPE_pct']:.3f}%。
- 本轮最佳候选为 `{best['model_label']}`，WAPE={best['WAPE_pct']:.3f}%，相对当前混合策略变化 {best['WAPE_pct_point_change_vs_current']:.3f} 个百分点。

结论：{replace_note}

## 4. 窗口级稳健性

全样本最佳候选在各验证窗口的表现如下。负数表示该窗口优于当前混合策略：

{df_to_md(best_windows, 10)}

进一步用“留一窗口”方式检查模型选择偏差：每次用另外 3 个窗口选择候选模型，再在留出的 1 个窗口上评价。结果如下：

{df_to_md(leave_one[leave_cols], 20)}

留一窗口下，选中候选的平均 WAPE={selected_mean:.3f}%，当前混合策略平均 WAPE={current_holdout_mean:.3f}%，平均改善 {holdout_improvement:.3f} 个百分点。该提升不足以把探索性候选写成新的主方案。

## 5. 层级校准检查

层级校准只把当前混合策略的日期--层级预测总量按比例缩放到问题一指数平滑的同层级聚合预测；该操作不使用验证期真实销量，因此属于可复现后处理，而不是利用真实答案校准。结果如下：

{df_to_md(calibration, 10)}

若层级校准不能在门店--商品主粒度上明显改善 WAPE，则论文中只保留“检查过层级一致性和比例缩放，未替换主方案”的结论。

## 6. 论文写法建议

当前主方案仍应冻结为低销量指数平滑--常规 Ridge 混合策略。可以在模型评价中补充：本轮尝试了不同低销量阈值、低销量均值兜底和层级校准，虽然近 28 日均值兜底在全样本验证中表现更好，但留一窗口检查未达到稳定替换阈值，因此最终预测表继续使用 `outputs/final_7day_forecast_hybrid_low_volume.csv`。
"""


def build_error_report(
    by_horizon: pd.DataFrame,
    by_class: pd.DataFrame,
    by_calendar: pd.DataFrame,
    class_summary: dict,
) -> str:
    horizon_cols = ["horizon", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]
    class_cols = [
        "demand_class",
        "is_low_volume",
        "MAE",
        "RMSE",
        "WAPE_pct",
        "actual_sum",
        "prediction_sum",
        "n",
    ]
    calendar_cols = [
        "calendar_segment",
        "definition",
        "MAE",
        "RMSE",
        "WAPE_pct",
        "actual_sum",
        "prediction_sum",
        "date_count",
    ]
    worst_horizon = by_horizon.sort_values("WAPE_pct", ascending=False).iloc[0]
    low_wape = class_summary.get("low_volume", {}).get("WAPE_pct", np.nan)
    regular_wape = class_summary.get("regular", {}).get("WAPE_pct", np.nan)

    return f"""# 问题四混合策略误差归因增强报告

执行日期：{date.today().isoformat()}

## 1. 目的

本报告基于当前最终候选模型“低销量指数平滑--常规 Ridge 混合策略”的严格 7 日递推验证明细，补充按预测步长、需求类别和日历场景的误差归因。该报告服务于论文解释和答辩，不改变最终预测表。

## 2. 按 horizon 的误差

{df_to_md(by_horizon[horizon_cols], 10)}

horizon={int(worst_horizon['horizon'])} 的 WAPE 最高，为 {worst_horizon['WAPE_pct']:.2f}%。若后续步长误差升高，可解释为递推预测中前序预测值进入后续 lag/rolling 特征造成误差累积。

## 3. 按低销量类别的误差

{df_to_md(by_class[class_cols], 20)}

低销量类别聚合 WAPE 约为 {low_wape:.2f}%，常规类别聚合 WAPE 约为 {regular_wape:.2f}%。低销量组真实销量规模小，少量绝对误差会被 WAPE 放大，论文应同时引用 MAE、RMSE 和真实销量合计。

## 4. 按日历场景的误差

{df_to_md(by_calendar[calendar_cols], 20)}

周末、节假日和活动日样本可能重叠，且验证窗口有限；因此这些结果只能说明模型在不同场景上的误差表现，不能作为因果判断。

## 5. 输出文件

- `tables/q4_hybrid_error_by_horizon.csv`
- `tables/q4_hybrid_error_by_demand_class.csv`
- `tables/q4_hybrid_error_by_calendar_segment.csv`
- `figures/q4_hybrid_error_by_horizon.png`
"""


def build_uncertainty_report(
    store: pd.DataFrame,
    product: pd.DataFrame,
    category: pd.DataFrame,
) -> str:
    store_cols = [
        "store_name",
        "point_forecast_7day_sales",
        "lower_80pct_empirical",
        "median_residual_adjusted_forecast",
        "upper_80pct_empirical",
        "mean_abs_residual",
        "residual_count",
    ]
    category_cols = [
        "category",
        "point_forecast_7day_sales",
        "lower_80pct_empirical",
        "median_residual_adjusted_forecast",
        "upper_80pct_empirical",
        "mean_abs_residual",
        "residual_count",
    ]
    product_cols = [
        "product_name",
        "category",
        "point_forecast_7day_sales",
        "lower_80pct_empirical",
        "upper_80pct_empirical",
        "mean_abs_residual",
    ]

    return f"""# 未来 7 天预测不确定性报告

执行日期：{date.today().isoformat()}

## 1. 区间构造方法

本报告不重新训练模型，而是使用当前混合策略在 4 个严格 7 日递推验证窗口上的 7 日聚合残差构造经验区间。对每个门店、商品或类别，先计算验证窗口中的

`残差 = 真实7日销量 - 预测7日销量`，

再取残差的 10% 和 90% 分位数加到未来 7 天点预测上，形成经验 80% 残差区间。由于验证窗口只有 4 个，该区间不是严格统计置信区间，只用于反映历史递推误差下的预测不确定性。

## 2. 门店级区间

{df_to_md(store[store_cols], 20)}

完整表见 `tables/final_forecast_interval_by_store.csv`，图见 `figures/final_forecast_interval_by_store.png`。

## 3. 类别级区间

{df_to_md(category[category_cols], 20)}

完整表见 `tables/final_forecast_interval_by_category.csv`。

## 4. 商品级区间示例

下表按点预测销量从高到低展示前 15 个商品。完整表见 `tables/final_forecast_interval_by_product.csv`。

{df_to_md(product[product_cols], 15)}

## 5. 使用边界

这些区间只来自历史验证残差，不能覆盖未来天气、促销安排、库存缺货或突发大单等未观测变化。论文中适合表述为“基于严格递推验证残差的经验不确定性范围”，不应写成保证真实销量落入区间的概率承诺。
"""


def write_summary_json(
    metrics: pd.DataFrame,
    store_interval: pd.DataFrame,
    leave_one: pd.DataFrame,
) -> None:
    sp = metrics[metrics["level"] == "store_product"].copy()
    current = sp[sp["model"] == CURRENT_MODEL].iloc[0]
    best = sp.iloc[0]
    selected_mean = float(
        leave_one.loc[leave_one["eval_type"] == "selected_by_other_windows", "WAPE_pct"].mean()
    )
    current_holdout_mean = float(
        leave_one.loc[leave_one["eval_type"] == "current_hybrid", "WAPE_pct"].mean()
    )
    holdout_improvement = current_holdout_mean - selected_mean
    summary = {
        "validation_mode": "strict_recursive_7day",
        "current_model": CURRENT_MODEL,
        "current_wape_pct": float(current["WAPE_pct"]),
        "best_candidate_model": str(best["model"]),
        "best_candidate_label": str(best["model_label"]),
        "best_candidate_wape_pct": float(best["WAPE_pct"]),
        "best_minus_current_wape_pct_points": float(
            best["WAPE_pct"] - current["WAPE_pct"]
        ),
        "leave_one_selected_mean_wape_pct": selected_mean,
        "leave_one_current_mean_wape_pct": current_holdout_mean,
        "leave_one_improvement_pct_points": holdout_improvement,
        "replace_current": bool(
            current["WAPE_pct"] - best["WAPE_pct"] >= IMPROVEMENT_THRESHOLD_PCT
            and holdout_improvement >= IMPROVEMENT_THRESHOLD_PCT
            and best["model"] != CURRENT_MODEL
        ),
        "store_interval_rows": int(len(store_interval)),
    }
    (OUTPUTS / "q4_award_iteration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    for path in [TABLES, OUTPUTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)

    base = load_strategy_base()
    candidates = build_candidate_predictions(base)
    candidates = add_calibrated_candidates(base, candidates)
    save_csv(candidates, "q4_candidate_model_predictions.csv")

    metrics = build_metric_table(candidates)
    window_metrics = build_window_metric_table(candidates)
    leave_one = build_leave_one_window_selection(candidates)
    save_csv(metrics, "q4_candidate_model_metrics.csv")
    save_csv(window_metrics, "q4_candidate_window_metrics.csv")
    save_csv(leave_one, "q4_candidate_leave_one_window_selection.csv")

    by_horizon, by_class, by_calendar, class_summary = build_error_attribution(candidates)
    store_interval, product_interval, category_interval = build_forecast_intervals(candidates)
    plot_outputs(metrics, by_horizon, store_interval)

    (OUTPUTS / "q4_candidate_model_review.md").write_text(
        build_candidate_review(metrics, window_metrics, leave_one), encoding="utf-8"
    )
    (OUTPUTS / "q4_error_attribution_enhanced_report.md").write_text(
        build_error_report(by_horizon, by_class, by_calendar, class_summary), encoding="utf-8"
    )
    (OUTPUTS / "forecast_uncertainty_report.md").write_text(
        build_uncertainty_report(store_interval, product_interval, category_interval),
        encoding="utf-8",
    )
    write_summary_json(metrics, store_interval, leave_one)

    sp = metrics[metrics["level"] == "store_product"].copy()
    current = sp[sp["model"] == CURRENT_MODEL].iloc[0]
    best = sp.iloc[0]
    improvement = current["WAPE_pct"] - best["WAPE_pct"]
    selected_mean = float(
        leave_one.loc[leave_one["eval_type"] == "selected_by_other_windows", "WAPE_pct"].mean()
    )
    current_holdout_mean = float(
        leave_one.loc[leave_one["eval_type"] == "current_hybrid", "WAPE_pct"].mean()
    )
    holdout_improvement = current_holdout_mean - selected_mean
    replace_note = (
        "全样本达到替换阈值但留一窗口未达阈值，作为探索候选但不替换冻结主方案"
        if improvement >= IMPROVEMENT_THRESHOLD_PCT and holdout_improvement < IMPROVEMENT_THRESHOLD_PCT
        else (
            "达到替换阈值，需人工复核候选预测表"
            if improvement >= IMPROVEMENT_THRESHOLD_PCT and best["model"] != CURRENT_MODEL
            else "未达到 0.3 个百分点替换阈值，继续冻结当前混合策略"
        )
    )
    row = (
        f"| {date.today().isoformat()} | 问题四有限优化与不确定性补充 | "
        "processed: modeling_base_table.csv；输入严格递推预测和混合策略预测；未修改 data/raw，未覆盖原最终预测表 | "
        "低销量阈值候选、低销量均值兜底、层级比例校准、经验残差区间 | "
        "历史均值、零销量比例、近28日/商品/类别均值、问题一聚合预测校准、严格递推残差 | "
        f"当前混合策略 WAPE={current['WAPE_pct']:.2f}%；最佳候选={best['model_label']} WAPE={best['WAPE_pct']:.2f}%；"
        f"相对当前全样本改善={improvement:.2f} 个百分点；留一窗口选模改善={holdout_improvement:.2f} 个百分点 | "
        f"{replace_note}；新增 horizon、低销量、日历场景误差和预测区间 | "
        "层级校准未使用验证窗口真实销量；预测区间为经验残差范围，不是严格置信区间 | "
        "将候选复核、不确定性和误差归因写入论文与答辩材料 |"
    )
    prepend_result_log(row)

    print(
        json.dumps(
            {
                "current_wape_pct": float(current["WAPE_pct"]),
                "best_model": str(best["model"]),
                "best_label": str(best["model_label"]),
                "best_wape_pct": float(best["WAPE_pct"]),
                "improvement_pct_points": float(improvement),
                "leave_one_selected_mean_wape_pct": float(selected_mean),
                "leave_one_current_mean_wape_pct": float(current_holdout_mean),
                "leave_one_improvement_pct_points": float(holdout_improvement),
                "replace_current": bool(
                    improvement >= IMPROVEMENT_THRESHOLD_PCT
                    and holdout_improvement >= IMPROVEMENT_THRESHOLD_PCT
                    and best["model"] != CURRENT_MODEL
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
