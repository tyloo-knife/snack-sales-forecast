from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics, paired_error_tests
from src.features import (
    add_leakage_safe_sales_features,
    build_future_external_scenario,
    build_store_product_panel,
)
from src.models import (
    build_random_forest_forecaster,
    build_ridge_forecaster,
    clipped_predict,
    validate_univariate_models,
)

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
PAPER = ROOT / "paper"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")
VALIDATION_END = pd.Timestamp("2022-03-30")
FUTURE_DATES = pd.date_range("2022-04-01", "2022-04-07", freq="D")

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


def save_csv(df: pd.DataFrame, name: str, to_outputs: bool = False) -> None:
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    if to_outputs:
        df.to_csv(OUTPUTS / name, index=False, encoding="utf-8-sig")


def upsert_section(path: Path, heading: str, content: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    content = content.strip() + "\n"
    if heading in text:
        start = text.index(heading)
        next_idx = text.find("\n## ", start + len(heading))
        if next_idx == -1:
            new_text = text[:start].rstrip() + "\n\n" + content
        else:
            new_text = (
                text[:start].rstrip()
                + "\n\n"
                + content
                + "\n"
                + text[next_idx + 1 :].lstrip()
            )
    else:
        new_text = text.rstrip() + "\n\n" + content
    path.write_text(new_text, encoding="utf-8")


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_leakage_safe_sales_features(panel, TARGET)
    out["store_id_str"] = out["store_id"].astype(str)
    out["product_id_str"] = out["product_id"].astype(str)
    out["category"] = out["category"].astype(str)
    out["weather"] = out["weather"].astype(str).fillna("未知")
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def validation_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.to_datetime(
        ["2022-03-01", "2022-03-08", "2022-03-15", "2022-03-22", "2022-03-29"]
    )
    windows = []
    for start in starts:
        end = min(start + pd.Timedelta(days=6), VALIDATION_END)
        windows.append((start, end))
    return windows


def make_model(model_name: str):
    if model_name == "ridge":
        return build_ridge_forecaster(NUMERIC_FEATURES, CATEGORICAL_FEATURES, alpha=10.0)
    if model_name == "random_forest":
        return build_random_forest_forecaster(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    raise ValueError(model_name)


def rolling_validate_ml(feature_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for window_start, window_end in validation_windows():
        train = feature_df[
            (feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)
        ].copy()
        valid = feature_df[
            (feature_df["date"] >= window_start)
            & (feature_df["date"] <= window_end)
            & (feature_df["has_external_data"] == 1)
        ].copy()
        model = make_model(model_name)
        model.fit(train[FEATURES], train[TARGET])
        valid["prediction"] = clipped_predict(model, valid[FEATURES])
        valid["model"] = model_name
        valid["model_label"] = {
            "ridge": "综合Ridge回归",
            "random_forest": "综合随机森林",
        }[model_name]
        valid["window_start"] = window_start
        rows.append(
            valid[
                [
                    "date",
                    "store_id",
                    "store_name",
                    "product_id",
                    "product_name",
                    "category",
                    TARGET,
                    "prediction",
                    "model",
                    "model_label",
                    "window_start",
                ]
            ].rename(columns={TARGET: "actual"})
        )
    return pd.concat(rows, ignore_index=True)


def baseline_predictions(feature_df: pd.DataFrame) -> pd.DataFrame:
    valid = feature_df[
        (feature_df["date"] >= VALIDATION_START)
        & (feature_df["date"] <= VALIDATION_END)
        & (feature_df["has_external_data"] == 1)
    ].copy()
    preds = []
    for model, label, col in [
        ("baseline_moving_average_7", "Baseline移动平均(7日)", "rolling_mean_7"),
        ("baseline_rolling_mean_14", "Baseline滚动均值(14日)", "rolling_mean_14"),
    ]:
        out = valid[
            [
                "date",
                "store_id",
                "store_name",
                "product_id",
                "product_name",
                "category",
                TARGET,
                col,
            ]
        ].rename(columns={TARGET: "actual", col: "prediction"})
        out["prediction"] = out["prediction"].clip(lower=0)
        out["model"] = model
        out["model_label"] = label
        out["window_start"] = pd.NaT
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def q1_store_product_exp_smoothing(panel: pd.DataFrame) -> pd.DataFrame:
    series = (
        panel.groupby(
            ["date", "store_id", "store_name", "product_id", "product_name", "category"],
            as_index=False,
        )[TARGET]
        .sum()
        .sort_values(["store_id", "product_id", "date"])
    )
    val = validate_univariate_models(
        series,
        ["store_id", "store_name", "product_id", "product_name", "category"],
        TARGET,
        VALIDATION_START,
    )
    val = val[
        (pd.to_datetime(val["date"]) >= VALIDATION_START)
        & (pd.to_datetime(val["date"]) <= VALIDATION_END)
        & (val["model"] == "exp_smoothing")
    ].copy()
    val["model"] = "q1_store_product_exp_smoothing"
    val["model_label"] = "问题一门店-商品简单指数平滑"
    return val[
        [
            "date",
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "category",
            "actual",
            "prediction",
            "model",
            "model_label",
        ]
    ]


def metrics_from_predictions(preds: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for (model, label), g in preds.groupby(["model", "model_label"]):
        metrics = calculate_metrics(g["actual"], g["prediction"])
        rows.append(
            {
                "level": level,
                "model": model,
                "model_label": label,
                **metrics,
                "actual_sum": float(g["actual"].sum()),
                "prediction_sum": float(g["prediction"].sum()),
                "n": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["WAPE", "MAE", "RMSE"]).reset_index(drop=True)
    out["WAPE_pct"] = out["WAPE"] * 100
    return out


def previous_metric_from_prediction_file(
    path: Path,
    model: str,
    previous_model_label: str,
) -> dict:
    """Recalculate previous-stage metrics on the stage-5 validation dates."""
    preds = pd.read_csv(path)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[
        (preds["date"] >= VALIDATION_START)
        & (preds["date"] <= VALIDATION_END)
        & (preds["model"] == model)
    ].copy()
    metrics = calculate_metrics(preds["actual"], preds["prediction"])
    return {
        "previous_model": previous_model_label,
        "previous_WAPE": metrics["WAPE"],
        "previous_MAE": metrics["MAE"],
        "previous_RMSE": metrics["RMSE"],
        "previous_actual_sum": float(preds["actual"].sum()),
        "previous_prediction_sum": float(preds["prediction"].sum()),
        "previous_n": int(len(preds)),
    }


def aggregate_predictions(preds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        preds.groupby(["date", "model", "model_label", *group_cols], as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(["model", *group_cols, "date"])
    )


def recursive_future_forecast(
    model,
    panel: pd.DataFrame,
    future_external: pd.DataFrame,
) -> pd.DataFrame:
    combos = (
        panel[["store_id", "store_name", "product_id", "product_name", "category"]]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
    )
    history = {
        (row.store_id, row.product_id): panel[
            (panel["store_id"] == row.store_id) & (panel["product_id"] == row.product_id)
        ]
        .sort_values("date")[TARGET]
        .astype(float)
        .tolist()
        for row in combos.itertuples()
    }
    rows = []
    ext_lookup = future_external.set_index("date").to_dict("index")
    for date in FUTURE_DATES:
        batch_rows = []
        ext = ext_lookup[pd.Timestamp(date)]
        for row in combos.itertuples():
            key = (row.store_id, row.product_id)
            vals = history[key]
            def lag(k: int) -> float:
                return float(vals[-k]) if len(vals) >= k else 0.0
            recent7 = vals[-7:] if vals else [0.0]
            recent14 = vals[-14:] if vals else [0.0]
            batch_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "store_id": row.store_id,
                    "store_name": row.store_name,
                    "product_id": row.product_id,
                    "product_name": row.product_name,
                    "category": row.category,
                    "lag_1": lag(1),
                    "lag_7": lag(7),
                    "lag_14": lag(14),
                    "rolling_mean_7": float(np.mean(recent7)),
                    "rolling_mean_14": float(np.mean(recent14)),
                    "rolling_std_7": float(np.std(recent7, ddof=1)) if len(recent7) >= 2 else 0.0,
                    "weekday": ext["weekday"],
                    "month": ext["month"],
                    "is_weekend": ext["is_weekend"],
                    "max_temperature": ext["max_temperature"],
                    "min_temperature": ext["min_temperature"],
                    "wind_power": ext["wind_power"],
                    "is_holiday": ext["is_holiday"],
                    "is_activity_day": ext["is_activity_day"],
                    "weather": ext["weather"],
                    "store_id_str": str(row.store_id),
                    "product_id_str": str(row.product_id),
                    "external_scenario_note": ext["external_scenario_note"],
                }
            )
        batch = pd.DataFrame(batch_rows)
        preds = clipped_predict(model, batch[FEATURES])
        batch["prediction"] = preds
        for pred_row in batch.itertuples():
            history[(pred_row.store_id, pred_row.product_id)].append(float(pred_row.prediction))
        rows.append(batch)
    return pd.concat(rows, ignore_index=True)


def paired_tests_for_models(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    label: str,
    group_cols: list[str] | None = None,
) -> dict:
    if group_cols:
        base = aggregate_predictions(baseline, group_cols)
        cand = aggregate_predictions(candidate, group_cols)
    else:
        base = baseline.copy()
        cand = candidate.copy()
    keys = ["date"] + (group_cols or [])
    merged = base[keys + ["actual", "prediction"]].merge(
        cand[keys + ["actual", "prediction"]],
        on=keys,
        suffixes=("_base", "_cand"),
    )
    daily = (
        merged.assign(
            abs_error_base=lambda x: (x["actual_base"] - x["prediction_base"]).abs(),
            abs_error_cand=lambda x: (x["actual_cand"] - x["prediction_cand"]).abs(),
        )
        .groupby("date", as_index=False)[["abs_error_base", "abs_error_cand"]]
        .sum()
    )
    return paired_error_tests(daily["abs_error_base"], daily["abs_error_cand"], label)


def build_notebook() -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 07 综合预测模型\n",
                    "\n",
                    "本 Notebook 对应阶段 5：问题四综合预测模型。目标是读取阶段 5 已生成的可复现结果，检查综合模型、误差比较和最终预测表。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 1：读取综合模型验证结果\n",
                    "\n",
                    "这段代码解决“综合模型是否比 baseline 更好”的问题。验证集使用 2022-03-01 至 2022-03-30，因为 2022-03-31 缺少外部变量。",
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
                    "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "metrics = pd.read_csv(ROOT / 'tables/q4_store_product_model_metrics.csv')\n",
                    "metrics[['model_label','MAE','RMSE','WAPE_pct']]\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：WAPE 越小，模型总体绝对误差占真实销量比例越低。综合模型必须和简单 baseline 放在一起比较。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 2：查看与问题一、问题二的比较\n",
                    "\n",
                    "这段代码解决“问题四是否相对前面模型有改进”的问题。不同粒度的误差需要在同一验证日期范围内比较。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comparison = pd.read_csv(ROOT / 'tables/q4_comparison_with_previous_models.csv')\n",
                    "comparison\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：`relative_wape_improvement_pct` 为相对 WAPE 改进比例。正值表示综合模型误差更低，负值表示没有改进。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 3：查看显著性检验\n",
                    "\n",
                    "这段代码解决“误差下降是否可能只是偶然”的问题。检验只说明验证误差差异，不证明模型在所有未来日期一定更好。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "tests = pd.read_csv(ROOT / 'tables/q4_significance_tests.csv')\n",
                    "tests\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：p 值较小时，说明候选模型在该验证误差序列上低于基准模型的证据更强；若 p 值不小，则不能写“显著改进”。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 4：读取未来 7 天最终预测\n",
                    "\n",
                    "这段代码解决“最终提交的门店-商品预测结果是什么”的问题。未来天气和活动日为历史同期参考情景，不是附件观测值。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "forecast = pd.read_csv(ROOT / 'outputs/final_7day_forecast.csv')\n",
                    "forecast.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：每行是某天、某门店、某商品的预测销量。7 天汇总表见 `q4_store_product_forecast_7day_total.csv`。",
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
    (NOTEBOOKS / "07_final_prediction.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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


def main() -> None:
    for path in [TABLES, FIGURES, OUTPUTS, NOTEBOOKS, PAPER]:
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140

    raw = pd.read_csv(ROOT / "data/processed/modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    feature_df = prepare_model_frame(panel)
    feature_df["has_external_data"] = feature_df["has_external_data"].fillna(0).astype(int)
    save_csv(feature_df, "q4_modeling_feature_table.csv")

    future_external = build_future_external_scenario(raw, FUTURE_DATES)
    save_csv(future_external, "q4_future_external_scenario.csv", True)

    base_preds = baseline_predictions(feature_df)
    q1_preds = q1_store_product_exp_smoothing(panel)
    ridge_preds = rolling_validate_ml(feature_df, "ridge")
    rf_preds = rolling_validate_ml(feature_df, "random_forest")
    validation_preds = pd.concat([q1_preds, base_preds, ridge_preds, rf_preds], ignore_index=True)
    validation_preds["abs_error"] = (validation_preds["actual"] - validation_preds["prediction"]).abs()
    save_csv(validation_preds, "q4_validation_predictions_store_product.csv", True)

    store_product_metrics = metrics_from_predictions(validation_preds, "store_product")
    save_csv(store_product_metrics, "q4_store_product_model_metrics.csv", True)

    comprehensive_candidates = store_product_metrics[
        store_product_metrics["model"].isin(["ridge", "random_forest"])
    ].sort_values("WAPE")
    best_comp_model = comprehensive_candidates.iloc[0]["model"]
    best_comp_label = comprehensive_candidates.iloc[0]["model_label"]
    best_comp_preds = validation_preds[validation_preds["model"] == best_comp_model].copy()

    # Aggregated metrics for comparison with Q2.
    store_category_preds = aggregate_predictions(
        validation_preds, ["store_id", "store_name", "category"]
    )
    category_preds = aggregate_predictions(validation_preds, ["category"])
    store_category_metrics = metrics_from_predictions(store_category_preds, "store_category")
    category_metrics = metrics_from_predictions(category_preds, "category")
    save_csv(store_category_metrics, "q4_store_category_model_metrics.csv", True)
    save_csv(category_metrics, "q4_category_model_metrics.csv", True)

    comparison_rows = []
    q1_row = store_product_metrics[
        store_product_metrics["model"] == "q1_store_product_exp_smoothing"
    ].iloc[0]
    q4_row = store_product_metrics[store_product_metrics["model"] == best_comp_model].iloc[0]
    comparison_rows.append(
        {
            "level": "store_product",
            "previous_model": q1_row["model_label"],
            "previous_WAPE": q1_row["WAPE"],
            "q4_model": q4_row["model_label"],
            "q4_WAPE": q4_row["WAPE"],
            "relative_wape_improvement_pct": (q1_row["WAPE"] - q4_row["WAPE"]) / q1_row["WAPE"] * 100,
        }
    )

    q2_store_cat = previous_metric_from_prediction_file(
        ROOT / "tables/q2_store_product_sum_store_category_validation_predictions.csv",
        "exp_smoothing",
        "问题二-门店-单品预测后按门店类别加总-简单指数平滑",
    )
    q4_store_cat = store_category_metrics[store_category_metrics["model"] == best_comp_model].iloc[0]
    comparison_rows.append(
        {
            "level": "store_category",
            "previous_model": q2_store_cat["previous_model"],
            "previous_WAPE": float(q2_store_cat["previous_WAPE"]),
            "q4_model": q4_store_cat["model_label"],
            "q4_WAPE": q4_store_cat["WAPE"],
            "relative_wape_improvement_pct": (float(q2_store_cat["previous_WAPE"]) - q4_store_cat["WAPE"]) / float(q2_store_cat["previous_WAPE"]) * 100,
        }
    )

    q2_cat = previous_metric_from_prediction_file(
        ROOT / "tables/q2_category_direct_validation_predictions.csv",
        "exp_smoothing",
        "问题二-类别聚合后直接预测-简单指数平滑",
    )
    q4_cat = category_metrics[category_metrics["model"] == best_comp_model].iloc[0]
    comparison_rows.append(
        {
            "level": "category",
            "previous_model": q2_cat["previous_model"],
            "previous_WAPE": float(q2_cat["previous_WAPE"]),
            "q4_model": q4_cat["model_label"],
            "q4_WAPE": q4_cat["WAPE"],
            "relative_wape_improvement_pct": (float(q2_cat["previous_WAPE"]) - q4_cat["WAPE"]) / float(q2_cat["previous_WAPE"]) * 100,
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, "q4_comparison_with_previous_models.csv", True)

    q1_best = validation_preds[validation_preds["model"] == "q1_store_product_exp_smoothing"]
    tests = [
        paired_tests_for_models(q1_best, best_comp_preds, f"{best_comp_label} vs 问题一门店-商品指数平滑"),
    ]
    baseline7 = validation_preds[validation_preds["model"] == "baseline_moving_average_7"]
    tests.append(paired_tests_for_models(baseline7, best_comp_preds, f"{best_comp_label} vs 7日移动平均baseline"))
    q2_store_cat_path = ROOT / "tables/q2_store_product_sum_store_category_validation_predictions.csv"
    if q2_store_cat_path.exists():
        q2_store_cat_preds = pd.read_csv(q2_store_cat_path)
        q2_store_cat_preds["date"] = pd.to_datetime(q2_store_cat_preds["date"])
        q2_store_cat_preds = q2_store_cat_preds[
            (q2_store_cat_preds["date"] >= VALIDATION_START)
            & (q2_store_cat_preds["date"] <= VALIDATION_END)
            & (q2_store_cat_preds["model"] == "exp_smoothing")
        ].copy()
        q2_store_cat_preds["model"] = "q2_store_category_best"
        q2_store_cat_preds["model_label"] = "问题二门店类别最佳"
        tests.append(
            paired_tests_for_models(
                q2_store_cat_preds,
                best_comp_preds,
                f"{best_comp_label}聚合到门店类别 vs 问题二门店类别最佳",
                ["store_id", "category"],
            )
        )
    q2_cat_path = ROOT / "tables/q2_category_direct_validation_predictions.csv"
    if q2_cat_path.exists():
        q2_cat_preds = pd.read_csv(q2_cat_path)
        q2_cat_preds["date"] = pd.to_datetime(q2_cat_preds["date"])
        q2_cat_preds = q2_cat_preds[
            (q2_cat_preds["date"] >= VALIDATION_START)
            & (q2_cat_preds["date"] <= VALIDATION_END)
            & (q2_cat_preds["model"] == "exp_smoothing")
        ].copy()
        q2_cat_preds["model"] = "q2_category_best"
        q2_cat_preds["model_label"] = "问题二类别最佳"
        tests.append(
            paired_tests_for_models(
                q2_cat_preds,
                best_comp_preds,
                f"{best_comp_label}聚合到类别 vs 问题二类别最佳",
                ["category"],
            )
        )
    tests_df = pd.DataFrame(tests)
    save_csv(tests_df, "q4_significance_tests.csv", True)

    # Train final comprehensive model and recursively forecast future 7 days.
    final_train = feature_df[
        (feature_df["date"] <= VALIDATION_END) & (feature_df["has_external_data"] == 1)
    ].copy()
    final_model = make_model(best_comp_model)
    final_model.fit(final_train[FEATURES], final_train[TARGET])
    future_daily = recursive_future_forecast(final_model, panel, future_external)
    future_daily["model"] = best_comp_model
    future_daily["model_label"] = best_comp_label
    future_daily = future_daily.rename(columns={"prediction": "predicted_sales"})
    future_daily_out = future_daily[
        [
            "date",
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "category",
            "weather",
            "is_holiday",
            "is_weekend",
            "is_activity_day",
            "predicted_sales",
            "model",
            "model_label",
            "external_scenario_note",
        ]
    ].copy()
    save_csv(future_daily_out, "q4_store_product_forecast_daily.csv")
    future_daily_out.to_csv(OUTPUTS / "final_7day_forecast.csv", index=False, encoding="utf-8-sig")

    store_product_total = (
        future_daily_out.groupby(
            ["store_id", "store_name", "product_id", "product_name", "category", "model", "model_label"],
            as_index=False,
        )["predicted_sales"]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values(["store_id", "predicted_7day_sales"], ascending=[True, False])
    )
    save_csv(store_product_total, "q4_store_product_forecast_7day_total.csv", True)
    store_total = (
        future_daily_out.groupby(["store_id", "store_name", "model", "model_label"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values("predicted_7day_sales", ascending=False)
    )
    product_total = (
        future_daily_out.groupby(["product_id", "product_name", "category", "model", "model_label"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values("predicted_7day_sales", ascending=False)
    )
    category_total = (
        future_daily_out.groupby(["category", "model", "model_label"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values("predicted_7day_sales", ascending=False)
    )
    save_csv(store_total, "q4_store_forecast_7day_total.csv", True)
    save_csv(product_total, "q4_product_forecast_7day_total.csv", True)
    save_csv(category_total, "q4_category_forecast_7day_total.csv", True)

    # Figures.
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_metrics = store_product_metrics.sort_values("WAPE", ascending=True)
    ax.barh(plot_metrics["model_label"], plot_metrics["WAPE_pct"], color="#4477AA")
    ax.set_title("图20 门店-商品粒度验证 WAPE 比较")
    ax.set_xlabel("WAPE (%)")
    for i, row in enumerate(plot_metrics.itertuples()):
        ax.text(row.WAPE_pct, i, f"{row.WAPE_pct:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_store_product_model_wape_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_comp = comparison.copy()
    labels = plot_comp["level"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, plot_comp["previous_WAPE"] * 100, width, label="前序模型", color="#88CCEE")
    ax.bar(x + width / 2, plot_comp["q4_WAPE"] * 100, width, label="综合模型", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("图21 综合模型与问题一/二模型 WAPE 比较")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_previous_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    store_total_plot = store_total.sort_values("predicted_7day_sales", ascending=True)
    ax.barh(store_total_plot["store_name"], store_total_plot["predicted_7day_sales"], color="#66AA55")
    ax.set_title("图22 各门店未来 7 天综合预测总销量")
    ax.set_xlabel("预测 7 日总销量")
    for i, row in enumerate(store_total_plot.itertuples()):
        ax.text(row.predicted_7day_sales, i, f"{row.predicted_7day_sales:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_store_forecast_7day_total.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    cat_plot = category_total.sort_values("predicted_7day_sales", ascending=True)
    ax.barh(cat_plot["category"], cat_plot["predicted_7day_sales"], color="#AA4499")
    ax.set_title("图23 各类别未来 7 天综合预测总销量")
    ax.set_xlabel("预测 7 日总销量")
    for i, row in enumerate(cat_plot.itertuples()):
        ax.text(row.predicted_7day_sales, i, f"{row.predicted_7day_sales:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_category_forecast_7day_total.png", bbox_inches="tight")
    plt.close(fig)

    # Reports and paper.
    feature_explain = pd.DataFrame(
        [
            ("lag_1", "前 1 天同门店同商品销量", "历史销量滞后项"),
            ("lag_7", "前 7 天同门店同商品销量", "同星期历史需求"),
            ("lag_14", "前 14 天同门店同商品销量", "两周前需求"),
            ("rolling_mean_7", "前 7 天滚动均值", "短期销量水平"),
            ("rolling_mean_14", "前 14 天滚动均值", "较平滑销量水平"),
            ("weekday/month/is_weekend", "星期、月份、周末", "日历规律"),
            ("store/product/category", "门店、商品、类别", "固定需求差异"),
            ("weather/temperature/wind", "天气、温度、风力", "外部环境情景"),
            ("is_holiday/is_activity_day", "节假日、活动日", "日历与促销情景"),
        ],
        columns=["feature", "meaning", "reason"],
    )
    save_csv(feature_explain, "q4_feature_dictionary.csv", True)

    comparison_report = comparison.copy()
    comparison_report["previous_WAPE_pct"] = comparison_report["previous_WAPE"] * 100
    comparison_report["q4_WAPE_pct"] = comparison_report["q4_WAPE"] * 100
    tests_report = tests_df.copy()

    improvement_store_product = float(comparison.loc[comparison["level"] == "store_product", "relative_wape_improvement_pct"].iloc[0])
    significant_note = "未达到显著改进" if tests_df["t_p_value_less"].iloc[0] >= 0.05 else "达到配对t检验意义上的显著改进"

    report = f"""# 阶段 5：问题四综合预测模型报告

执行日期：2026-05-02

## 1. 问题四核心任务

问题四要求结合前三问模型和结论，预测各门店各种零食未来 7 天总销量，并与问题一、问题二模型比较误差是否有显著改进。本阶段以 `store_id + product_id` 作为门店-商品粒度，继续使用 `positive_sales` 作为顾客正向需求销量口径。

## 2. 什么是综合预测模型

综合预测模型是把多类信息放进同一个模型：历史销量滞后项、滚动均值、星期月份、门店、商品、类别、天气、节假日和活动日。直觉上，它既保留问题一的时间序列规律，也吸收问题二的类别信息和问题三的外部因素统计关联。

## 3. 为什么要和问题一、问题二模型比较

复杂模型只有在同一验证集上明显优于简单模型时才有价值。问题一提供门店-商品层级的简单时间序列基准，问题二提供类别聚合基准；问题四必须与它们比较，才能判断“综合”是否真的带来误差下降。

## 4. 什么叫误差改进是否显著

本阶段先比较 MAE、RMSE、WAPE 的数值差异，再对同一验证日期上的绝对误差做配对 t 检验、Wilcoxon 检验和简化 Diebold-Mariano 检验。显著性检验只能说明验证误差差异是否较稳定，不能证明未来一定更好。

## 5. 综合建模数据与特征

特征设计如下：

{df_to_md(feature_explain, 20)}

未来 7 天天气没有附件真实观测。本阶段采用历史同月日天气、温度、风力和活动日的参考情景；2022-04-03 至 2022-04-05 按清明节假日处理。该处理会使外部变量预测具有情景假设性质，若真实天气或活动安排不同，预测结果会受到影响。

未来外部变量情景如下：

{df_to_md(future_external, 10)}

## 6. 模型选择

本阶段保留 7 日移动平均和 14 日滚动均值作为 baseline，同时建立 Ridge 回归和随机森林两个综合模型。没有使用 LightGBM/XGBoost，因为当前环境中对应包不可用；也没有使用 SARIMAX，因为本题需要同时处理大量门店-商品组合和多种分类变量，SARIMAX 不适合作为主要综合模型。

Ridge 回归适合解释线性加权关系，随机森林适合捕捉非线性和变量交互。本阶段不做复杂调参，只使用小规模参数设置，避免为了追求局部验证集表现而过拟合。

## 7. 滚动验证

验证区间为 2022-03-01 至 2022-03-30。2022-03-31 因附件二外部变量缺失，不纳入综合模型验证。验证按周设置滚动窗口：每个窗口只用窗口开始日前的数据训练，再预测该窗口内日期。时间序列不能随机切分，因为随机切分会让未来日期信息进入训练过程，造成信息泄露。

## 8. 门店-商品层级误差比较

{df_to_md(store_product_metrics[["model_label", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]], 20)}

综合模型中验证 WAPE 最低的是 `{best_comp_label}`。与问题一门店-商品简单指数平滑相比，门店-商品层级 WAPE 相对改进为 {improvement_store_product:.2f}%。

## 9. 与问题一、问题二模型比较

{df_to_md(comparison_report[["level", "previous_model", "previous_WAPE_pct", "q4_model", "q4_WAPE_pct", "relative_wape_improvement_pct"]], 10)}

## 10. 显著性检验

{df_to_md(tests_report, 10)}

解释：检验使用每日绝对误差总和作为配对序列。若 p 值小于 0.05，可写为验证误差下降具有统计证据；否则只能写“数值上下降/未能证明显著下降”。当前门店-商品层级结论为：{significant_note}。

## 11. 未来 7 天预测结果

各门店未来 7 天预测总销量：

{df_to_md(store_total[["store_id", "store_name", "model_label", "predicted_7day_sales"]], 20)}

各商品未来 7 天预测总销量：

{df_to_md(product_total[["product_id", "product_name", "category", "model_label", "predicted_7day_sales"]], 20)}

各类别未来 7 天预测总销量：

{df_to_md(category_total[["category", "model_label", "predicted_7day_sales"]], 20)}

完整门店-商品-日期预测表已保存至 `outputs/final_7day_forecast.csv`，门店-商品 7 天汇总表已保存至 `tables/q4_store_product_forecast_7day_total.csv`。

## 12. 结论

综合模型能够把前三问信息统一到门店-商品粒度，但是否优于简单模型必须看验证误差。若综合模型 WAPE 和显著性检验均优于前序模型，可写为“综合模型在验证集上表现出误差改进”；若未通过检验，则应写为“综合模型未能证明显著改进，简单时间序列模型仍具有竞争力”。本阶段不把外部变量写成因果影响，只把它们作为预测特征和统计关联信息使用。
"""
    (OUTPUTS / "stage5_q4_final_model_report.md").write_text(report, encoding="utf-8")

    q4_model_building = f"""## 问题四模型建立

问题四需要在门店-商品粒度上进行综合预测。设 $y_{{s,p,t}}$ 表示门店 $s$、商品 $p$ 在日期 $t$ 的正向销量，综合模型的特征包括历史销量滞后项、滚动均值、日历变量、门店、商品、类别和外部变量。

主要特征包括：

{df_to_md(feature_explain, 20)}

综合模型可以写为：

$$
\\hat y_{{s,p,t}}=f(Lag_{{s,p,t}},Roll_{{s,p,t}},Cal_t,Store_s,Product_p,Category_p,External_t)
$$

其中，$Lag$ 表示滞后销量，$Roll$ 表示滚动均值，$Cal$ 表示星期、月份和周末，$External$ 表示天气、节假日和活动日。本文保留移动平均 baseline，并建立 Ridge 回归和随机森林综合模型。Ridge 适合解释线性关系，随机森林用于捕捉非线性和交互关系。LightGBM/XGBoost 当前环境不可用，SARIMAX 不适合大量门店-商品组合和多分类外部变量，因此未作为主模型。
"""

    q4_model_solution = f"""## 问题四模型求解

验证区间取 2022-03-01 至 2022-03-30。由于 2022-03-31 缺少附件二外部变量，综合模型验证不使用该日期。验证采用滚动窗口：每个窗口只使用窗口开始日前的历史数据训练，预测该窗口内日期，从而避免随机切分带来的时间泄露。

门店-商品层级误差比较如下：

{df_to_md(store_product_metrics[["model_label", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]], 20)}

与前序模型比较如下：

{df_to_md(comparison_report[["level", "previous_model", "previous_WAPE_pct", "q4_model", "q4_WAPE_pct", "relative_wape_improvement_pct"]], 10)}

显著性检验如下：

{df_to_md(tests_report, 10)}

未来天气和活动日无附件观测，因此采用历史同月日参考情景；清明节日历按 2022-04-03 至 2022-04-05 为节假日处理。
"""

    q4_result_analysis = f"""## 问题四结果分析

问题四综合模型中，验证 WAPE 最低的综合模型为 `{best_comp_label}`。与问题一门店-商品简单指数平滑相比，门店-商品层级 WAPE 相对改进为 {improvement_store_product:.2f}%。显著性检验结论为：{significant_note}。

未来 7 天各门店预测总销量如下：

{df_to_md(store_total[["store_id", "store_name", "model_label", "predicted_7day_sales"]], 20)}

未来 7 天各类别预测总销量如下：

{df_to_md(category_total[["category", "model_label", "predicted_7day_sales"]], 20)}

需要注意，未来天气、温度、风力和活动日采用历史同期参考情景，不是附件真实观测。因此，最终预测结果是“给定该外部变量情景下”的销量预测。若真实天气或活动安排发生变化，预测结果也可能改变。

从模型比较看，若综合模型未通过显著性检验，则论文中不能写“综合模型显著优于前序模型”；只能写“在验证集上数值误差变化情况如下，并给出显著性检验结果”。本阶段仍不把天气、节假日和活动日解释为因果影响。
"""

    upsert_section(PAPER / "model_building.md", "## 问题四模型建立", q4_model_building)
    upsert_section(PAPER / "model_solution.md", "## 问题四模型求解", q4_model_solution)
    upsert_section(PAPER / "result_analysis.md", "## 问题四结果分析", q4_result_analysis)
    build_notebook()

    row = (
        f"| 2026-05-02 | 阶段 5 问题四综合预测模型 | processed: modeling_base_table.csv | "
        f"7日移动平均baseline、Ridge、RandomForest | 滞后销量、滚动均值、星期、月份、门店、商品、类别、天气、节假日、活动日 | "
        f"综合最优={best_comp_label}，WAPE={q4_row['WAPE']*100:.2f}%；相对问题一改进={improvement_store_product:.2f}% | "
        f"已完成门店-商品综合预测、与问题一/二误差比较、显著性检验和未来7天预测表 | "
        f"2022-03-31 缺少外部变量未纳入综合验证；未来天气/活动日为历史同期参考情景；显著性结论需按检验结果克制表述 | "
        f"停止在阶段 5，等待确认后进入阶段 6 完整论文写作 |"
    )
    prepend_result_log(row)

    summary = {
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "n_validation_predictions": int(len(validation_preds)),
        "best_comprehensive_model": str(best_comp_model),
        "best_comprehensive_label": str(best_comp_label),
        "best_comprehensive_wape": float(q4_row["WAPE"]),
        "q1_exp_smoothing_wape": float(q1_row["WAPE"]),
        "relative_wape_improvement_vs_q1_pct": float(improvement_store_product),
        "future_forecast_rows": int(len(future_daily_out)),
        "future_store_product_combinations": int(store_product_total[["store_id", "product_id"]].drop_duplicates().shape[0]),
        "future_total_sales": float(future_daily_out["predicted_sales"].sum()),
        "future_external_weather_observed": False,
    }
    (OUTPUTS / "stage5_q4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
