from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics
from src.features import build_store_product_panel

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

TARGET = "positive_sales"
BASE_PREDICTION_PATH = TABLES / "q4_ablation_predictions.csv"


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


def demand_class_from_history(values: pd.Series) -> str:
    values = pd.Series(values, dtype=float).fillna(0.0)
    if values.empty or (values > 0).sum() == 0:
        return "no_positive_history"
    avg = float(values.mean())
    zero_ratio = float((values <= 0).mean())
    nonzero_ratio = float((values > 0).mean())
    if nonzero_ratio <= 0.15 or avg < 0.20:
        return "very_sparse"
    if zero_ratio >= 0.75 or avg < 0.50:
        return "low_volume_intermittent"
    if zero_ratio >= 0.50:
        return "intermittent"
    if values.std(ddof=1) / avg >= 1.50:
        return "volatile"
    return "regular"


def is_low_volume_class(label: str) -> bool:
    return label in {"no_positive_history", "very_sparse", "low_volume_intermittent"}


def classification_at_window(panel: pd.DataFrame, window_start: pd.Timestamp) -> pd.DataFrame:
    history = panel[panel["date"] < window_start].copy()
    combo_cols = ["store_id", "store_name", "product_id", "product_name", "category"]
    combos = (
        panel[combo_cols]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )

    product_mean = history.groupby("product_id")[TARGET].mean().to_dict()
    category_mean = history.groupby("category")[TARGET].mean().to_dict()
    global_mean = float(history[TARGET].mean()) if not history.empty else 0.0

    rows = []
    for row in combos.itertuples():
        hist = history[
            (history["store_id"] == row.store_id)
            & (history["product_id"] == row.product_id)
        ].sort_values("date")
        values = hist[TARGET].astype(float)
        recent28 = values.tail(28)
        series_mean = float(values.mean()) if not values.empty else 0.0
        recent28_mean = float(recent28.mean()) if not recent28.empty else series_mean
        prod_mean = float(product_mean.get(row.product_id, global_mean))
        cat_mean = float(category_mean.get(row.category, global_mean))
        hierarchy_target = max(
            0.0,
            0.5 * recent28_mean + 0.3 * prod_mean + 0.2 * cat_mean,
        )
        label = demand_class_from_history(values)
        rows.append(
            {
                "window_start": pd.Timestamp(window_start),
                "store_id": int(row.store_id),
                "store_name": row.store_name,
                "product_id": int(row.product_id),
                "product_name": row.product_name,
                "category": row.category,
                "history_days": int(len(values)),
                "history_total_sales": float(values.sum()),
                "history_average_sales": series_mean,
                "history_zero_sales_ratio": float((values <= 0).mean()) if len(values) else 1.0,
                "history_nonzero_sales_ratio": float((values > 0).mean()) if len(values) else 0.0,
                "recent28_mean": recent28_mean,
                "product_history_mean": prod_mean,
                "category_history_mean": cat_mean,
                "hierarchy_shrink_target": hierarchy_target,
                "demand_class": label,
                "is_low_volume": int(is_low_volume_class(label)),
            }
        )
    return pd.DataFrame(rows)


def build_classification(panel: pd.DataFrame, window_starts: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.concat(
        [classification_at_window(panel, start) for start in window_starts],
        ignore_index=True,
    )


def read_base_predictions() -> pd.DataFrame:
    if not BASE_PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "缺少 q4_ablation_predictions.csv，请先运行 src/stage5_q4_ablation.py。"
        )
    preds = pd.read_csv(BASE_PREDICTION_PATH)
    preds["date"] = pd.to_datetime(preds["date"])
    preds["window_start"] = pd.to_datetime(preds["window_start"])
    required_models = {"q1_store_product_exp_smoothing", "ridge_full_external"}
    existing = set(preds["model"].unique())
    missing = sorted(required_models - existing)
    if missing:
        raise RuntimeError(f"q4_ablation_predictions.csv 缺少模型：{missing}")
    return preds


def pivot_base_predictions(preds: pd.DataFrame) -> pd.DataFrame:
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
    subset = preds[preds["model"].isin(["q1_store_product_exp_smoothing", "ridge_full_external"])]
    pivot = (
        subset.pivot_table(
            index=key_cols,
            columns="model",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivot.rename(
        columns={
            "q1_store_product_exp_smoothing": "pred_q1_exp_smoothing",
            "ridge_full_external": "pred_ridge_full",
        }
    )


def build_strategy_predictions(base: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(
        cls[
            [
                "window_start",
                "store_id",
                "product_id",
                "history_average_sales",
                "history_zero_sales_ratio",
                "recent28_mean",
                "hierarchy_shrink_target",
                "demand_class",
                "is_low_volume",
            ]
        ],
        on=["window_start", "store_id", "product_id"],
        how="left",
    )
    merged["is_low_volume"] = merged["is_low_volume"].fillna(0).astype(int)
    merged["hierarchy_shrink_target"] = merged["hierarchy_shrink_target"].fillna(0.0)
    merged["q1_low_volume_shrunk"] = np.where(
        merged["is_low_volume"].eq(1),
        0.5 * merged["pred_q1_exp_smoothing"] + 0.5 * merged["hierarchy_shrink_target"],
        merged["pred_q1_exp_smoothing"],
    )
    merged["ridge_low_volume_shrunk"] = np.where(
        merged["is_low_volume"].eq(1),
        0.5 * merged["pred_ridge_full"] + 0.5 * merged["hierarchy_shrink_target"],
        merged["pred_ridge_full"],
    )
    merged["hybrid_low_q1_regular_ridge"] = np.where(
        merged["is_low_volume"].eq(1),
        merged["pred_q1_exp_smoothing"],
        merged["pred_ridge_full"],
    )
    merged["hybrid_low_shrunk_regular_ridge"] = np.where(
        merged["is_low_volume"].eq(1),
        merged["q1_low_volume_shrunk"],
        merged["pred_ridge_full"],
    )

    strategy_specs = [
        (
            "q1_store_product_exp_smoothing",
            "问题一简单指数平滑",
            "pred_q1_exp_smoothing",
            "原问题一门店-商品指数平滑，作为严格递推强基准",
        ),
        (
            "ridge_full_external",
            "完整综合Ridge",
            "pred_ridge_full",
            "原问题四完整综合 Ridge，作为综合模型基准",
        ),
        (
            "q1_low_volume_shrunk",
            "低销量收缩指数平滑",
            "q1_low_volume_shrunk",
            "低销量序列用 50% 指数平滑 + 50% 历史层级均值收缩，其余保持指数平滑",
        ),
        (
            "ridge_low_volume_shrunk",
            "低销量收缩Ridge",
            "ridge_low_volume_shrunk",
            "低销量序列用 50% Ridge + 50% 历史层级均值收缩，其余保持 Ridge",
        ),
        (
            "hybrid_low_q1_regular_ridge",
            "低销量指数平滑-常规Ridge混合",
            "hybrid_low_q1_regular_ridge",
            "低销量序列使用指数平滑，非低销量序列使用完整 Ridge",
        ),
        (
            "hybrid_low_shrunk_regular_ridge",
            "低销量收缩-常规Ridge混合",
            "hybrid_low_shrunk_regular_ridge",
            "低销量序列使用收缩指数平滑，非低销量序列使用完整 Ridge",
        ),
    ]
    rows = []
    base_cols = [
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
        "recent28_mean",
        "hierarchy_shrink_target",
        "demand_class",
        "is_low_volume",
    ]
    for model, label, col, rule in strategy_specs:
        tmp = merged[base_cols].copy()
        tmp["prediction"] = np.maximum(0.0, merged[col].astype(float))
        tmp["model"] = model
        tmp["model_label"] = label
        tmp["strategy_rule"] = rule
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out["abs_error"] = (out["actual"] - out["prediction"]).abs()
    return out


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


def metric_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in ["store_product", "store", "product", "category"]:
        metric_input = aggregate_predictions(preds, level)
        for (model, label), group in metric_input.groupby(["model", "model_label"], dropna=False):
            metrics = calculate_metrics(group["actual"], group["prediction"])
            rule = preds.loc[preds["model"] == model, "strategy_rule"].dropna().iloc[0]
            rows.append(
                {
                    "validation_mode": "strict_recursive_7day",
                    "level": level,
                    "model": model,
                    "model_label": label,
                    "strategy_rule": rule,
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
    q1_wape = float(sp.loc["q1_store_product_exp_smoothing", "WAPE_pct"])
    ridge_wape = float(sp.loc["ridge_full_external", "WAPE_pct"])
    out["WAPE_pct_point_change_vs_q1"] = out["WAPE_pct"] - q1_wape
    out["WAPE_pct_point_change_vs_ridge"] = out["WAPE_pct"] - ridge_wape
    return out.sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)


def metrics_by_demand_class(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label, cls), group in preds.groupby(
        ["model", "model_label", "demand_class"], dropna=False
    ):
        metrics = calculate_metrics(group["actual"], group["prediction"])
        rows.append(
            {
                "validation_mode": "strict_recursive_7day",
                "model": model,
                "model_label": label,
                "demand_class": cls,
                "is_low_volume": int(group["is_low_volume"].max()),
                **metrics,
                "WAPE_pct": metrics["WAPE"] * 100,
                "actual_sum": float(group["actual"].sum()),
                "prediction_sum": float(group["prediction"].sum()),
                "n": int(len(group)),
                "series_count": int(
                    group[["window_start", "store_id", "product_id"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["demand_class", "WAPE", "model"]).reset_index(drop=True)


def bootstrap_wape_diff_by_window(
    preds: pd.DataFrame,
    baseline_model: str,
    candidate_model: str,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap WAPE difference in percentage points by resampling 7-day windows."""
    subset = preds[preds["model"].isin([baseline_model, candidate_model])].copy()
    window_rows = []
    for window_start, group in subset.groupby("window_start", dropna=False):
        base = group[group["model"] == baseline_model]
        cand = group[group["model"] == candidate_model]
        actual_sum = float(base["actual"].sum())
        base_abs = float((base["actual"] - base["prediction"]).abs().sum())
        cand_abs = float((cand["actual"] - cand["prediction"]).abs().sum())
        window_rows.append(
            {
                "window_start": pd.Timestamp(window_start),
                "actual_sum": actual_sum,
                "baseline_abs_error_sum": base_abs,
                "candidate_abs_error_sum": cand_abs,
            }
        )
    blocks = pd.DataFrame(window_rows)
    if blocks.empty or blocks["actual_sum"].sum() == 0:
        return np.nan, np.nan, np.nan
    point = (
        blocks["baseline_abs_error_sum"].sum() / blocks["actual_sum"].sum()
        - blocks["candidate_abs_error_sum"].sum() / blocks["actual_sum"].sum()
    ) * 100
    rng = np.random.default_rng(seed)
    values = []
    block_idx = np.arange(len(blocks))
    for _ in range(n_bootstrap):
        sampled = blocks.iloc[rng.choice(block_idx, size=len(block_idx), replace=True)]
        actual_sum = float(sampled["actual_sum"].sum())
        if actual_sum == 0:
            continue
        diff = (
            sampled["baseline_abs_error_sum"].sum() / actual_sum
            - sampled["candidate_abs_error_sum"].sum() / actual_sum
        ) * 100
        values.append(diff)
    if not values:
        return point, np.nan, np.nan
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(point), float(lower), float(upper)


def paired_significance_tests(preds: pd.DataFrame) -> pd.DataFrame:
    """Test whether candidate models reduce strict-recursive errors versus Q1 SES."""
    baseline_model = "q1_store_product_exp_smoothing"
    candidate_specs = [
        ("ridge_full_external", "综合 Ridge vs 问题一简单指数平滑"),
        ("hybrid_low_q1_regular_ridge", "低销量混合策略 vs 问题一简单指数平滑"),
    ]
    labels = preds[["model", "model_label"]].drop_duplicates().set_index("model")["model_label"]
    rows = []
    for candidate_model, comparison in candidate_specs:
        base = preds[preds["model"] == baseline_model].copy()
        cand = preds[preds["model"] == candidate_model].copy()
        key_cols = ["window_start", "window_end", "store_id", "store_name", "product_id", "product_name", "category"]
        base_grouped = (
            base.groupby(key_cols, as_index=False, dropna=False)
            .agg(actual_sum=("actual", "sum"), baseline_abs_error=("abs_error", "sum"))
        )
        cand_grouped = (
            cand.groupby(key_cols, as_index=False, dropna=False)
            .agg(candidate_abs_error=("abs_error", "sum"))
        )
        paired = base_grouped.merge(cand_grouped, on=key_cols, how="inner")
        diff = paired["baseline_abs_error"] - paired["candidate_abs_error"]
        baseline_wape = float(base["abs_error"].sum() / base["actual"].sum() * 100)
        candidate_wape = float(cand["abs_error"].sum() / cand["actual"].sum() * 100)
        wape_diff, ci_low, ci_high = bootstrap_wape_diff_by_window(
            preds, baseline_model, candidate_model
        )
        if len(diff) < 3 or np.allclose(diff, 0):
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
            note = "配对差值不足或全为 0，Wilcoxon 不适用"
        else:
            result = stats.wilcoxon(
                paired["baseline_abs_error"],
                paired["candidate_abs_error"],
                alternative="greater",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(result.statistic)
            wilcoxon_p = float(result.pvalue)
            note = "p<0.05 且 CI 不含 0 才判定误差显著下降"
        if pd.notna(wilcoxon_p) and wilcoxon_p < 0.05 and ci_low > 0:
            conclusion = "达到统计显著改进"
        else:
            conclusion = "未达统计显著改进"
        rows.append(
            {
                "comparison": comparison,
                "baseline_model": baseline_model,
                "baseline_label": labels.get(baseline_model, baseline_model),
                "candidate_model": candidate_model,
                "candidate_label": labels.get(candidate_model, candidate_model),
                "paired_unit": "window_start + store_id + product_id 的 7 日绝对误差",
                "n_pairs": int(len(paired)),
                "baseline_WAPE_pct": baseline_wape,
                "candidate_WAPE_pct": candidate_wape,
                "WAPE_diff_baseline_minus_candidate_pct_points": wape_diff,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p_value_greater": wilcoxon_p,
                "bootstrap_n": 5000,
                "bootstrap_CI_lower_pct_points": ci_low,
                "bootstrap_CI_upper_pct_points": ci_high,
                "conclusion": conclusion,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def classification_summary(cls: pd.DataFrame) -> pd.DataFrame:
    out = (
        cls.groupby(["window_start", "demand_class", "is_low_volume"], as_index=False)
        .agg(
            series_count=("product_id", "count"),
            mean_history_sales=("history_average_sales", "mean"),
            mean_zero_ratio=("history_zero_sales_ratio", "mean"),
        )
        .sort_values(["window_start", "is_low_volume", "demand_class"], ascending=[True, False, True])
    )
    out["window_start"] = out["window_start"].dt.date.astype(str)
    return out


def plot_strategy_wape(metrics: pd.DataFrame) -> None:
    sp = metrics[metrics["level"] == "store_product"].sort_values("WAPE_pct")
    label_map = {
        "q1_store_product_exp_smoothing": "Q1 exp smoothing",
        "ridge_full_external": "Full Ridge",
        "q1_low_volume_shrunk": "Low-volume shrink SES",
        "ridge_low_volume_shrunk": "Low-volume shrink Ridge",
        "hybrid_low_q1_regular_ridge": "Low SES + regular Ridge",
        "hybrid_low_shrunk_regular_ridge": "Low shrink + regular Ridge",
    }
    labels = [label_map.get(model, model) for model in sp["model"]]
    values = sp["WAPE_pct"].tolist()
    plt.figure(figsize=(10, 5.4))
    colors = ["#2F6B55" if model == sp.iloc[0]["model"] else "#6B7280" for model in sp["model"]]
    plt.barh(labels, values, color=colors)
    plt.xlabel("WAPE (%)")
    for idx, value in enumerate(values):
        plt.text(value + 0.2, idx, f"{value:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "q4_low_volume_strategy_wape.png", dpi=180)
    plt.close()


def build_report(
    metrics: pd.DataFrame,
    by_class: pd.DataFrame,
    cls_summary: pd.DataFrame,
    significance: pd.DataFrame,
) -> str:
    sp = metrics[metrics["level"] == "store_product"].copy()
    sp_table = sp[
        [
            "model_label",
            "strategy_rule",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "WAPE_pct_point_change_vs_q1",
            "WAPE_pct_point_change_vs_ridge",
            "actual_sum",
            "prediction_sum",
            "bias_pct_of_actual",
            "n",
        ]
    ]
    best = sp.sort_values(["WAPE", "MAE", "RMSE"]).iloc[0]
    q1 = sp[sp["model"] == "q1_store_product_exp_smoothing"].iloc[0]
    ridge = sp[sp["model"] == "ridge_full_external"].iloc[0]

    low_volume_rows = by_class[by_class["is_low_volume"] == 1][
        ["model_label", "demand_class", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]
    ]
    class_counts = (
        cls_summary.groupby(["demand_class", "is_low_volume"], as_index=False)
        .agg(
            mean_series_count=("series_count", "mean"),
            mean_history_sales=("mean_history_sales", "mean"),
            mean_zero_ratio=("mean_zero_ratio", "mean"),
        )
        .sort_values(["is_low_volume", "demand_class"], ascending=[False, True])
    )
    best_note = (
        "可作为论文中的稳健性补充方案，但仍需说明其只是规则化后处理。"
        if best["model"] not in {"q1_store_product_exp_smoothing", "ridge_full_external"}
        else "说明新增低销量后处理没有带来稳定主口径改进，不应强行替换主模型。"
    )

    return f"""# 问题四低销量序列鲁棒性策略检验报告

执行日期：{date.today().isoformat()}

## 1. 检验目的

门店-商品序列中低销量和零销量较多，最细粒度 WAPE 偏高。为避免盲目引入 Croston/TSB 等复杂间歇需求模型，本检验只评估可解释的低销量分层后处理策略：每个验证窗口开始日前，根据历史平均销量、零销量比例和非零销量比例识别低销量序列；低销量序列可向“近 28 日均值、商品历史均值、类别历史均值”的层级均值收缩。

分类和收缩目标均只使用窗口开始日前历史数据，未使用验证窗口真实销量。

## 2. 低销量分类概况

{df_to_md(class_counts, 20)}

逐窗口分类明细见 `tables/q4_low_volume_classification.csv`。

## 3. 门店-商品主粒度策略比较

{df_to_md(sp_table, 20)}

完整分层指标见 `tables/q4_low_volume_strategy_metrics.csv`，按需求类别拆分指标见 `tables/q4_low_volume_strategy_by_class.csv`。图表已保存至 `figures/q4_low_volume_strategy_wape.png`。

## 4. 低销量类别误差

{df_to_md(low_volume_rows, 30)}

## 5. 方法取舍

严格递推门店-商品粒度下，本检验最佳策略为 **{best['model_label']}**，WAPE={best['WAPE_pct']:.2f}%。问题一简单指数平滑 WAPE={q1['WAPE_pct']:.2f}%，完整综合 Ridge WAPE={ridge['WAPE_pct']:.2f}%。

{best_note}

## 6. 误差显著性检验

{df_to_md(significance, 10)}

检验以同一严格 7 日递推验证集为基础。Wilcoxon 检验的配对单位为“验证窗口 × 门店--商品序列”的 7 日绝对误差；bootstrap 置信区间按 7 日窗口块重采样计算 WAPE 差。若 p 值不小于 0.05 或置信区间包含 0，本文不写“误差显著改进”。

论文建议把低销量分析写入模型评价和局限性部分：低销量序列是细粒度误差的主要来源之一；简单的层级收缩可作为稳健性检查，但若未显著优于主模型，就不应为了“看起来高级”而替换最终预测模型。
"""


def main() -> None:
    for path in [TABLES, OUTPUTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "processed" / "modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    panel["date"] = pd.to_datetime(panel["date"])

    base_preds = read_base_predictions()
    base = pivot_base_predictions(base_preds)
    window_starts = sorted(pd.to_datetime(base["window_start"]).drop_duplicates().tolist())
    cls = build_classification(panel, window_starts)
    save_csv(cls, "q4_low_volume_classification.csv")

    strategy_preds = build_strategy_predictions(base, cls)
    save_csv(strategy_preds, "q4_low_volume_strategy_predictions.csv")

    metrics = metric_table(strategy_preds)
    by_class = metrics_by_demand_class(strategy_preds)
    cls_summary = classification_summary(cls)
    significance = paired_significance_tests(strategy_preds)
    save_csv(metrics, "q4_low_volume_strategy_metrics.csv")
    save_csv(by_class, "q4_low_volume_strategy_by_class.csv")
    save_csv(cls_summary, "q4_low_volume_classification_summary.csv")
    save_csv(significance, "q4_significance_tests.csv")
    plot_strategy_wape(metrics)

    report = build_report(metrics, by_class, cls_summary, significance)
    (OUTPUTS / "q4_low_volume_strategy_report.md").write_text(report, encoding="utf-8")

    sp = metrics[metrics["level"] == "store_product"].set_index("model")
    best = metrics[metrics["level"] == "store_product"].sort_values("WAPE").iloc[0]
    low_share = float(cls["is_low_volume"].mean())
    row = (
        f"| {date.today().isoformat()} | 问题四低销量序列鲁棒性策略检验 | "
        "processed: modeling_base_table.csv；输入 q4_ablation_predictions.csv；未修改 data/raw | "
        "问题一指数平滑、完整Ridge、低销量层级收缩、低销量/常规混合策略 | "
        "窗口开始日前历史均值、零销量比例、近28日均值、商品均值、类别均值；验证窗口内基模型仍为严格递推预测 | "
        f"低销量序列窗口占比={low_share:.1%}；门店-商品 WAPE：最佳={best['model_label']} {best['WAPE_pct']:.2f}%；"
        f"问题一指数平滑={sp.loc['q1_store_product_exp_smoothing','WAPE_pct']:.2f}%；"
        f"完整Ridge={sp.loc['ridge_full_external','WAPE_pct']:.2f}% | "
        "低销量策略用于解释和稳健性检查；是否替换主模型以严格递推指标为准 | "
        "分类与收缩只使用窗口开始日前历史数据；层级收缩为规则后处理，不等同于新因果机制 | "
        "将低销量误差来源和策略取舍写入模型评价/附录 |"
    )
    prepend_result_log(row)

    summary = {
        "validation_mode": "strict_recursive_7day",
        "low_volume_window_share": low_share,
        "store_product_metrics": metrics[metrics["level"] == "store_product"][
            ["model", "model_label", "MAE", "RMSE", "WAPE", "WAPE_pct"]
        ].to_dict("records"),
        "best_store_product_model": str(best["model"]),
        "best_store_product_wape_pct": float(best["WAPE_pct"]),
    }
    (OUTPUTS / "q4_low_volume_strategy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
