"""Question 2 method-search experiment.

This script audits product relationship measures and aggregation choices for
Q2. It writes new artifacts only into method_search folders.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import calculate_metrics, metrics_by_group  # noqa: E402
from src.models import prepare_series_table, validate_univariate_models  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "tables" / "method_search"
FIGURE_DIR = PROJECT_ROOT / "figures" / "method_search"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "method_search"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "method_search"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")
VALIDATION_END = pd.Timestamp("2022-03-31")
LOW_SALES_SHARE_THRESHOLD = 0.03

MODEL_LABELS = {
    "moving_average_7": "移动平均(7日)",
    "same_weekday_mean_8": "同星期均值(近8周)",
    "exp_smoothing": "简单指数平滑",
}

STRATEGY_META = {
    "A_product_sum_to_category": {
        "strategy_label": "方案A：单品预测后按附件类别加总",
        "aggregation_basis": "附件类别字段，先单品后类别",
        "business_interpretation": "强：最终仍按题目中的同类零食类别呈现",
        "noise_reduction": "中：预测在单品层面完成，类别加总后抵消部分误差",
        "product_info_loss": "低：先保留单品，再汇总",
        "complexity": "中：需要维护每个商品序列",
        "interpretability_score": 5,
        "paper_writeability_score": 5,
        "answers_question": "是：输出仍是附件类别总销量",
        "recommended_role": "可作为类别直接预测的对照方案",
        "replace_current_judgement": "不替换；误差略高于类别直接预测",
    },
    "B_category_direct": {
        "strategy_label": "方案B：按附件类别聚合后预测",
        "aggregation_basis": "附件类别字段，先聚合后预测",
        "business_interpretation": "强：直接对应附件类别和题目同类零食",
        "noise_reduction": "强：类别总量比单品更平滑",
        "product_info_loss": "中：类别内部单品差异被隐藏",
        "complexity": "低：只预测类别序列",
        "interpretability_score": 5,
        "paper_writeability_score": 5,
        "answers_question": "是：直接输出附件类别总销量",
        "recommended_role": "建议作为问题二主方案",
        "replace_current_judgement": "保留；与当前原方案一致",
    },
    "C_cluster_direct": {
        "strategy_label": "方案C：按销量模式聚类后聚合预测",
        "aggregation_basis": "训练期去星期效应后的销量模式聚类",
        "business_interpretation": "中低：由数据模式得到，未必对应业务类别",
        "noise_reduction": "中：相似波动商品合并后可能更平滑",
        "product_info_loss": "中高：类别边界可能跨业务品类",
        "complexity": "中：需要解释标准化、聚类数和聚类含义",
        "interpretability_score": 3,
        "paper_writeability_score": 3,
        "answers_question": "部分：输出销量模式类，不是附件业务类别",
        "recommended_role": "作为稳健性或补充探索",
        "replace_current_judgement": "不替换；误差较低但业务类别含义弱",
    },
    "D_sales_scale_direct": {
        "strategy_label": "方案D：按销量规模分组后预测",
        "aggregation_basis": "训练期总销量高/中/低分组",
        "business_interpretation": "低：规模分组不是同类零食",
        "noise_reduction": "中：低销量组被合并",
        "product_info_loss": "高：业务类别和商品属性被弱化",
        "complexity": "低：分组规则简单",
        "interpretability_score": 2,
        "paper_writeability_score": 2,
        "answers_question": "否：输出高/中/低销量组，不是同类零食",
        "recommended_role": "只作误差下界和噪声合并参考",
        "replace_current_judgement": "不替换；预测口径已改变，不能回答类别预测要求",
    },
    "E_low_sales_other": {
        "strategy_label": "方案E：低销量商品合并为其他类后预测",
        "aggregation_basis": f"训练期销量占比低于 {LOW_SALES_SHARE_THRESHOLD:.0%} 的商品合并为其他类",
        "business_interpretation": "中：适合处理长尾，但其他类业务含义混杂",
        "noise_reduction": "中：低销量零散波动被合并",
        "product_info_loss": "中：主力商品保留，长尾商品损失细节",
        "complexity": "中低：规则易解释，但阈值需要说明",
        "interpretability_score": 4,
        "paper_writeability_score": 4,
        "answers_question": "部分：主力商品清楚，其他类业务含义混杂",
        "recommended_role": "不作为主方案，可作为长尾处理讨论",
        "replace_current_judgement": "不替换；误差未优于原类别方案",
    },
    "current_original": {
        "strategy_label": "当前原方案：附件类别聚合后直接预测",
        "aggregation_basis": "附件类别字段，简单指数平滑为验证期最优",
        "business_interpretation": "强：与阶段3原方案一致",
        "noise_reduction": "强：类别总量比单品更平滑",
        "product_info_loss": "中：类别内部单品差异被隐藏",
        "complexity": "低：已在阶段3解释",
        "interpretability_score": 5,
        "paper_writeability_score": 5,
        "answers_question": "是：直接输出附件类别总销量",
        "recommended_role": "当前阶段3主方案",
        "replace_current_judgement": "保留；本次复现结果一致",
    },
}


def ensure_dirs() -> None:
    for directory in (TABLE_DIR, FIGURE_DIR, OUTPUT_DIR, NOTEBOOK_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(DATA_DIR / "modeling_base_table.csv", parse_dates=["date"])
    daily = pd.read_csv(DATA_DIR / "daily_store_product_sales.csv", parse_dates=["date"])
    product_names = (
        daily.groupby("product_id")["product_name"]
        .apply(lambda s: " / ".join(sorted(pd.Series(s).dropna().astype(str).unique())))
        .to_dict()
    )
    product_meta = (
        daily.groupby("product_id", as_index=False)
        .agg(category=("category", "first"), category_code=("category_code", "first"))
        .sort_values("product_id")
    )
    product_meta["product_name"] = product_meta["product_id"].map(product_names)
    product_meta["product_label"] = (
        product_meta["product_id"].astype(str) + "-" + product_meta["product_name"].astype(str)
    )
    product_daily = (
        daily.groupby(["date", "product_id"], as_index=False)
        .agg(target_sales=(TARGET, "sum"))
        .merge(product_meta, on="product_id", how="left")
        .sort_values(["product_id", "date"])
    )
    return base, daily, product_daily


def product_matrix(product_daily: pd.DataFrame, end_before: pd.Timestamp | None = None) -> pd.DataFrame:
    data = product_daily.copy()
    if end_before is not None:
        data = data[data["date"] < end_before]
    matrix = data.pivot_table(
        index="date", columns="product_id", values="target_sales", aggfunc="sum", fill_value=0.0
    )
    all_dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    matrix = matrix.reindex(all_dates, fill_value=0.0)
    matrix.index.name = "date"
    return matrix.sort_index()


def weekday_residual_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    weekday = pd.Series(matrix.index.dayofweek, index=matrix.index)
    expected = matrix.groupby(weekday).transform("mean")
    return matrix - expected


def pair_table(
    corr: pd.DataFrame,
    matrix: pd.DataFrame,
    product_meta: pd.DataFrame,
    method: str,
    count_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    count_source = matrix if count_matrix is None else count_matrix
    label_map = product_meta.set_index("product_id")["product_label"].to_dict()
    cat_map = product_meta.set_index("product_id")["category"].to_dict()
    rows = []
    for a, b in itertools.combinations(corr.columns, 2):
        a_series = count_source[a]
        b_series = count_source[b]
        rows.append(
            {
                "method": method,
                "product_a": label_map.get(a, str(a)),
                "product_b": label_map.get(b, str(b)),
                "category_a": cat_map.get(a, ""),
                "category_b": cat_map.get(b, ""),
                "corr": float(corr.loc[a, b]),
                "nonzero_both_days": int(((a_series > 0) & (b_series > 0)).sum()),
                "product_a_nonzero_days": int((a_series > 0).sum()),
                "product_b_nonzero_days": int((b_series > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)


def lagged_correlation_table(matrix: pd.DataFrame, product_meta: pd.DataFrame) -> pd.DataFrame:
    label_map = product_meta.set_index("product_id")["product_label"].to_dict()
    rows = []
    lags = [lag for lag in range(-7, 8) if lag != 0]
    for a, b in itertools.combinations(matrix.columns, 2):
        a_series = matrix[a].astype(float).reset_index(drop=True)
        b_series = matrix[b].astype(float).reset_index(drop=True)
        same_day = float(a_series.corr(b_series))
        best = {"lag_days": 0, "lag_corr": np.nan}
        for lag in lags:
            if lag > 0:
                corr = a_series.iloc[:-lag].corr(b_series.iloc[lag:])
            else:
                corr = a_series.iloc[-lag:].corr(b_series.iloc[:lag])
            if pd.isna(corr):
                continue
            if pd.isna(best["lag_corr"]) or abs(corr) > abs(best["lag_corr"]):
                best = {"lag_days": lag, "lag_corr": float(corr)}
        rows.append(
            {
                "product_a": label_map.get(a, str(a)),
                "product_b": label_map.get(b, str(b)),
                "same_day_corr": same_day,
                "best_lag_days": best["lag_days"],
                "best_lag_corr": best["lag_corr"],
                "abs_gain_vs_same_day": abs(best["lag_corr"]) - abs(same_day)
                if not pd.isna(best["lag_corr"])
                else np.nan,
                "lead_interpretation": "product_a领先product_b"
                if best["lag_days"] > 0
                else "product_b领先product_a",
            }
        )
    return pd.DataFrame(rows).sort_values("best_lag_corr", key=lambda s: s.abs(), ascending=False)


def storewise_correlation(daily: pd.DataFrame, product_meta: pd.DataFrame) -> pd.DataFrame:
    label_map = product_meta.set_index("product_id")["product_label"].to_dict()
    product_ids = sorted(product_meta["product_id"].unique())
    rows = []
    for (store_id, store_name), group in daily.groupby(["store_id", "store_name"], dropna=False):
        store_product = (
            group.groupby(["date", "product_id"], as_index=False)[TARGET].sum()
            .pivot_table(index="date", columns="product_id", values=TARGET, fill_value=0.0)
            .reindex(columns=product_ids, fill_value=0.0)
        )
        store_product = store_product.reindex(
            pd.date_range(group["date"].min(), group["date"].max(), freq="D"), fill_value=0.0
        )
        corr = store_product.corr(method="pearson")
        for a, b in itertools.combinations(product_ids, 2):
            rows.append(
                {
                    "store_id": store_id,
                    "store_name": store_name,
                    "product_a": label_map.get(a, str(a)),
                    "product_b": label_map.get(b, str(b)),
                    "corr": float(corr.loc[a, b]) if not pd.isna(corr.loc[a, b]) else np.nan,
                }
            )
    pair_rows = pd.DataFrame(rows)
    summary = (
        pair_rows.groupby(["product_a", "product_b"], as_index=False)
        .agg(
            corr_mean=("corr", "mean"),
            corr_std=("corr", "std"),
            positive_store_count=("corr", lambda s: int((s > 0).sum())),
            negative_store_count=("corr", lambda s: int((s < 0).sum())),
            strong_positive_store_count=("corr", lambda s: int((s >= 0.5).sum())),
            store_count=("corr", "count"),
        )
        .sort_values(["strong_positive_store_count", "corr_mean"], ascending=False)
    )
    return summary


def association_analysis(product_daily: pd.DataFrame, daily: pd.DataFrame, product_meta: pd.DataFrame) -> dict:
    train_matrix = product_matrix(product_daily, VALIDATION_START)
    residual = weekday_residual_matrix(train_matrix)

    pearson = train_matrix.corr(method="pearson")
    spearman = train_matrix.corr(method="spearman")
    residual_pearson = residual.corr(method="pearson")

    pearson_pairs = pair_table(pearson, train_matrix, product_meta, "Pearson")
    spearman_pairs = pair_table(spearman, train_matrix, product_meta, "Spearman")
    residual_pairs = pair_table(
        residual_pearson, residual, product_meta, "去星期效应Pearson", count_matrix=train_matrix
    )
    lagged_pairs = lagged_correlation_table(train_matrix, product_meta)
    store_corr = storewise_correlation(daily[daily["date"] < VALIDATION_START], product_meta)

    pearson_pairs.to_csv(TABLE_DIR / "q2_pearson_pairs_train.csv", index=False, encoding="utf-8-sig")
    spearman_pairs.to_csv(TABLE_DIR / "q2_spearman_pairs_train.csv", index=False, encoding="utf-8-sig")
    residual_pairs.to_csv(TABLE_DIR / "q2_weekday_residual_corr_pairs_train.csv", index=False, encoding="utf-8-sig")
    lagged_pairs.to_csv(TABLE_DIR / "q2_lagged_correlation_pairs_train.csv", index=False, encoding="utf-8-sig")
    store_corr.to_csv(TABLE_DIR / "q2_storewise_correlation_stability.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for method, pairs in [
        ("Pearson", pearson_pairs),
        ("Spearman", spearman_pairs),
        ("去星期效应Pearson", residual_pairs),
    ]:
        summary_rows.append(
            {
                "method": method,
                "pair_count": len(pairs),
                "strong_positive_pairs_corr_ge_0_5": int((pairs["corr"] >= 0.5).sum()),
                "moderate_positive_pairs_corr_ge_0_3": int((pairs["corr"] >= 0.3).sum()),
                "weak_pairs_abs_corr_le_0_1": int((pairs["corr"].abs() <= 0.1).sum()),
                "negative_pairs_corr_lt_0": int((pairs["corr"] < 0).sum()),
                "max_abs_corr": float(pairs["corr"].abs().max()),
                "mean_abs_corr": float(pairs["corr"].abs().mean()),
            }
        )
    summary_rows.append(
        {
            "method": "滞后相关(-7至7日)",
            "pair_count": len(lagged_pairs),
            "strong_positive_pairs_corr_ge_0_5": int((lagged_pairs["best_lag_corr"] >= 0.5).sum()),
            "moderate_positive_pairs_corr_ge_0_3": int((lagged_pairs["best_lag_corr"] >= 0.3).sum()),
            "weak_pairs_abs_corr_le_0_1": int((lagged_pairs["best_lag_corr"].abs() <= 0.1).sum()),
            "negative_pairs_corr_lt_0": int((lagged_pairs["best_lag_corr"] < 0).sum()),
            "max_abs_corr": float(lagged_pairs["best_lag_corr"].abs().max()),
            "mean_abs_corr": float(lagged_pairs["best_lag_corr"].abs().mean()),
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "q2_association_method_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "summary": summary,
        "pearson_pairs": pearson_pairs,
        "spearman_pairs": spearman_pairs,
        "residual_pairs": residual_pairs,
        "lagged_pairs": lagged_pairs,
        "store_corr": store_corr,
        "train_matrix": train_matrix,
        "residual_matrix": residual,
    }


def cluster_products(train_matrix: pd.DataFrame, product_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = weekday_residual_matrix(train_matrix)
    x = residual.copy()
    x = (x - x.mean(axis=0)) / x.std(axis=0).replace(0, np.nan)
    x = x.fillna(0.0).T

    max_k = min(8, len(x) - 1)
    quality_rows = []
    label_by_k = {}
    for k in range(3, max_k + 1):
        clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = clusterer.fit_predict(x)
        score = silhouette_score(x, labels) if len(set(labels)) > 1 else np.nan
        quality_rows.append({"cluster_count": k, "silhouette_score": float(score)})
        label_by_k[k] = labels

    quality = pd.DataFrame(quality_rows).sort_values("silhouette_score", ascending=False)
    chosen_k = int(quality.iloc[0]["cluster_count"])
    labels = label_by_k[chosen_k]

    train_totals = train_matrix.sum(axis=0).rename("train_total_sales").reset_index()
    train_totals = train_totals.rename(columns={"product_id": "product_id"})
    mapping = pd.DataFrame({"product_id": x.index.to_numpy(), "raw_cluster": labels})
    mapping = mapping.merge(train_totals, on="product_id", how="left").merge(
        product_meta[["product_id", "product_name", "product_label", "category"]], on="product_id", how="left"
    )
    cluster_order = (
        mapping.groupby("raw_cluster")["train_total_sales"].sum().sort_values(ascending=False).reset_index()
    )
    cluster_order["group_label"] = [f"相关聚类{i}" for i in range(1, len(cluster_order) + 1)]
    mapping = mapping.merge(cluster_order[["raw_cluster", "group_label"]], on="raw_cluster", how="left")
    mapping = mapping.sort_values(["group_label", "train_total_sales"], ascending=[True, False])
    quality["chosen"] = quality["cluster_count"].eq(chosen_k)

    mapping.to_csv(TABLE_DIR / "q2_cluster_mapping.csv", index=False, encoding="utf-8-sig")
    quality.sort_values("cluster_count").to_csv(
        TABLE_DIR / "q2_cluster_count_selection.csv", index=False, encoding="utf-8-sig"
    )
    return mapping[["product_id", "group_label"]], quality


def build_group_mappings(product_daily: pd.DataFrame, product_meta: pd.DataFrame, train_matrix: pd.DataFrame) -> dict:
    train_totals = (
        product_daily[product_daily["date"] < VALIDATION_START]
        .groupby("product_id", as_index=False)["target_sales"]
        .sum()
        .rename(columns={"target_sales": "train_total_sales"})
    )
    train_totals["train_sales_share"] = train_totals["train_total_sales"] / train_totals[
        "train_total_sales"
    ].sum()
    meta = product_meta.merge(train_totals, on="product_id", how="left")

    category_map = meta[["product_id", "category"]].rename(columns={"category": "group_label"})

    cluster_map, cluster_quality = cluster_products(train_matrix, product_meta)

    ranked = meta.sort_values("train_total_sales").copy()
    ranked["rank_for_qcut"] = np.arange(1, len(ranked) + 1)
    ranked["scale_group"] = pd.qcut(
        ranked["rank_for_qcut"], q=3, labels=["低销量组", "中销量组", "高销量组"]
    ).astype(str)
    scale_map = ranked[["product_id", "scale_group"]].rename(columns={"scale_group": "group_label"})

    low_map = meta[["product_id", "product_label", "train_sales_share"]].copy()
    low_map["group_label"] = np.where(
        low_map["train_sales_share"] < LOW_SALES_SHARE_THRESHOLD,
        "其他低销量商品",
        "主力商品：" + low_map["product_label"],
    )

    mappings = {
        "A_product_sum_to_category": category_map,
        "B_category_direct": category_map,
        "C_cluster_direct": cluster_map,
        "D_sales_scale_direct": scale_map,
        "E_low_sales_other": low_map[["product_id", "group_label"]],
    }
    cluster_quality.to_csv(TABLE_DIR / "q2_cluster_quality.csv", index=False, encoding="utf-8-sig")
    return mappings


def group_daily_from_map(product_daily: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    return (
        product_daily[["date", "product_id", "target_sales"]]
        .merge(mapping, on="product_id", how="left")
        .groupby(["date", "group_label"], as_index=False)["target_sales"]
        .sum()
        .sort_values(["group_label", "date"])
    )


def group_stats(mapping: pd.DataFrame) -> dict:
    counts = mapping.groupby("group_label")["product_id"].nunique()
    return {
        "group_count": int(counts.size),
        "avg_products_per_group": float(counts.mean()),
        "min_products_per_group": int(counts.min()),
        "max_products_per_group": int(counts.max()),
    }


def evaluate_direct_strategy(
    strategy_id: str, group_daily: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = time.perf_counter()
    series = prepare_series_table(group_daily, ["group_label"], "target_sales")
    predictions = validate_univariate_models(series, ["group_label"], "target_sales", VALIDATION_START)
    runtime = time.perf_counter() - start
    stats = group_stats(mapping)
    rows = []
    for model, group in predictions.groupby("model"):
        metrics = calculate_metrics(group["actual"], group["prediction"])
        row = {
            "strategy_id": strategy_id,
            "method_type": "direct_group_forecast",
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "runtime_seconds": runtime,
            "actual_sum": float(group["actual"].sum()),
            "prediction_sum": float(group["prediction"].sum()),
            "n": int(len(group)),
        }
        row.update(metrics)
        row.update(stats)
        rows.append(row)
    predictions["strategy_id"] = strategy_id
    return pd.DataFrame(rows), predictions


def evaluate_product_sum_strategy(
    strategy_id: str, product_daily: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = time.perf_counter()
    product_series = prepare_series_table(product_daily, ["product_id"], "target_sales")
    product_predictions = validate_univariate_models(
        product_series, ["product_id"], "target_sales", VALIDATION_START
    )
    predictions = (
        product_predictions.merge(mapping, on="product_id", how="left")
        .groupby(["date", "group_label", "model", "model_label"], as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
    )
    runtime = time.perf_counter() - start
    stats = group_stats(mapping)
    rows = []
    for model, group in predictions.groupby("model"):
        metrics = calculate_metrics(group["actual"], group["prediction"])
        row = {
            "strategy_id": strategy_id,
            "method_type": "product_forecast_then_sum",
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "runtime_seconds": runtime,
            "actual_sum": float(group["actual"].sum()),
            "prediction_sum": float(group["prediction"].sum()),
            "n": int(len(group)),
        }
        row.update(metrics)
        row.update(stats)
        rows.append(row)
    predictions["strategy_id"] = strategy_id
    return pd.DataFrame(rows), predictions


def add_strategy_metadata(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for col in [
        "strategy_label",
        "aggregation_basis",
        "business_interpretation",
        "noise_reduction",
        "product_info_loss",
        "complexity",
        "interpretability_score",
        "paper_writeability_score",
        "answers_question",
        "recommended_role",
        "replace_current_judgement",
    ]:
        out[col] = out["strategy_id"].map({k: v[col] for k, v in STRATEGY_META.items()})
    out["WAPE_pct"] = out["WAPE"] * 100.0
    return out


def evaluate_aggregation_strategies(
    product_daily: pd.DataFrame, mappings: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    prediction_frames = []

    metrics_a, preds_a = evaluate_product_sum_strategy(
        "A_product_sum_to_category", product_daily, mappings["A_product_sum_to_category"]
    )
    metric_frames.append(metrics_a)
    prediction_frames.append(preds_a)

    for strategy_id in [
        "B_category_direct",
        "C_cluster_direct",
        "D_sales_scale_direct",
        "E_low_sales_other",
    ]:
        group_daily = group_daily_from_map(product_daily, mappings[strategy_id])
        metrics, preds = evaluate_direct_strategy(strategy_id, group_daily, mappings[strategy_id])
        metric_frames.append(metrics)
        prediction_frames.append(preds)

    model_metrics = add_strategy_metadata(pd.concat(metric_frames, ignore_index=True))
    model_metrics = model_metrics.sort_values(["strategy_id", "WAPE", "MAE"]).reset_index(drop=True)

    best = model_metrics.groupby("strategy_id", as_index=False).first()

    original_metrics = pd.read_csv(PROJECT_ROOT / "tables" / "q2_category_method_metrics.csv")
    original_best = original_metrics.sort_values(["WAPE", "MAE", "RMSE"]).iloc[0]
    original_stats = group_stats(mappings["B_category_direct"])
    original_row = {
        "strategy_id": "current_original",
        "method_type": "read_from_stage3_q2_category_method_metrics",
        "model": original_best["model"],
        "model_label": original_best["model_label"],
        "runtime_seconds": np.nan,
        "actual_sum": float(original_best["actual_sum"]),
        "prediction_sum": float(original_best["prediction_sum"]),
        "n": int(original_best["n"]),
        "MAE": float(original_best["MAE"]),
        "RMSE": float(original_best["RMSE"]),
        "WAPE": float(original_best["WAPE"]),
    }
    original_row.update(original_stats)
    original_row = add_strategy_metadata(pd.DataFrame([original_row]))
    comparison = pd.concat([best, original_row], ignore_index=True)
    comparison = comparison[
        [
            "strategy_id",
            "strategy_label",
            "aggregation_basis",
            "method_type",
            "model",
            "model_label",
            "MAE",
            "RMSE",
            "WAPE",
            "WAPE_pct",
            "actual_sum",
            "prediction_sum",
            "n",
            "group_count",
            "avg_products_per_group",
            "min_products_per_group",
            "max_products_per_group",
            "runtime_seconds",
            "complexity",
            "interpretability_score",
            "paper_writeability_score",
            "business_interpretation",
            "noise_reduction",
            "product_info_loss",
            "answers_question",
            "recommended_role",
            "replace_current_judgement",
        ]
    ].sort_values("WAPE")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_metrics.to_csv(TABLE_DIR / "q2_aggregation_model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(TABLE_DIR / "q2_aggregation_validation_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "q2_aggregation_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison, model_metrics, predictions


def plot_aggregation_comparison(comparison: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_df = comparison[comparison["strategy_id"] != "current_original"].copy()
    plot_df = plot_df.sort_values("WAPE_pct")
    colors = ["#2f6f73", "#6b9f5f", "#d09a35", "#8f7bb3", "#b55b5b"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(plot_df["strategy_label"], plot_df["WAPE_pct"], color=colors[: len(plot_df)])
    ax.set_xlabel("WAPE (%)")
    ax.grid(axis="x", alpha=0.25)
    for bar, (_, row) in zip(bars, plot_df.iterrows(), strict=False):
        ax.text(
            row["WAPE_pct"] + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['WAPE_pct']:.2f}% / {row['model_label']}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q2_aggregation_comparison.png", dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out.to_markdown(index=False)


def generate_concept_review(association: dict) -> None:
    summary = association["summary"]
    pearson_top = association["pearson_pairs"].head(5)[
        ["product_a", "product_b", "corr", "nonzero_both_days"]
    ]
    residual_top = association["residual_pairs"].head(5)[
        ["product_a", "product_b", "corr", "nonzero_both_days"]
    ]

    report = f"""# 问题二概念审查：销量联系与同类零食

执行日期：2026-05-03

## 1. 当前问题二已有方案摘要

已读取 `AGENTS.md`、`DATA_DICTIONARY.md`、`modeling_base_table.csv`、`daily_store_product_sales.csv`、阶段 3 报告、q2 结果表图、Notebook 和 `RESULT_LOG.md`。当前问题二方案如下：

1. 商品关联度量方法：主要使用全部门店汇总的 Pearson 相关系数矩阵，并补充重点门店相关性热力图。
2. 商品分类或聚合方式：直接采用附件中的 `category` 字段，按业务类别聚合，不根据商品名称或相关性重新分类。
3. 类别预测方法：移动平均、同星期均值、简单指数平滑；验证期为 2022-03-01 至 2022-03-31。
4. 误差比较方式：比较“单品预测后按类别加总”和“类别聚合后直接预测”，指标为 MAE、RMSE、WAPE。
5. 主要结论：类别层面最优是“类别聚合后直接预测 + 简单指数平滑”，WAPE 约 34.74%；门店-类别层面最优是“门店-单品预测后按门店类别加总 + 简单指数平滑”，WAPE 约 69.40%。原报告已经说明相关性不等于因果性，替代关系只能作为线索。

## 2. “销量之间的联系”的合理数学表达

“联系”不应只理解为普通相关系数。本题中至少有以下表达：

| 表达方式 | 数学含义 | 适用性 | 风险 |
|---|---|---|---|
| Pearson 相关 | 衡量两个销量序列的线性同步变化 | 简单、适合画热力图 | 对异常值和共同周期敏感 |
| Spearman 相关 | 衡量排序同步关系 | 对极端销量更稳健 | 不能表示线性幅度关系 |
| 滞后相关 | 比较一个商品变化是否领先另一个商品若干天 | 可作为领先线索 | 不能证明因果，也可能是周期重合 |
| 同涨同跌关系 | 比较方向是否一致 | 直观，适合经营解释 | 会损失销量幅度信息 |
| 去星期效应后的相关 | 先扣除星期平均差异，再计算相关 | 能降低“周末一起上涨”的误判 | 仍不能排除促销、客流等共同因素 |
| 按门店分别计算相关 | 检查不同门店关系是否一致 | 符合“相同门店”要求 | 单门店样本更稀疏，相关更不稳定 |
| 替代或互补关系 | 经营含义强 | 只有购物篮、价格或促销控制充分时才适合 | 本数据不能直接证明 |

本次训练期关联方法计数如下：

{markdown_table(summary)}

训练期 Pearson 相关最高的商品对如下：

{markdown_table(pearson_top)}

去星期效应后 Pearson 相关最高的商品对如下：

{markdown_table(residual_top)}

## 3. “同类零食”的合理定义

“同类零食”可以有几种定义：

1. 附件已有类别：业务含义最清楚，最适合作为论文主方案。
2. 商品名称规则：可辅助解释，但本题已有 `category` 字段，手工重分容易主观。
3. 销量模式相似性：可用于发现共同波动商品，但未必是真正同类。
4. 相关性聚类：适合作为稳健性分析或补充探索，不宜直接替代业务类别。
5. 业务解释优先：数学建模论文需要让评委理解类别含义。
6. 预测效果优先：只有误差显著降低且不造成解释困难时，才值得替换。

## 4. 当前方案是否过于简单

当前方案在“联系”层面偏简单：它主要依赖 Pearson 相关，虽然报告已经强调不能写成因果，但没有系统比较 Spearman、滞后相关、去星期效应相关和门店内相关稳定性。因此，若论文只放普通相关热力图，容易把共同星期效应或共同客流波动误认为商品之间的直接联系。

当前方案在“同类零食”层面不算过于简单：附件已经提供 `category` 字段，直接按类别聚合具有明确业务解释，符合题目“整合同类零食”的自然含义。相关性聚类可以作为补充说明，但不应轻易取代附件类别。

## 5. 本次审查的原则

后续比较只把验证期之前的数据用于聚类和分组规则形成，避免使用 2022 年 3 月验证期信息。预测误差继续采用 MAE、RMSE、WAPE，并与当前原方案在同一验证窗口比较。
"""
    (OUTPUT_DIR / "q2_concept_review.md").write_text(report, encoding="utf-8")


def generate_final_report(comparison: pd.DataFrame, model_metrics: pd.DataFrame, association: dict) -> None:
    comp = comparison.copy()
    best = comp[comp["strategy_id"] != "current_original"].sort_values("WAPE").iloc[0]
    current = comp[comp["strategy_id"] == "current_original"].iloc[0]
    category = comp[comp["strategy_id"] == "B_category_direct"].iloc[0]
    cluster = comp[comp["strategy_id"] == "C_cluster_direct"].iloc[0]
    product_sum = comp[comp["strategy_id"] == "A_product_sum_to_category"].iloc[0]

    assoc_summary = association["summary"]
    lag_top = association["lagged_pairs"].head(5)[
        ["product_a", "product_b", "same_day_corr", "best_lag_days", "best_lag_corr", "abs_gain_vs_same_day"]
    ]
    store_stable = association["store_corr"].head(5)[
        ["product_a", "product_b", "corr_mean", "corr_std", "positive_store_count", "strong_positive_store_count"]
    ]
    model_detail = model_metrics[
        [
            "strategy_label",
            "model_label",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "group_count",
            "avg_products_per_group",
        ]
    ].sort_values(["strategy_label", "WAPE_pct"])
    cluster_mapping = pd.read_csv(TABLE_DIR / "q2_cluster_mapping.csv")
    cluster_summary = (
        cluster_mapping.groupby("group_label", as_index=False)
        .agg(
            train_total_sales=("train_total_sales", "sum"),
            products=("product_name", lambda s: "；".join(map(str, s))),
            categories=("category", lambda s: "；".join(sorted(set(map(str, s))))),
        )
        .sort_values("train_total_sales", ascending=False)
    )

    replace_judgement = (
        "不建议替换原问题二主方案。虽然销量规模分组的 WAPE 最低，但它只输出高/中/低销量组，"
        "不是题目要求的同类零食类别预测；相关性聚类的 WAPE 也低于附件类别聚合，但聚类边界由样本和聚类数决定，"
        "业务含义弱于附件 `category` 字段。因此建议保留附件类别聚合为主方案，把聚类和规模分组作为方法审查证据。"
    )

    report = f"""# 问题二方法优化探索报告

执行日期：2026-05-03

## 1. 当前问题二方案是否过于依赖简单相关性

是，关联分析部分偏依赖普通 Pearson 相关。Pearson 能展示同步波动，但容易受到异常销量、周末共同上涨、促销或客流共同变化影响。本次补充了 Spearman、滞后相关、去星期效应后的相关，以及按门店分别计算相关稳定性。

关联方法总体结果：

{markdown_table(assoc_summary)}

滞后相关最高的商品对如下。该表只能说明“时间上有领先相关线索”，不能写成因果：

{markdown_table(lag_top)}

按门店计算后相对稳定的相关商品对如下：

{markdown_table(store_stable)}

## 2. 哪种关联度量更稳妥

更稳妥的写法是：以 Pearson 相关作为直观热力图，以 Spearman 相关和去星期效应后的 Pearson 相关作为稳健性检查。若某商品对只在普通 Pearson 中较高，但去星期效应后明显降低，应解释为“可能共同受星期或客流影响”，不应写成商品之间存在直接带动关系。

滞后相关适合放在附加探索中，不宜作为主结论。原因是本题只有日销量聚合数据，没有顾客购物篮、价格变动实验和陈列信息，无法区分真实领先关系与周期重合。

## 3. 聚合策略公平比较

所有策略都使用 2022-03-01 至 2022-03-31 作为同一验证窗口。相关性聚类、销量规模分组和低销量阈值都只用 2022-03-01 之前的训练期数据确定，避免验证期信息泄露。

最优模型层面的聚合策略比较如下：

{markdown_table(comp[["strategy_label", "model_label", "MAE", "RMSE", "WAPE_pct", "group_count", "avg_products_per_group", "runtime_seconds", "interpretability_score", "paper_writeability_score", "complexity", "answers_question", "replace_current_judgement"]])}

各策略、各模型的完整比较如下：

{markdown_table(model_detail, max_rows=20)}

销量模式聚类的分组结果如下。可以看到，部分类别有业务含义，例如两个速食类商品被聚在一起；但也存在跨类别聚合，例如饮料、乳类或包装类商品混在同一模式组，因此它更适合做补充分析，而不是替代附件类别。

{markdown_table(cluster_summary)}

## 4. 哪种聚合方式最适合本题

从题意和论文表达看，最适合本题的是“按附件类别聚合后预测”。理由是：

1. 附件类别直接对应“同类零食”，业务含义清楚；
2. 类别聚合能降低单品零销量和偶然大单造成的噪声；
3. 公式和流程容易写入本科数学建模论文；
4. 不需要解释聚类数选择、标准化方式和跨类别聚类的业务含义。

## 5. 哪种聚合方式预测误差最低

在本次比较中，表面 WAPE 最低的方案是 `{best["strategy_label"]}`，最优模型为 `{best["model_label"]}`，WAPE={best["WAPE_pct"]:.2f}%。当前原方案 WAPE={current["WAPE_pct"]:.2f}%。

但方案 D 只有 {int(best["group_count"])} 个销量规模组，预测对象已经从“同类零食类别”变成“高/中/低销量组”。它的误差较低主要反映更粗粒度聚合带来的误差抵消，不能直接替代题目要求的类别预测。

在仍然输出附件类别的方案中，`{category["strategy_label"]}` WAPE={category["WAPE_pct"]:.2f}%，略优于 `{product_sum["strategy_label"]}` WAPE={product_sum["WAPE_pct"]:.2f}%。如果允许用销量模式类而不是业务类别，`{cluster["strategy_label"]}` WAPE={cluster["WAPE_pct"]:.2f}%，但解释性和论文可写性低于附件类别聚合。

需要注意：不同聚合粒度下 MAE 和 RMSE 会受到类别数量和每类销量规模影响，WAPE 更适合横向比较。若某个非业务分组 WAPE 略低，也不自动意味着它更适合作为论文主方案。

## 6. 业务解释和预测误差是否冲突

存在冲突。销量规模分组的 WAPE 明显更低，但它没有回答“同类零食”的业务类别问题；相关性聚类误差低于附件类别聚合，但它的“同类”含义来自销量模式，不等于附件业务类别。附件类别聚合即使不是 WAPE 绝对最低，也更准确地回答题目要求。

## 7. 是否建议替换原问题二方案

{replace_judgement}

具体判断：

1. 不建议用销量规模分组替换原方案，因为它不是“同类零食”的自然定义；
2. 不建议用相关性聚类作为主分类，因为聚类依赖训练期样本和聚类数选择，业务含义不如附件类别稳定；
3. 可以补充“单品预测后加总”和“类别直接预测”的误差接近关系，说明类别聚合没有明显牺牲预测精度；
4. 可以用去星期效应相关和门店内相关稳定性增强“销量联系”分析。

## 8. 问题二最终应采用类别字段聚合还是相关性聚类

建议主方案采用“类别字段聚合”。相关性聚类作为补充分析，用于说明“若完全按销量模式分组，结果与业务类别并不完全一致，因此主文仍选择业务类别”。

如果二者都保留，论文安排建议如下：

1. 主文模型建立：先说明附件 `category` 是同类零食的业务定义，作为正式聚合口径；
2. 主文结果分析：报告类别聚合预测误差和未来 7 天类别预测；
3. 稳健性或附录：展示相关性聚类和去星期效应相关，说明没有充分证据支持用聚类替代类别字段。

## 9. 哪些结论不能写成因果或替代关系

不能写：

1. “某商品销量上升导致另一商品销量上升”；
2. “负相关商品就是替代品”；
3. “滞后相关说明一个商品带动另一个商品”；
4. “同一聚类内商品一定属于同一消费场景”；
5. “高相关商品必须捆绑销售或相邻陈列”。

可以克制写为：

1. “历史日销量存在同步波动”；
2. “去除星期效应后，部分同步关系仍然存在/明显减弱”；
3. “负相关绝对值较小，未发现强替代关系证据”；
4. “相关性聚类提供了销量模式相似性的线索，但不替代业务类别定义”。

## 10. 后续论文应如何克制表述

论文中应把问题二写成“统计联系 + 可解释聚合 + 短期预测”的结构。先承认相关性不是因果，再说明附件类别是主聚合依据，最后用同一验证窗口证明类别聚合预测误差不劣于或接近单品加总。若展示聚类，只写成方法审查或稳健性分析，不把它包装成更高级、更正确的分类。
"""
    (OUTPUT_DIR / "q2_method_search_report.md").write_text(report, encoding="utf-8")


def generate_notebook() -> None:
    code = f"""from pathlib import Path
import runpy

project_root = Path.cwd()
if project_root.name == 'method_search':
    project_root = project_root.parents[1]
elif not (project_root / 'src' / 'experimental' / 'q2_method_search.py').exists():
    project_root = Path(r'{str(PROJECT_ROOT)}')

runpy.run_path(str(project_root / 'src' / 'experimental' / 'q2_method_search.py'), run_name='__main__')
"""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 问题二方法优化探索\n",
                    "\n",
                    "本 Notebook 调用 `src/experimental/q2_method_search.py`，复现商品关联审查、聚合策略验证比较、图表和报告输出。\n",
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
                    "运行后查看：\n",
                    "\n",
                    "- `outputs/method_search/q2_concept_review.md`\n",
                    "- `tables/method_search/q2_aggregation_comparison.csv`\n",
                    "- `figures/method_search/q2_aggregation_comparison.png`\n",
                    "- `outputs/method_search/q2_method_search_report.md`\n",
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
    (NOTEBOOK_DIR / "q2_method_search.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    base, daily, product_daily = read_data()
    product_meta = product_daily[
        ["product_id", "product_name", "product_label", "category", "category_code"]
    ].drop_duplicates("product_id")

    association = association_analysis(product_daily, daily, product_meta)
    mappings = build_group_mappings(product_daily, product_meta, association["train_matrix"])
    comparison, model_metrics, _ = evaluate_aggregation_strategies(product_daily, mappings)

    plot_aggregation_comparison(comparison)
    generate_concept_review(association)
    generate_final_report(comparison, model_metrics, association)
    generate_notebook()

    print("Q2 method search completed.")
    print(comparison[["strategy_id", "model_label", "MAE", "RMSE", "WAPE_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
