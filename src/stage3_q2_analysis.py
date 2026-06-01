from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics
from src.models import (
    MODEL_SPECS,
    forecast_future_univariate,
    prepare_series_table,
    validate_univariate_models,
)

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")
FUTURE_DATES = pd.date_range("2022-04-01", "2022-04-07", freq="D")

METHOD_LABELS = {
    "product_sum": "单品预测后按类别加总",
    "category_direct": "类别聚合后直接预测",
    "store_product_sum": "门店-单品预测后按门店类别加总",
    "store_category_direct": "门店类别聚合后直接预测",
}


def short_label(text: object, max_len: int = 12) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


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
    df = df.copy()
    for col in ["prediction", "predicted_sales", "predicted_7day_sales"]:
        if col in df.columns:
            df.loc[df[col].abs() < 1e-9, col] = 0.0
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    if to_outputs:
        df.to_csv(OUTPUTS / name, index=False, encoding="utf-8-sig")


def correlation_pair_table(
    matrix: pd.DataFrame, store_id: int | None = None, store_name: str | None = None
) -> pd.DataFrame:
    corr = matrix.corr(method="pearson")
    rows: list[dict] = []
    cols = list(matrix.columns)
    for a, b in combinations(cols, 2):
        s1 = matrix[a]
        s2 = matrix[b]
        row = {
            "product_a": a,
            "product_b": b,
            "corr": float(corr.loc[a, b]),
            "nonzero_both_days": int(((s1 > 0) & (s2 > 0)).sum()),
            "product_a_nonzero_days": int((s1 > 0).sum()),
            "product_b_nonzero_days": int((s2 > 0).sum()),
            "n_days": int(len(matrix)),
        }
        if store_id is not None:
            row = {"store_id": store_id, "store_name": store_name, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def save_heatmap(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    data = corr.values.astype(float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(
        [short_label(c, 14) for c in corr.columns], rotation=50, ha="right", fontsize=8
    )
    ax.set_yticklabels([short_label(c, 14) for c in corr.index], fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if abs(val) < 0.65 else "white",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson 相关系数")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def prepend_result_log(row: str) -> None:
    log_path = ROOT / "RESULT_LOG.md"
    log_text = log_path.read_text(encoding="utf-8")
    if row in log_text:
        return
    lines = log_text.splitlines()
    insert_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("|------"):
            insert_idx = idx + 1
            break
    if insert_idx is not None:
        lines.insert(insert_idx, row)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [TABLES, FIGURES, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140

    df = pd.read_csv(ROOT / "data/processed/modeling_base_table.csv")
    df["date"] = pd.to_datetime(df["date"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)

    product_name_map = (
        df.groupby("product_id")["product_name"]
        .apply(
            lambda s: " / ".join(
                sorted(pd.Series(s.dropna().astype(str).unique()).tolist())
            )
        )
        .to_dict()
    )
    product_category = (
        df.groupby("product_id")["category"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "待确认")
        .to_dict()
    )
    store_name_map = (
        df.groupby("store_id")["store_name"].agg(lambda s: s.dropna().mode().iloc[0]).to_dict()
    )
    product_labels = {
        pid: f"{pid}-{short_label(product_name_map[pid], 8)}"
        for pid in sorted(product_name_map)
    }

    product_daily = df.groupby(["date", "product_id"], as_index=False)[TARGET].sum()
    product_daily["product_name"] = product_daily["product_id"].map(product_name_map)
    product_daily["category"] = product_daily["product_id"].map(product_category)
    product_daily = product_daily[["date", "product_id", "product_name", "category", TARGET]]

    store_product_daily = df.groupby(
        ["date", "store_id", "product_id"], as_index=False
    )[TARGET].sum()
    store_product_daily["store_name"] = store_product_daily["store_id"].map(store_name_map)
    store_product_daily["product_name"] = store_product_daily["product_id"].map(
        product_name_map
    )
    store_product_daily["category"] = store_product_daily["product_id"].map(product_category)
    store_product_daily = store_product_daily[
        ["date", "store_id", "store_name", "product_id", "product_name", "category", TARGET]
    ]

    overall_matrix = (
        product_daily.pivot_table(
            index="date", columns="product_id", values=TARGET, aggfunc="sum", fill_value=0
        )
        .sort_index()
        .rename(columns=product_labels)
    )
    overall_matrix.to_csv(TABLES / "q2_sales_matrix_all_stores.csv", encoding="utf-8-sig")

    for store_id, group in store_product_daily.groupby("store_id"):
        matrix = (
            group.pivot_table(
                index="date", columns="product_id", values=TARGET, aggfunc="sum", fill_value=0
            )
            .sort_index()
            .rename(columns=product_labels)
        )
        matrix.to_csv(
            TABLES / f"q2_sales_matrix_store_{int(store_id)}.csv", encoding="utf-8-sig"
        )

    corr_overall = overall_matrix.corr(method="pearson")
    corr_overall.to_csv(TABLES / "q2_correlation_matrix_all_stores.csv", encoding="utf-8-sig")
    pairs_overall = correlation_pair_table(overall_matrix).sort_values(
        "corr", ascending=False
    )
    pairs_overall = pairs_overall.reset_index(drop=True)
    save_csv(pairs_overall, "q2_product_correlation_pairs_overall.csv", to_outputs=True)

    strong_positive = pairs_overall[pairs_overall["corr"] >= 0.50].copy()
    weak_pairs = pairs_overall[pairs_overall["corr"].abs() <= 0.10].sort_values(
        "corr", key=lambda s: s.abs()
    )
    negative_pairs = pairs_overall[pairs_overall["corr"] < 0].sort_values("corr")
    possible_substitute = negative_pairs[
        (negative_pairs["product_a_nonzero_days"] >= 30)
        & (negative_pairs["product_b_nonzero_days"] >= 30)
    ].copy()
    save_csv(strong_positive, "q2_strong_positive_pairs.csv", to_outputs=True)
    save_csv(weak_pairs, "q2_weak_correlation_pairs.csv")
    save_csv(negative_pairs, "q2_negative_correlation_pairs.csv")
    save_csv(possible_substitute, "q2_possible_substitute_pairs.csv", to_outputs=True)

    store_pair_frames = []
    for store_id, group in store_product_daily.groupby("store_id"):
        matrix = (
            group.pivot_table(
                index="date", columns="product_id", values=TARGET, aggfunc="sum", fill_value=0
            )
            .sort_index()
            .rename(columns=product_labels)
        )
        matrix.corr(method="pearson").to_csv(
            TABLES / f"q2_correlation_matrix_store_{int(store_id)}.csv",
            encoding="utf-8-sig",
        )
        store_pair_frames.append(
            correlation_pair_table(matrix, int(store_id), store_name_map.get(store_id, ""))
        )
    store_pairs = pd.concat(store_pair_frames, ignore_index=True).sort_values(
        ["store_id", "corr"], ascending=[True, False]
    )
    save_csv(store_pairs, "q2_product_correlation_pairs_by_store.csv")

    category_daily = (
        product_daily.groupby(["date", "category"], as_index=False)[TARGET]
        .sum()
        .sort_values(["category", "date"])
    )
    category_stats = (
        category_daily.groupby("category")[TARGET]
        .agg(
            total_sales="sum",
            avg_daily_sales="mean",
            std_daily_sales="std",
            nonzero_days=lambda s: int((s > 0).sum()),
        )
        .reset_index()
    )
    category_stats["sales_share"] = (
        category_stats["total_sales"] / category_stats["total_sales"].sum()
    )
    category_stats["cv"] = category_stats["std_daily_sales"] / category_stats[
        "avg_daily_sales"
    ].replace(0, np.nan)
    category_stats = category_stats.sort_values("total_sales", ascending=False).reset_index(
        drop=True
    )
    save_csv(category_stats, "q2_category_sales_stats.csv", to_outputs=True)

    store_category_daily = (
        store_product_daily.groupby(
            ["date", "store_id", "store_name", "category"], as_index=False
        )[TARGET]
        .sum()
        .sort_values(["store_id", "category", "date"])
    )
    store_category_stats = (
        store_category_daily.groupby(["store_id", "store_name", "category"])[TARGET]
        .agg(
            total_sales="sum",
            avg_daily_sales="mean",
            std_daily_sales="std",
            nonzero_days=lambda s: int((s > 0).sum()),
        )
        .reset_index()
    )
    store_category_stats["cv"] = store_category_stats["std_daily_sales"] / store_category_stats[
        "avg_daily_sales"
    ].replace(0, np.nan)
    store_category_stats = store_category_stats.sort_values(
        ["store_id", "total_sales"], ascending=[True, False]
    ).reset_index(drop=True)
    save_csv(store_category_stats, "q2_store_category_sales_stats.csv")

    category_series = prepare_series_table(category_daily, ["category"], TARGET)
    category_val = validate_univariate_models(
        category_series, ["category"], TARGET, VALIDATION_START
    )
    save_csv(category_val, "q2_category_direct_validation_predictions.csv")

    product_series = prepare_series_table(product_daily, ["product_id", "category"], TARGET)
    product_val = validate_univariate_models(
        product_series, ["product_id", "category"], TARGET, VALIDATION_START
    )
    product_sum_val = product_val.groupby(
        ["date", "category", "model", "model_label"], as_index=False
    ).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
    save_csv(product_sum_val, "q2_product_sum_category_validation_predictions.csv")

    method_rows = []
    for model_name, g in product_sum_val.groupby("model"):
        method_rows.append(
            {
                "method": "product_sum",
                "method_label": METHOD_LABELS["product_sum"],
                "model": model_name,
                "model_label": MODEL_SPECS[model_name].label,
                **calculate_metrics(g["actual"], g["prediction"]),
                "actual_sum": float(g["actual"].sum()),
                "prediction_sum": float(g["prediction"].sum()),
                "n": int(len(g)),
            }
        )
    for model_name, g in category_val.groupby("model"):
        method_rows.append(
            {
                "method": "category_direct",
                "method_label": METHOD_LABELS["category_direct"],
                "model": model_name,
                "model_label": MODEL_SPECS[model_name].label,
                **calculate_metrics(g["actual"], g["prediction"]),
                "actual_sum": float(g["actual"].sum()),
                "prediction_sum": float(g["prediction"].sum()),
                "n": int(len(g)),
            }
        )
    category_method_metrics = pd.DataFrame(method_rows).sort_values(["WAPE", "MAE"])
    category_method_metrics = category_method_metrics.reset_index(drop=True)
    category_method_metrics["WAPE_pct"] = category_method_metrics["WAPE"] * 100
    save_csv(category_method_metrics, "q2_category_method_metrics.csv", to_outputs=True)

    combined_category_val = pd.concat(
        [
            product_sum_val.assign(method="product_sum"),
            category_val.assign(method="category_direct"),
        ],
        ignore_index=True,
        sort=False,
    )
    by_category_rows = []
    for (method, model_name), g in combined_category_val.groupby(["method", "model"]):
        for cat, gg in g.groupby("category"):
            by_category_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "model": model_name,
                    "model_label": MODEL_SPECS[model_name].label,
                    "category": cat,
                    **calculate_metrics(gg["actual"], gg["prediction"]),
                    "actual_sum": float(gg["actual"].sum()),
                    "prediction_sum": float(gg["prediction"].sum()),
                    "n": int(len(gg)),
                }
            )
    category_by_category_metrics = pd.DataFrame(by_category_rows).sort_values(
        ["category", "WAPE"]
    )
    category_by_category_metrics["WAPE_pct"] = category_by_category_metrics["WAPE"] * 100
    save_csv(category_by_category_metrics, "q2_category_method_metrics_by_category.csv")

    store_category_series = prepare_series_table(
        store_category_daily, ["store_id", "store_name", "category"], TARGET
    )
    store_category_val = validate_univariate_models(
        store_category_series, ["store_id", "store_name", "category"], TARGET, VALIDATION_START
    )
    save_csv(store_category_val, "q2_store_category_direct_validation_predictions.csv")

    store_product_series = prepare_series_table(
        store_product_daily, ["store_id", "store_name", "product_id", "category"], TARGET
    )
    store_product_val = validate_univariate_models(
        store_product_series,
        ["store_id", "store_name", "product_id", "category"],
        TARGET,
        VALIDATION_START,
    )
    store_product_sum_val = store_product_val.groupby(
        ["date", "store_id", "store_name", "category", "model", "model_label"],
        as_index=False,
    ).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
    save_csv(
        store_product_sum_val,
        "q2_store_product_sum_store_category_validation_predictions.csv",
    )

    store_cat_method_rows = []
    for model_name, g in store_product_sum_val.groupby("model"):
        store_cat_method_rows.append(
            {
                "method": "store_product_sum",
                "method_label": METHOD_LABELS["store_product_sum"],
                "model": model_name,
                "model_label": MODEL_SPECS[model_name].label,
                **calculate_metrics(g["actual"], g["prediction"]),
                "actual_sum": float(g["actual"].sum()),
                "prediction_sum": float(g["prediction"].sum()),
                "n": int(len(g)),
            }
        )
    for model_name, g in store_category_val.groupby("model"):
        store_cat_method_rows.append(
            {
                "method": "store_category_direct",
                "method_label": METHOD_LABELS["store_category_direct"],
                "model": model_name,
                "model_label": MODEL_SPECS[model_name].label,
                **calculate_metrics(g["actual"], g["prediction"]),
                "actual_sum": float(g["actual"].sum()),
                "prediction_sum": float(g["prediction"].sum()),
                "n": int(len(g)),
            }
        )
    store_category_method_metrics = pd.DataFrame(store_cat_method_rows).sort_values(
        ["WAPE", "MAE"]
    )
    store_category_method_metrics = store_category_method_metrics.reset_index(drop=True)
    store_category_method_metrics["WAPE_pct"] = store_category_method_metrics["WAPE"] * 100
    save_csv(store_category_method_metrics, "q2_store_category_method_metrics.csv", True)

    best_direct_category = category_method_metrics[
        category_method_metrics["method"] == "category_direct"
    ].iloc[0]
    best_product_sum = category_method_metrics[
        category_method_metrics["method"] == "product_sum"
    ].iloc[0]
    best_overall_category = category_method_metrics.iloc[0]

    category_direct_future_daily = forecast_future_univariate(
        category_series,
        ["category"],
        TARGET,
        FUTURE_DATES,
        best_direct_category["model"],
    ).rename(columns={"prediction": "predicted_sales"})
    save_csv(category_direct_future_daily, "q2_category_direct_forecast_daily.csv")
    category_direct_future_total = (
        category_direct_future_daily.groupby(["category", "model", "model_label"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values("predicted_7day_sales", ascending=False)
    )
    save_csv(category_direct_future_total, "q2_category_direct_forecast_7day_total.csv", True)

    product_future_daily = forecast_future_univariate(
        product_series,
        ["product_id", "category"],
        TARGET,
        FUTURE_DATES,
        best_product_sum["model"],
    ).rename(columns={"prediction": "predicted_sales"})
    product_sum_future_daily = product_future_daily.groupby(
        ["date", "category", "model", "model_label"], as_index=False
    )["predicted_sales"].sum()
    save_csv(product_sum_future_daily, "q2_product_sum_category_forecast_daily.csv")
    product_sum_future_total = (
        product_sum_future_daily.groupby(["category", "model", "model_label"], as_index=False)[
            "predicted_sales"
        ]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values("predicted_7day_sales", ascending=False)
    )
    save_csv(product_sum_future_total, "q2_product_sum_category_forecast_7day_total.csv", True)

    future_compare = category_direct_future_total[
        ["category", "model_label", "predicted_7day_sales"]
    ].rename(
        columns={
            "model_label": "direct_model_label",
            "predicted_7day_sales": "direct_category_predicted_7day_sales",
        }
    ).merge(
        product_sum_future_total[["category", "model_label", "predicted_7day_sales"]].rename(
            columns={
                "model_label": "product_sum_model_label",
                "predicted_7day_sales": "product_sum_predicted_7day_sales",
            }
        ),
        on="category",
        how="outer",
    )
    future_compare["difference_direct_minus_product_sum"] = (
        future_compare["direct_category_predicted_7day_sales"]
        - future_compare["product_sum_predicted_7day_sales"]
    )
    save_csv(
        future_compare.sort_values("direct_category_predicted_7day_sales", ascending=False),
        "q2_category_future_method_comparison.csv",
        True,
    )

    if best_overall_category["method"] == "category_direct":
        recommended_category_forecast = category_direct_future_total.copy()
        recommended_category_forecast["recommended_method"] = METHOD_LABELS["category_direct"]
    else:
        recommended_category_forecast = product_sum_future_total.copy()
        recommended_category_forecast["recommended_method"] = METHOD_LABELS["product_sum"]
    save_csv(recommended_category_forecast, "q2_category_forecast_7day_total.csv", True)

    best_direct_store_category = store_category_method_metrics[
        store_category_method_metrics["method"] == "store_category_direct"
    ].iloc[0]
    best_store_overall = store_category_method_metrics.iloc[0]
    store_category_future_daily = forecast_future_univariate(
        store_category_series,
        ["store_id", "store_name", "category"],
        TARGET,
        FUTURE_DATES,
        best_direct_store_category["model"],
    ).rename(columns={"prediction": "predicted_sales"})
    save_csv(store_category_future_daily, "q2_store_category_forecast_daily.csv")
    store_category_future_total = (
        store_category_future_daily.groupby(
            ["store_id", "store_name", "category", "model", "model_label"], as_index=False
        )["predicted_sales"]
        .sum()
        .rename(columns={"predicted_sales": "predicted_7day_sales"})
        .sort_values(["store_id", "predicted_7day_sales"], ascending=[True, False])
    )
    save_csv(store_category_future_total, "q2_store_category_forecast_7day_total.csv", True)

    save_heatmap(
        corr_overall,
        FIGURES / "q2_overall_product_correlation_heatmap.png",
    )
    store_totals = store_product_daily.groupby(["store_id", "store_name"])[TARGET].sum()
    store_totals = store_totals.reset_index().sort_values(TARGET, ascending=False)
    focus_store_id = int(store_totals.iloc[0]["store_id"])
    focus_store_name = store_totals.iloc[0]["store_name"]
    focus_matrix = (
        store_product_daily[store_product_daily["store_id"] == focus_store_id]
        .pivot_table(
            index="date", columns="product_id", values=TARGET, aggfunc="sum", fill_value=0
        )
        .sort_index()
        .rename(columns=product_labels)
    )
    # 探索性附图：用于检查重点门店相关结构，不进入论文正文。
    save_heatmap(
        focus_matrix.corr(method="pearson"),
        FIGURES / "q2_focus_store_product_correlation_heatmap.png",
    )

    # 探索性附图：用于比较类别聚合口径，不进入论文正文。
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_df = category_stats.sort_values("total_sales", ascending=True)
    ax.barh(bar_df["category"], bar_df["total_sales"], color="#3A78B7")
    ax.set_xlabel("累计销量")
    for i, v in enumerate(bar_df["total_sales"]):
        ax.text(v, i, f"{v:.0f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_category_total_sales.png", bbox_inches="tight")
    plt.close(fig)

    trend = category_daily.sort_values(["category", "date"]).copy()
    trend["rolling_7d_sales"] = trend.groupby("category")[TARGET].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    for cat, g in trend.groupby("category"):
        ax.plot(g["date"], g["rolling_7d_sales"], label=cat, linewidth=1.5)
    ax.set_xlabel("日期")
    ax.set_ylabel("7 日滚动平均销量")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_category_daily_trend.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = category_method_metrics.copy()
    plot_df["label"] = plot_df["method_label"] + "\n" + plot_df["model_label"]
    colors = ["#4477AA" if m == "category_direct" else "#66AA55" for m in plot_df["method"]]
    ax.bar(range(len(plot_df)), plot_df["WAPE_pct"], color=colors)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("WAPE (%)")
    for i, v in enumerate(plot_df["WAPE_pct"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_category_method_wape_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 探索性附图：用于比较门店-类别聚合口径，不进入论文正文。
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = store_category_method_metrics.copy()
    plot_df["label"] = plot_df["method_label"] + "\n" + plot_df["model_label"]
    colors = [
        "#CC6677" if m == "store_category_direct" else "#88CCEE"
        for m in plot_df["method"]
    ]
    ax.bar(range(len(plot_df)), plot_df["WAPE_pct"], color=colors)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("WAPE (%)")
    for i, v in enumerate(plot_df["WAPE_pct"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "q2_store_category_method_wape_comparison.png", bbox_inches="tight"
    )
    plt.close(fig)

    # 探索性附图：用于检查门店-类别销量结构，不进入论文正文。
    pivot_sc = store_category_stats.pivot_table(
        index="store_name", columns="category", values="total_sales", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot_sc.values, cmap="YlGnBu")
    ax.set_xticks(range(len(pivot_sc.columns)))
    ax.set_yticks(range(len(pivot_sc.index)))
    ax.set_xticklabels(pivot_sc.columns, rotation=35, ha="right")
    ax.set_yticklabels(pivot_sc.index)
    for i in range(pivot_sc.shape[0]):
        for j in range(pivot_sc.shape[1]):
            ax.text(j, i, f"{pivot_sc.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="累计销量")
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_store_category_sales_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    strong_count = len(strong_positive)
    weak_count = len(weak_pairs)
    neg_count = len(negative_pairs)
    sub_count = len(possible_substitute)

    category_stats_report = category_stats[
        [
            "category",
            "total_sales",
            "sales_share",
            "avg_daily_sales",
            "std_daily_sales",
            "cv",
            "nonzero_days",
        ]
    ].copy()
    category_stats_report["sales_share"] *= 100
    category_stats_report = category_stats_report.rename(
        columns={"sales_share": "sales_share_pct"}
    )
    method_report = category_method_metrics[
        ["method_label", "model_label", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]
    ]
    store_method_report = store_category_method_metrics[
        ["method_label", "model_label", "MAE", "RMSE", "WAPE_pct", "actual_sum", "prediction_sum", "n"]
    ]
    future_report = recommended_category_forecast[
        ["category", "model_label", "predicted_7day_sales", "recommended_method"]
    ]
    direct_future_report = category_direct_future_total[
        ["category", "model_label", "predicted_7day_sales"]
    ]

    report = f"""# 阶段 3：问题二商品关联与类别预测报告

执行日期：2026-05-02

## 1. 问题二核心任务

问题二要求分析同一门店内不同零食销量之间的联系，并在此基础上整合同类零食，预测未来 7 天总销量。本阶段继续沿用阶段 1 和阶段 2 的需求口径：以 `positive_sales` 表示顾客正向购买销量，负数量对应损耗或冲销类调整，不作为正常需求销量。

本阶段的分析分为四步：第一，构造“日期 $\\times$ 商品”的日销量矩阵；第二，计算商品之间的 Pearson 相关系数，识别同步变化或弱相关商品；第三，利用附件中的 `category` 字段按类别聚合销量；第四，比较“单品预测后加总”和“类别聚合后直接预测”两种方法的验证误差，并给出未来 7 天类别预测。

## 2. 商品关联度量

本阶段采用 Pearson 相关系数衡量两个商品日销量的线性同步程度。设 $x_t$ 和 $z_t$ 分别表示两个商品在第 $t$ 天的销量，则 Pearson 相关系数为：

$$
r=\\frac{{\\sum_{{t=1}}^n (x_t-\\bar x)(z_t-\\bar z)}}{{\\sqrt{{\\sum_{{t=1}}^n (x_t-\\bar x)^2}}\\sqrt{{\\sum_{{t=1}}^n (z_t-\\bar z)^2}}}}
$$

当 $r>0$ 时，两个商品销量倾向于同升同降；当 $r<0$ 时，一个商品销量较高时另一个商品销量可能偏低；当 $r$ 接近 0 时，线性同步关系较弱。需要强调：相关性不等于因果性，相关系数只能说明历史销量同步关系，不能证明一种商品导致另一种商品销量变化。

## 3. 类别聚合的稳定性机理

单个商品日销量经常出现零销量、偶然大单或短期波动。按类别聚合后，不同商品的随机波动会相互抵消，类别总销量通常更平滑，因此预测误差可能下降。但聚合也会损失单品差异，例如同属碳酸饮料的不同规格饮料可能有不同价格、陈列和促销响应。若库存需要精确到商品，仍需要保留单品层级预测。

## 4. 与问题一的差异

问题一重点是分别预测门店总销量、商品总销量和门店-商品销量；问题二重点不是单独预测每个商品，而是研究同一门店内商品之间是否存在同步变化，并判断类别聚合是否能改善未来 7 天总销量预测。因此，问题二新增了商品相关性矩阵、类别销量趋势和聚合预测误差比较。

## 5. 销量矩阵构造

本阶段已按全部门店汇总构造 `q2_sales_matrix_all_stores.csv`，并为 7 家门店分别构造 `q2_sales_matrix_store_*.csv`。矩阵的行是日期，列是商品，单元格是该日期对应商品的正向销量。对于 `商品代码=11001020`，按商品编号合并两个海苔名称，避免同一商品因名称录入差异被重复建模。

重点门店选择历史总销量最高的 `{focus_store_name}`，用于展示门店内部商品相关性。

## 6. 相关性分析结果

全部门店汇总层面共计算 {len(pairs_overall)} 对商品相关系数，其中强正相关商品对数量为 {strong_count}，弱相关商品对数量为 {weak_count}，负相关商品对数量为 {neg_count}。强正相关阈值取 $r\\ge 0.50$，弱相关阈值取 $|r|\\le 0.10$。

强正相关商品对示例：

{df_to_md(strong_positive.head(10)[["product_a", "product_b", "corr", "nonzero_both_days"]], 10)}

弱相关商品对示例：

{df_to_md(weak_pairs.head(10)[["product_a", "product_b", "corr", "nonzero_both_days"]], 10)}

负相关商品对可以作为替代关系线索，但不能直接认定为替代关系。当前筛选到 {sub_count} 对同时具有一定非零销售天数且相关系数为负的商品对，示例如下：

{df_to_md(possible_substitute.head(10)[["product_a", "product_b", "corr", "nonzero_both_days"]], 10)}

这些负相关候选的相关系数绝对值均很小，属于极弱反向关系。因此，本阶段不能支持“存在明确替代关系”的结论，只能写为“未发现强替代关系，少数商品对存在很弱的反向波动线索”。

### 6.1 商品关联结论的证据分级

为避免把相关性过度解释为因果性，本阶段将商品关联结论分为三类。

第一类是数据可以直接支持的结论：

1. 全部门店汇总层面共计算 66 对商品相关系数，其中 2 对商品满足 $r\\ge 0.50$，对应表述为“存在较强正相关”。
2. `三养辣火鸡味拌面` 与 `乡吧哥蜜汁鸡翅` 的相关系数为 0.521，`三养辣火鸡味拌面` 与 `李子柒螺蛳粉335g` 的相关系数为 0.506，对应表述为“历史日销量同步波动较明显”。
3. 有 25 对商品满足 $|r|\\le 0.10$，对应表述为“线性同步关系较弱”。
4. 筛选出的负相关候选相关系数绝对值均很小，例如 -0.025 和 -0.018，对应表述为“未发现强替代关系”。

第二类是合理推断但不确定的结论：

1. 强正相关商品可能共同受到客流量、消费场景、陈列位置或促销节奏影响，但当前数据不能区分具体原因。
2. 速食类商品与包装散称商品之间的正相关，可能反映部分消费场景相近，但仍需要购物篮明细或顾客层面数据才能验证。
3. 弱相关商品在补货和陈列上可以暂时按相对独立商品处理，但这只是经营上的简化假设，不是数据已经证明“完全无关”。

第三类是不能写进论文的过度解释：

1. 不能写“某商品销量上升导致另一商品销量上升”，因为相关系数不能证明因果关系。
2. 不能写“负相关商品就是替代品”，因为负相关系数绝对值很小，且没有顾客选择过程数据。
3. 不能写“强正相关商品一定应捆绑销售或必须相邻陈列”，因为当前只分析了日销量同步，没有验证陈列或组合购买效果。
4. 不能把相关性直接解释为顾客偏好、消费心理或购买动机，除非后续有额外调查或购物篮数据支持。

图7展示全部门店汇总后的商品相关性热力图，图8展示重点门店 `{focus_store_name}` 的商品相关性热力图。两张图用于证明：商品销量之间确实存在不同程度的同步变化，但这种同步关系主要是统计联系，不能直接写成因果关系或确定的替代关系。

## 7. 同类零食整合

附件一中存在 `category` 字段，因此本阶段直接使用原始类别字段进行同类零食整合，不再根据商品名称主观重新分类。类别历史销量统计如下：

{df_to_md(category_stats_report, 20)}

图9展示各类别历史累计销量，用于说明类别之间需求规模差异；图10展示类别销量 7 日滚动均值趋势，用于观察类别层面的趋势和波动；图13展示门店-类别历史销量热力图，用于说明不同门店的品类结构差异。

## 8. 类别预测模型与误差比较

本阶段比较两种策略：

1. 单品预测后按类别加总：先对每个商品建立时间序列预测，再按类别求和。
2. 类别聚合后直接预测：先把同类商品聚合为类别销量，再对类别总销量建模。

两种策略均使用问题一已经解释过的三个模型：移动平均、同星期均值和简单指数平滑。验证期仍为 2022-03-01 至 2022-03-31，每个验证日只使用该日之前的历史销量，避免时间序列泄露。

整体类别层面的误差比较如下：

{df_to_md(method_report, 20)}

从 WAPE 看，类别层面验证最优策略为 `{best_overall_category["method_label"]}` + `{best_overall_category["model_label"]}`，WAPE 为 {best_overall_category["WAPE"]*100:.2f}%。其中，类别聚合后直接预测的最优模型为 `{best_direct_category["model_label"]}`，WAPE 为 {best_direct_category["WAPE"]*100:.2f}%；单品预测后加总的最优模型为 `{best_product_sum["model_label"]}`，WAPE 为 {best_product_sum["WAPE"]*100:.2f}%。

门店-类别层面的误差比较如下：

{df_to_md(store_method_report, 20)}

门店-类别层面验证最优策略为 `{best_store_overall["method_label"]}` + `{best_store_overall["model_label"]}`，WAPE 为 {best_store_overall["WAPE"]*100:.2f}%。类别聚合后直接预测的门店-类别最优模型为 `{best_direct_store_category["model_label"]}`，WAPE 为 {best_direct_store_category["WAPE"]*100:.2f}%。

## 9. 未来 7 天类别预测结果

未来 7 天日期范围仍为 2022-04-01 至 2022-04-07。本阶段主推荐类别预测结果采用验证期 WAPE 最小的策略。结果如下：

{df_to_md(future_report, 20)}

同时，类别聚合后直接预测结果已单独保存，便于论文讨论聚合模型：

{df_to_md(direct_future_report, 20)}

门店-类别未来 7 天预测结果已保存至 `tables/q2_store_category_forecast_7day_total.csv` 和 `outputs/q2_store_category_forecast_7day_total.csv`，可作为后续问题四中门店库存分配的中间依据。

## 10. 聚合效果解释

1. 聚合可能降低随机波动：单个商品的偶然零销量或偶然大单在类别求和后会被部分抵消，类别销量序列通常更稳定。
2. 聚合可能损失单品差异：同一类别内部商品仍可能有不同价格、规格、促销响应和消费场景，聚合后这些差异被隐藏。
3. 聚合模型更好的情况：当单品销量稀疏、零值多、商品之间需求方向相近时，先聚合再预测往往更稳。
4. 单品模型更好的情况：当类别内部商品差异明显，或者最终决策必须精确到单个商品库存时，单品预测更有解释价值。

## 11. 本阶段限制

1. 相关性分析只说明历史同步变化，不能证明因果关系。
2. 替代关系只根据负相关进行初步识别，当前数据无法直接观察顾客选择过程，因此不能确认真实替代行为。
3. 本阶段仍未引入天气、节假日、促销活动等外部变量，这些因素将在问题三分析。
4. 未来 7 天天气和活动日没有附件真实观测，因此问题二仍主要使用历史销量自身规律。
"""
    (OUTPUTS / "stage3_q2_relationship_report.md").write_text(report, encoding="utf-8")

    q2_model_building = f"""## 问题二模型建立

问题二关注同一门店中不同零食销量之间的联系，并要求整合同类零食预测未来 7 天总销量。设 $y_{{t,p}}$ 表示第 $t$ 天商品 $p$ 的正向销量，$c(p)$ 表示商品所属类别。

### 商品销量矩阵

对每个门店构造日期 $\\times$ 商品的日销量矩阵：

$$
Y_s=(y_{{t,p}}),\\quad t=1,2,\\ldots,n,\\ p=1,2,\\ldots,m
$$

其中 $s$ 表示门店，矩阵中每一列是一种商品的日销量序列。该矩阵用于计算商品之间的同步波动关系。

### 商品相关性模型

对任意两个商品 $p$ 和 $q$，使用 Pearson 相关系数衡量其日销量同步程度：

$$
r_{{pq}}=\\frac{{\\sum_{{t=1}}^n (y_{{t,p}}-\\bar y_p)(y_{{t,q}}-\\bar y_q)}}{{\\sqrt{{\\sum_{{t=1}}^n (y_{{t,p}}-\\bar y_p)^2}}\\sqrt{{\\sum_{{t=1}}^n (y_{{t,q}}-\\bar y_q)^2}}}}
$$

$r_{{pq}}>0$ 表示两个商品销量倾向于同向波动，$r_{{pq}}<0$ 表示存在反向波动线索，$r_{{pq}}$ 接近 0 表示线性同步关系弱。该指标只表示相关性，不表示因果性。

### 类别聚合模型

由于附件中包含商品类别字段，本文直接按 `category` 聚合同类零食。类别 $g$ 在第 $t$ 天的销量定义为：

$$
Y_{{t,g}}=\\sum_{{p:c(p)=g}} y_{{t,p}}
$$

类别聚合后，分别比较两种预测策略：一是先对商品 $p$ 预测 $\\hat y_{{t,p}}$，再按类别加总；二是先得到类别销量 $Y_{{t,g}}$，再直接预测 $\\hat Y_{{t,g}}$。

### 类别预测模型

类别预测继续使用移动平均、同星期均值和简单指数平滑三个可解释模型。保留这三个模型的原因是：问题二的重点是比较聚合方式，而不是堆叠复杂模型；同一组简单模型有利于观察“聚合本身”是否改善预测稳定性。
"""

    q2_model_solution = f"""## 问题二模型求解

本文以 2020-04-01 至 2022-02-28 为训练期，以 2022-03-01 至 2022-03-31 为验证期。验证时每个目标日只使用该日之前的历史销量，避免时间序列数据泄露。

首先，按全部门店汇总和单门店分别构造日期 $\\times$ 商品销量矩阵，并计算商品相关系数矩阵。其次，按类别聚合销量，得到类别日销量序列和门店-类别日销量序列。最后，比较“单品预测后按类别加总”和“类别聚合后直接预测”两种策略。

类别层面验证误差如下：

{df_to_md(method_report, 20)}

由表可知，类别层面验证最优策略为 `{best_overall_category["method_label"]}` + `{best_overall_category["model_label"]}`，WAPE 为 {best_overall_category["WAPE"]*100:.2f}%。类别聚合后直接预测的最优模型为 `{best_direct_category["model_label"]}`，WAPE 为 {best_direct_category["WAPE"]*100:.2f}%。

门店-类别层面验证误差如下：

{df_to_md(store_method_report, 20)}

据此，对 2022-04-01 至 2022-04-07 进行类别销量预测，并保存类别预测表和门店-类别预测表。
"""

    q2_result_analysis = f"""## 问题二结果分析

商品相关性分析表明，全部门店汇总层面共得到 {strong_count} 对强正相关商品对和 {weak_count} 对弱相关商品对。强正相关商品对说明这些商品在历史数据中存在同步销售现象，可能共同受到客流量、消费场景或促销陈列影响；但相关性不等于因果性，不能据此认定商品之间存在直接带动关系。

负相关商品对可以作为替代关系的初步线索，但本题数据只记录销售结果，未记录顾客在多个商品之间的选择过程，并且本阶段筛选出的负相关系数绝对值很小。因此，当前数据无法确认真实替代行为，论文中应表述为“未发现强替代关系”。

为保证论文表述克制，商品关联结论按证据强度分为三类。第一，数据可以直接支持的结论包括：全部门店汇总层面共有 2 对强正相关商品对，25 对弱相关商品对；`三养辣火鸡味拌面` 与 `乡吧哥蜜汁鸡翅`、`李子柒螺蛳粉335g` 的历史日销量同步波动较明显；负相关候选的相关系数绝对值很小，因此未发现强替代关系。第二，合理但不确定的推断包括：强正相关商品可能共同受到客流量、消费场景、陈列位置或促销节奏影响；速食类商品与包装散称商品之间的正相关可能反映消费场景相近，但仍需购物篮数据验证。第三，不能写进论文的过度解释包括：不能把相关性写成因果性，不能把弱负相关直接写成替代关系，不能据此断言商品必须捆绑销售或相邻陈列，也不能直接解释为顾客购买动机。

从类别销量看，历史销量最高的类别为 `{category_stats.iloc[0]["category"]}`，累计销量为 {category_stats.iloc[0]["total_sales"]:.0f}。类别聚合能够降低单品层面的随机波动，但也会掩盖同一类别内部商品的差异。因此，类别预测适合用于品类级备货和趋势判断，单品预测仍适合用于具体商品补货。

未来 7 天推荐类别预测结果如下：

{df_to_md(future_report, 20)}

类别聚合后直接预测结果如下：

{df_to_md(direct_future_report, 20)}

本阶段结果为后续问题三、问题四提供两个基础：第一，商品之间确实存在不同程度的同步变化，可以在综合模型中考虑品类结构；第二，类别聚合预测与单品加总预测的误差不同，说明预测粒度会影响模型稳定性。
"""

    row = (
        f"| 2026-05-02 | 阶段 3 问题二商品关联与类别预测 | "
        f"processed: modeling_base_table.csv | 相关系数矩阵、移动平均、同星期均值、简单指数平滑 | "
        f"正向日销量、商品、门店、类别 | 类别最优 WAPE={best_overall_category['WAPE']*100:.2f}%；"
        f"门店-类别最优 WAPE={best_store_overall['WAPE']*100:.2f}% | "
        f"已完成商品相关性分析、类别聚合、单品加总与类别直接预测比较；"
        f"未来 7 天推荐类别预测合计 {recommended_category_forecast['predicted_7day_sales'].sum():.3f} | "
        f"相关性不等于因果性；替代关系只能作为线索；类别聚合会损失单品差异 | "
        f"停止在阶段 3，等待确认后进入阶段 4 问题三建模 |"
    )
    # RESULT_LOG.md is maintained as a concise project-level summary.

    summary = {
        "rows_modeling_base": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_stores": int(df["store_id"].nunique()),
        "n_products_by_id": int(df["product_id"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "focus_store_id": focus_store_id,
        "focus_store_name": focus_store_name,
        "strong_positive_pair_count": int(strong_count),
        "weak_pair_count_abs_le_0_10": int(weak_count),
        "negative_pair_count": int(neg_count),
        "possible_substitute_pair_count": int(sub_count),
        "best_category_method": str(best_overall_category["method_label"]),
        "best_category_model": str(best_overall_category["model_label"]),
        "best_category_wape": float(best_overall_category["WAPE"]),
        "best_direct_category_model": str(best_direct_category["model_label"]),
        "best_direct_category_wape": float(best_direct_category["WAPE"]),
        "best_product_sum_model": str(best_product_sum["model_label"]),
        "best_product_sum_wape": float(best_product_sum["WAPE"]),
        "best_store_category_method": str(best_store_overall["method_label"]),
        "best_store_category_model": str(best_store_overall["model_label"]),
        "best_store_category_wape": float(best_store_overall["WAPE"]),
        "recommended_category_forecast_7day_sum": float(
            recommended_category_forecast["predicted_7day_sales"].sum()
        ),
        "direct_category_forecast_7day_sum": float(
            category_direct_future_total["predicted_7day_sales"].sum()
        ),
        "store_category_forecast_7day_sum": float(
            store_category_future_total["predicted_7day_sales"].sum()
        ),
    }
    (OUTPUTS / "stage3_q2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
