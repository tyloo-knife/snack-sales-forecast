from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"

TARGET_MODEL = "ridge"
TARGET_MODEL_LABEL = "综合Ridge回归"
VALIDATION_SOURCE_CANDIDATES = [
    TABLES / "q4_recursive_7day_validation.csv",
    OUTPUTS / "q4_recursive_7day_validation.csv",
    TABLES / "q4_validation_predictions_store_product.csv",
    OUTPUTS / "q4_validation_predictions_store_product.csv",
]

TABLE_OUTPUTS = {
    "store": TABLES / "q4_error_by_store.csv",
    "product": TABLES / "q4_error_by_product.csv",
    "category": TABLES / "q4_error_by_category.csv",
    "calendar": TABLES / "q4_error_by_calendar_type.csv",
    "volume": TABLES / "q4_error_by_sales_volume_group.csv",
}

FIGURE_OUTPUTS = {
    "store": FIGURES / "q4_error_by_store_bar.png",
    "product": FIGURES / "q4_error_top_products.png",
    "category": FIGURES / "q4_error_by_category_bar.png",
    "scatter": FIGURES / "q4_actual_vs_predicted_scatter.png",
    "calendar": FIGURES / "q4_error_by_calendar_type_bar.png",
}


def backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.{stamp}.backup")
    backup.write_bytes(path.read_bytes())


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    if df is None or df.empty:
        return "无。"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
    columns = list(out.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append(
            "| "
            + " | ".join("" if pd.isna(row[col]) else str(row[col]) for col in columns)
            + " |"
        )
    return "\n".join(lines)


def load_validation_predictions() -> tuple[pd.DataFrame, Path, str]:
    source = next((path for path in VALIDATION_SOURCE_CANDIDATES if path.exists()), None)
    if source is None:
        candidates = "\n".join(str(path) for path in VALIDATION_SOURCE_CANDIDATES)
        raise FileNotFoundError(f"未找到问题四验证集预测表，候选路径：\n{candidates}")

    preds = pd.read_csv(source)
    required = {
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
    }
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"{source} 缺少必要字段：{sorted(missing)}")

    preds["date"] = pd.to_datetime(preds["date"])
    model_preds = preds[preds["model"] == TARGET_MODEL].copy()
    if model_preds.empty:
        available = ", ".join(sorted(preds["model"].dropna().astype(str).unique()))
        raise ValueError(f"{source} 中找不到模型 {TARGET_MODEL}；已有模型：{available}")

    mode = (
        str(model_preds["validation_mode"].dropna().iloc[0])
        if "validation_mode" in model_preds.columns
        and model_preds["validation_mode"].notna().any()
        else "daily_rolling_one_step"
    )
    model_preds["actual"] = pd.to_numeric(model_preds["actual"], errors="coerce").fillna(0.0)
    model_preds["prediction"] = (
        pd.to_numeric(model_preds["prediction"], errors="coerce").fillna(0.0).clip(lower=0)
    )
    model_preds["abs_error"] = (model_preds["actual"] - model_preds["prediction"]).abs()
    model_preds["squared_error"] = (model_preds["actual"] - model_preds["prediction"]) ** 2
    return model_preds, source, mode


def load_calendar_features() -> pd.DataFrame:
    feature_path = TABLES / "q4_modeling_feature_table.csv"
    fallback_path = ROOT / "data/processed/modeling_base_table.csv"
    path = feature_path if feature_path.exists() else fallback_path
    if not path.exists():
        raise FileNotFoundError("未找到可合并日历字段的 q4_modeling_feature_table 或 modeling_base_table。")

    calendar = pd.read_csv(
        path,
        usecols=lambda col: col
        in {"date", "is_weekend", "is_holiday", "is_activity_day", "weekday"},
    )
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.drop_duplicates("date").copy()
    for col in ["is_weekend", "is_holiday", "is_activity_day"]:
        calendar[col] = pd.to_numeric(calendar[col], errors="coerce").fillna(0).astype(int)
    if "weekday" in calendar.columns:
        calendar["weekday"] = pd.to_numeric(calendar["weekday"], errors="coerce")
    return calendar


def metrics_for_frame(df: pd.DataFrame) -> dict:
    metrics = calculate_metrics(df["actual"], df["prediction"])
    return {
        **metrics,
        "WAPE_pct": metrics["WAPE"] * 100 if pd.notna(metrics["WAPE"]) else np.nan,
        "actual_sum": float(df["actual"].sum()),
        "prediction_sum": float(df["prediction"].sum()),
        "abs_error_sum": float((df["actual"] - df["prediction"]).abs().sum()),
        "bias_sum": float((df["prediction"] - df["actual"]).sum()),
        "n": int(len(df)),
    }


def aggregate_daily(preds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        preds.groupby(["date", *group_cols], dropna=False, as_index=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(["date", *group_cols])
    )


def grouped_metrics(
    preds: pd.DataFrame,
    group_cols: list[str],
    sort_by: str = "WAPE",
) -> pd.DataFrame:
    rows = []
    for keys, group in preds.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metrics_for_frame(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values([sort_by, "MAE", "RMSE"], ascending=[False, False, False]).reset_index(
        drop=True
    )


def build_entity_error_tables(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store_daily = aggregate_daily(preds, ["store_id", "store_name"])
    product_daily = aggregate_daily(preds, ["product_id", "product_name", "category"])
    category_daily = aggregate_daily(preds, ["category"])

    by_store = grouped_metrics(store_daily, ["store_id", "store_name"])
    by_product = grouped_metrics(product_daily, ["product_id", "product_name", "category"])
    by_category = grouped_metrics(category_daily, ["category"])

    by_store.insert(0, "diagnosis_unit", "store_daily_total")
    by_product.insert(0, "diagnosis_unit", "product_daily_total")
    by_category.insert(0, "diagnosis_unit", "category_daily_total")
    return by_store, by_product, by_category


def build_calendar_error_table(preds: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        ("工作日", "非周末且非节假日", (preds["is_weekend"] == 0) & (preds["is_holiday"] == 0)),
        ("周末", "is_weekend = 1", preds["is_weekend"] == 1),
        ("节假日", "is_holiday = 1", preds["is_holiday"] == 1),
        ("活动日", "is_activity_day = 1", preds["is_activity_day"] == 1),
    ]
    rows = []
    for label, definition, mask in conditions:
        subset = preds.loc[mask].copy()
        if subset.empty:
            row = {
                "calendar_type": label,
                "definition": definition,
                "MAE": np.nan,
                "RMSE": np.nan,
                "WAPE": np.nan,
                "WAPE_pct": np.nan,
                "actual_sum": 0.0,
                "prediction_sum": 0.0,
                "abs_error_sum": 0.0,
                "bias_sum": 0.0,
                "n": 0,
                "date_count": 0,
            }
        else:
            daily = subset.groupby("date", as_index=False).agg(
                actual=("actual", "sum"), prediction=("prediction", "sum")
            )
            row = {
                "calendar_type": label,
                "definition": definition,
                **metrics_for_frame(daily),
                "date_count": int(daily["date"].nunique()),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def build_volume_error_table(preds: pd.DataFrame) -> pd.DataFrame:
    product_volume = (
        preds.groupby(["product_id", "product_name", "category"], as_index=False)["actual"]
        .sum()
        .rename(columns={"actual": "validation_actual_sum"})
    )
    positive_volume = product_volume[product_volume["validation_actual_sum"] > 0].copy()
    if positive_volume.empty:
        product_volume["sales_volume_group"] = "无正销量商品"
    else:
        median_volume = float(positive_volume["validation_actual_sum"].median())
        product_volume["sales_volume_group"] = np.where(
            product_volume["validation_actual_sum"] >= median_volume,
            "高销量商品",
            "低销量商品",
        )

    labeled = preds.merge(
        product_volume[
            ["product_id", "product_name", "category", "validation_actual_sum", "sales_volume_group"]
        ],
        on=["product_id", "product_name", "category"],
        how="left",
    )
    daily = aggregate_daily(labeled, ["sales_volume_group"])
    rows = []
    for group, frame in daily.groupby("sales_volume_group", dropna=False):
        products = product_volume[product_volume["sales_volume_group"] == group]
        row = {
            "sales_volume_group": group,
            **metrics_for_frame(frame),
            "product_count": int(len(products)),
            "product_actual_sum_min": float(products["validation_actual_sum"].min()),
            "product_actual_sum_max": float(products["validation_actual_sum"].max()),
            "product_actual_sum_median": float(products["validation_actual_sum"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sales_volume_group").reset_index(drop=True)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def plot_error_tables(
    by_store: pd.DataFrame,
    by_product: pd.DataFrame,
    by_category: pd.DataFrame,
    by_calendar: pd.DataFrame,
    preds: pd.DataFrame,
) -> None:
    setup_plot_style()

    store_plot = by_store.sort_values("WAPE_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(store_plot["store_name"], store_plot["WAPE_pct"], color="#4C78A8")
    ax.set_title("问题四各门店验证误差（WAPE）")
    ax.set_xlabel("WAPE (%)")
    ax.set_ylabel("门店")
    for idx, row in enumerate(store_plot.itertuples()):
        if pd.notna(row.WAPE_pct):
            ax.text(row.WAPE_pct, idx, f"{row.WAPE_pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIGURE_OUTPUTS["store"])

    product_plot = by_product.dropna(subset=["WAPE_pct"]).head(10).sort_values(
        "WAPE_pct", ascending=True
    )
    labels = product_plot["product_name"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, product_plot["WAPE_pct"], color="#F58518")
    ax.set_title("问题四商品误差 Top 10（按 WAPE 从高到低选取）")
    ax.set_xlabel("WAPE (%)")
    ax.set_ylabel("商品")
    for idx, row in enumerate(product_plot.itertuples()):
        ax.text(row.WAPE_pct, idx, f"{row.WAPE_pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIGURE_OUTPUTS["product"])

    category_plot = by_category.sort_values("WAPE_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(category_plot["category"], category_plot["WAPE_pct"], color="#54A24B")
    ax.set_title("问题四各类别验证误差（WAPE）")
    ax.set_xlabel("WAPE (%)")
    ax.set_ylabel("类别")
    for idx, row in enumerate(category_plot.itertuples()):
        if pd.notna(row.WAPE_pct):
            ax.text(row.WAPE_pct, idx, f"{row.WAPE_pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIGURE_OUTPUTS["category"])

    scatter = aggregate_daily(preds, ["store_id", "store_name", "product_id", "product_name"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(scatter["actual"], scatter["prediction"], alpha=0.45, s=18, color="#4C78A8")
    max_value = float(max(scatter["actual"].max(), scatter["prediction"].max()))
    ax.plot([0, max_value], [0, max_value], color="#D62728", linewidth=1.2, label="真实值=预测值")
    ax.set_title("问题四验证集真实值 vs 预测值")
    ax.set_xlabel("真实销量")
    ax.set_ylabel("预测销量")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, FIGURE_OUTPUTS["scatter"])

    calendar_plot = by_calendar.dropna(subset=["WAPE_pct"]).sort_values("WAPE_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(calendar_plot["calendar_type"], calendar_plot["WAPE_pct"], color="#B279A2")
    ax.set_title("问题四日历场景验证误差（WAPE）")
    ax.set_xlabel("WAPE (%)")
    ax.set_ylabel("日历类型")
    for idx, row in enumerate(calendar_plot.itertuples()):
        ax.text(row.WAPE_pct, idx, f"{row.WAPE_pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIGURE_OUTPUTS["calendar"])


def first_nonempty(df: pd.DataFrame, sort_col: str, ascending: bool) -> pd.Series | None:
    valid = df.dropna(subset=[sort_col])
    if valid.empty:
        return None
    return valid.sort_values(sort_col, ascending=ascending).iloc[0]


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "无可计算样本"
    return f"{value:.2f}%"


def build_report(
    preds: pd.DataFrame,
    source: Path,
    validation_mode: str,
    by_store: pd.DataFrame,
    by_product: pd.DataFrame,
    by_category: pd.DataFrame,
    by_calendar: pd.DataFrame,
    by_volume: pd.DataFrame,
) -> str:
    overall = metrics_for_frame(aggregate_daily(preds, ["store_id", "product_id"]))
    best_store = first_nonempty(by_store, "WAPE_pct", True)
    worst_store = first_nonempty(by_store, "WAPE_pct", False)
    best_category = first_nonempty(by_category, "WAPE_pct", True)
    worst_category = first_nonempty(by_category, "WAPE_pct", False)
    best_calendar = first_nonempty(by_calendar, "WAPE_pct", True)
    worst_calendar = first_nonempty(by_calendar, "WAPE_pct", False)
    high_low = by_volume.set_index("sales_volume_group") if not by_volume.empty else pd.DataFrame()

    no_sample_calendar = by_calendar.loc[by_calendar["n"] == 0, "calendar_type"].tolist()
    limited_calendar = by_calendar.loc[
        (by_calendar["n"] > 0) & (by_calendar["date_count"] < 3),
        ["calendar_type", "date_count"],
    ]
    zero_actual_products = by_product.loc[
        by_product["actual_sum"] == 0, ["product_name", "category", "prediction_sum"]
    ]
    zero_actual_categories = by_category.loc[
        by_category["actual_sum"] == 0, ["category", "prediction_sum"]
    ]

    reliable_parts = []
    unreliable_parts = []
    if best_store is not None:
        reliable_parts.append(
            f"门店 `{best_store['store_name']}` 的 WAPE 最低，为 {format_pct(best_store['WAPE_pct'])}"
        )
    if best_category is not None:
        reliable_parts.append(
            f"类别 `{best_category['category']}` 的 WAPE 最低，为 {format_pct(best_category['WAPE_pct'])}"
        )
    if "高销量商品" in high_low.index and "低销量商品" in high_low.index:
        high_wape = float(high_low.loc["高销量商品", "WAPE_pct"])
        low_wape = float(high_low.loc["低销量商品", "WAPE_pct"])
        if high_wape <= low_wape:
            reliable_parts.append(
                f"高销量商品组 WAPE={high_wape:.2f}%，低于低销量商品组的 {low_wape:.2f}%"
            )
        else:
            unreliable_parts.append(
                f"高销量商品组 WAPE={high_wape:.2f}%，高于低销量商品组的 {low_wape:.2f}%"
            )
    if worst_store is not None:
        unreliable_parts.append(
            f"门店 `{worst_store['store_name']}` 的 WAPE 最高，为 {format_pct(worst_store['WAPE_pct'])}"
        )
    if worst_category is not None:
        unreliable_parts.append(
            f"类别 `{worst_category['category']}` 的 WAPE 最高，为 {format_pct(worst_category['WAPE_pct'])}"
        )
    if worst_calendar is not None:
        calendar_suffix = (
            f"，但样本仅 {int(worst_calendar['date_count'])} 天"
            if int(worst_calendar["date_count"]) < 3
            else ""
        )
        unreliable_parts.append(
            f"日历场景 `{worst_calendar['calendar_type']}` 的 WAPE 最高，为 {format_pct(worst_calendar['WAPE_pct'])}{calendar_suffix}"
        )

    reliable_text = "；".join(reliable_parts) + "。"
    unreliable_text = "；".join(unreliable_parts)
    if unreliable_text:
        unreliable_text += "。"
    if no_sample_calendar:
        unreliable_text += (
            "验证集中没有可用于评价的日历类型："
            + "、".join(no_sample_calendar)
            + "，不能据此判断模型在这些场景下的可靠性。"
        )
    if not limited_calendar.empty:
        limited_text = "、".join(
            f"{row.calendar_type}（{int(row.date_count)} 天）"
            for row in limited_calendar.itertuples()
        )
        unreliable_text += f"验证样本较少的日历类型包括：{limited_text}，相关结论只能作为提示。"
    if not zero_actual_products.empty:
        zero_product_text = "、".join(
            f"{row.product_name}（预测合计 {row.prediction_sum:.2f}）"
            for row in zero_actual_products.itertuples()
        )
        unreliable_text += (
            "验证期真实销量为 0 的商品无法计算 WAPE，但若模型给出正预测值，仍说明存在误判风险："
            + zero_product_text
            + "。"
        )
    if not zero_actual_categories.empty:
        zero_category_text = "、".join(
            f"{row.category}（预测合计 {row.prediction_sum:.2f}）"
            for row in zero_actual_categories.itertuples()
        )
        unreliable_text += (
            "验证期真实销量为 0 的类别无法用 WAPE 衡量，需单独说明："
            + zero_category_text
            + "。"
        )

    date_min = preds["date"].min().date().isoformat()
    date_max = preds["date"].max().date().isoformat()
    source_display = source.relative_to(ROOT).as_posix()

    return f"""# 问题四误差诊断报告

执行日期：2026-05-03

## 1. 诊断口径

本报告基于 `{source_display}` 中的验证集预测结果，筛选模型 `{TARGET_MODEL_LABEL}`（`model={TARGET_MODEL}`）进行诊断。验证日期范围为 {date_min} 至 {date_max}，验证口径为 `{validation_mode}`。

本报告使用三类误差指标：

- MAE：平均绝对误差，表示平均每个诊断单位相差多少销量。
- RMSE：均方根误差，对极端大误差更敏感。
- WAPE：绝对误差总和除以真实销量总和，适合销量中存在大量零值和低值的情况。

整体门店-商品日粒度误差为：MAE={overall['MAE']:.3f}，RMSE={overall['RMSE']:.3f}，WAPE={overall['WAPE_pct']:.2f}%。

## 2. 分门店误差

统计口径：先按“日期-门店”汇总所有商品真实销量与预测销量，再计算各门店 MAE、RMSE、WAPE。

{df_to_md(by_store[['store_id', 'store_name', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'n']], 20)}

## 3. 分商品误差

统计口径：先按“日期-商品”汇总所有门店真实销量与预测销量，再计算各商品 MAE、RMSE、WAPE。下表按 WAPE 从高到低排列。

{df_to_md(by_product[['product_id', 'product_name', 'category', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'n']], 20)}

## 4. 分类别误差

统计口径：先按“日期-类别”汇总所有门店和商品真实销量与预测销量，再计算各类别 MAE、RMSE、WAPE。

{df_to_md(by_category[['category', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'n']], 20)}

## 5. 日历场景误差

统计口径：工作日定义为非周末且非节假日；周末、节假日、活动日按附件二字段分别筛选。节假日、活动日可能与周末存在重叠，因此这里是场景诊断，不是互斥拆分。

{df_to_md(by_calendar[['calendar_type', 'definition', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'date_count']], 20)}

## 6. 高销量与低销量商品误差差异

统计口径：按验证集内商品真实销量总量的中位数划分高销量商品与低销量商品，再按“日期-销量组”汇总真实销量与预测销量后计算误差。该分组只用于误差诊断，不作为训练特征。

{df_to_md(by_volume[['sales_volume_group', 'product_count', 'MAE', 'RMSE', 'WAPE_pct', 'actual_sum', 'prediction_sum', 'product_actual_sum_min', 'product_actual_sum_max']], 10)}

## 7. 图表文件

- `figures/q4_error_by_store_bar.png`：各门店 WAPE 柱状图。
- `figures/q4_error_top_products.png`：商品误差 Top 10。
- `figures/q4_error_by_category_bar.png`：各类别 WAPE 对比图。
- `figures/q4_actual_vs_predicted_scatter.png`：真实值 vs 预测值散点图。
- `figures/q4_error_by_calendar_type_bar.png`：日历场景 WAPE 对比图。

## 8. 可靠性判断

相对可靠场景：{reliable_text}

相对不可靠场景：{unreliable_text}

需要强调的是，WAPE 高并不一定表示模型完全失效。对于低销量商品，只要真实销量总量很小，少量绝对误差也会造成很高的百分比误差。因此论文中应同时报告 MAE、RMSE、WAPE，并结合真实销量规模解释。

## 9. 论文写法建议

可写为：模型在销量规模较高、历史销售较稳定的商品或类别上误差相对较低，说明历史销量滞后项和滚动均值能够捕捉主要短期需求水平；但在低销量商品、部分门店和样本不足的日历场景下，误差明显放大，说明模型对稀疏需求和特殊日期的刻画能力有限。节假日或活动日若在验证集中样本不足，不能直接宣称模型在这些场景下可靠。
"""


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
        backup_if_exists(path)
        lines.insert(insert_idx, row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [TABLES, FIGURES, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)

    preds, source, validation_mode = load_validation_predictions()
    calendar = load_calendar_features()
    preds = preds.merge(calendar, on="date", how="left")
    for col in ["is_weekend", "is_holiday", "is_activity_day"]:
        preds[col] = pd.to_numeric(preds[col], errors="coerce").fillna(0).astype(int)

    by_store, by_product, by_category = build_entity_error_tables(preds)
    by_calendar = build_calendar_error_table(preds)
    by_volume = build_volume_error_table(preds)

    save_csv(by_store, TABLE_OUTPUTS["store"])
    save_csv(by_product, TABLE_OUTPUTS["product"])
    save_csv(by_category, TABLE_OUTPUTS["category"])
    save_csv(by_calendar, TABLE_OUTPUTS["calendar"])
    save_csv(by_volume, TABLE_OUTPUTS["volume"])

    plot_error_tables(by_store, by_product, by_category, by_calendar, preds)

    report = build_report(
        preds,
        source,
        validation_mode,
        by_store,
        by_product,
        by_category,
        by_calendar,
        by_volume,
    )
    report_path = OUTPUTS / "q4_error_diagnosis_report.md"
    backup_if_exists(report_path)
    report_path.write_text(report, encoding="utf-8")

    overall = metrics_for_frame(aggregate_daily(preds, ["store_id", "product_id"]))
    worst_store = first_nonempty(by_store, "WAPE_pct", False)
    worst_category = first_nonempty(by_category, "WAPE_pct", False)
    high_low = by_volume.set_index("sales_volume_group")
    high_low_text = ""
    if "高销量商品" in high_low.index and "低销量商品" in high_low.index:
        high_low_text = (
            f"高销量组 WAPE={high_low.loc['高销量商品', 'WAPE_pct']:.2f}%、"
            f"低销量组 WAPE={high_low.loc['低销量商品', 'WAPE_pct']:.2f}%"
        )

    row = (
        "| 2026-05-03 | 问题四误差诊断模块 | "
        f"{source.relative_to(ROOT).as_posix()}；模型={TARGET_MODEL_LABEL} | "
        "Ridge 综合预测误差诊断 | "
        "门店、商品、类别、工作日/周末/节假日/活动日、高低销量分组 | "
        f"整体 WAPE={overall['WAPE_pct']:.2f}%；"
        f"{high_low_text} | "
        f"已生成 q4_error_by_store/product/category/calendar_type 表、误差图和 q4_error_diagnosis_report.md；"
        f"最高门店 WAPE={worst_store['WAPE_pct']:.2f}%（{worst_store['store_name']}），"
        f"最高类别 WAPE={worst_category['WAPE_pct']:.2f}%（{worst_category['category']}） | "
        "节假日/活动日若验证样本不足，不能写成可靠场景结论；低销量商品百分比误差容易放大 | "
        "将诊断结论写入问题四结果分析与模型评价部分 |"
    )
    prepend_result_log(row)

    summary = {
        "source": source.relative_to(ROOT).as_posix(),
        "model": TARGET_MODEL,
        "model_label": TARGET_MODEL_LABEL,
        "validation_mode": validation_mode,
        "date_min": preds["date"].min().date().isoformat(),
        "date_max": preds["date"].max().date().isoformat(),
        "overall": overall,
        "outputs": {
            "tables": {key: value.relative_to(ROOT).as_posix() for key, value in TABLE_OUTPUTS.items()},
            "figures": {key: value.relative_to(ROOT).as_posix() for key, value in FIGURE_OUTPUTS.items()},
            "report": report_path.relative_to(ROOT).as_posix(),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
