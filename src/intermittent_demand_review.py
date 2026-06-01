from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_base_table.csv"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
TARGET = "positive_sales"


def combine_unique_names(values: pd.Series) -> str:
    names = [str(x) for x in values.dropna().unique()]
    return " / ".join(sorted(names))


def to_markdown_table(
    df: pd.DataFrame,
    max_rows: int | None = None,
    float_digits: int = 3,
) -> str:
    if df.empty:
        return "无。"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}"
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


def build_store_product_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to date-store-product_id grain and keep zero-demand days."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    product_names = data.groupby("product_id")["product_name"].apply(combine_unique_names)
    category_mode = (
        data.groupby("product_id")["category"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "待确认")
        .rename("product_category")
    )

    panel = (
        data.groupby(["date", "store_id", "product_id"], as_index=False)
        .agg(
            positive_sales=(TARGET, "sum"),
            daily_sales=("daily_sales", "sum"),
            store_name=("store_name", "first"),
            has_sales_record=("has_sales_record", "max"),
        )
        .sort_values(["store_id", "product_id", "date"])
    )
    panel["product_name"] = panel["product_id"].map(product_names)
    panel = panel.merge(category_mode, on="product_id", how="left")
    panel = panel.rename(columns={"product_category": "category"})
    return panel


def formal_demand_class(row: pd.Series) -> str:
    """Syntetos-Boylan style diagnostic using ADI and CV squared."""
    if row["nonzero_sales_days"] == 0:
        return "无历史正向需求"
    if row["average_demand_interval"] < 1.32 and row["cv_squared"] < 0.49:
        return "平稳需求"
    if row["average_demand_interval"] >= 1.32 and row["cv_squared"] < 0.49:
        return "间歇需求"
    if row["average_demand_interval"] < 1.32 and row["cv_squared"] >= 0.49:
        return "波动需求"
    return "块状/高度波动间歇需求"


def practical_demand_class(row: pd.Series) -> str:
    if row["nonzero_sales_days"] == 0:
        return "无历史正向需求"
    if row["nonzero_sales_day_ratio"] <= 0.15 or row["average_sales"] < 0.20:
        return "极低销量稀疏序列"
    if row["zero_sales_ratio"] >= 0.75 or row["average_sales"] < 0.50:
        return "低销量间歇序列"
    if row["zero_sales_ratio"] >= 0.50:
        return "间歇性需求序列"
    if row["coefficient_of_variation"] >= 1.50:
        return "有销量但波动较大序列"
    return "相对稳定序列"


def recommend_strategy(row: pd.Series) -> str:
    label = row["demand_class"]
    if label in {"无历史正向需求", "极低销量稀疏序列"}:
        return (
            "收缩预测为主：门店-商品均值向商品/类别均值收缩，7日总量可低于1；"
            "不单独训练复杂模型"
        )
    if label == "低销量间歇序列":
        return (
            "保留简单均值和同星期均值，并加入收缩均值兜底；"
            "复杂模型只作聚合层补充"
        )
    if label == "间歇性需求序列":
        return (
            "使用同星期均值、近28日均值、简单指数平滑比较；"
            "预测后做非负裁剪和低销量收缩"
        )
    if label == "有销量但波动较大序列":
        return (
            "可保留综合模型，但必须与baseline比较；"
            "异常大单不应被复杂模型过度拟合"
        )
    return "可使用现有baseline或综合模型，但仍需时间切分验证"


def make_summary(panel: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["store_id", "store_name", "product_id", "product_name", "category"]
    summary = (
        panel.groupby(group_cols, dropna=False)["positive_sales"]
        .agg(
            total_days="count",
            total_sales="sum",
            average_sales="mean",
            max_sales="max",
            std_sales="std",
            nonzero_sales_days=lambda s: int((s > 0).sum()),
            zero_sales_days=lambda s: int((s <= 0).sum()),
            mean_positive_sales_when_nonzero=lambda s: (
                float(s[s > 0].mean()) if (s > 0).any() else np.nan
            ),
        )
        .reset_index()
    )
    summary["nonzero_sales_day_ratio"] = (
        summary["nonzero_sales_days"] / summary["total_days"]
    )
    summary["zero_sales_ratio"] = summary["zero_sales_days"] / summary["total_days"]
    summary["coefficient_of_variation"] = summary["std_sales"] / summary[
        "average_sales"
    ].replace(0, np.nan)
    summary["cv_squared"] = summary["coefficient_of_variation"] ** 2
    summary["average_demand_interval"] = np.where(
        summary["nonzero_sales_days"] > 0,
        summary["total_days"] / summary["nonzero_sales_days"],
        np.inf,
    )
    summary["is_intermittent_demand"] = (
        (summary["zero_sales_ratio"] >= 0.50)
        | (summary["average_demand_interval"] >= 2.0)
    ).astype(int)
    summary["demand_class"] = summary.apply(practical_demand_class, axis=1)
    summary["formal_adi_cv_class"] = summary.apply(formal_demand_class, axis=1)
    summary["recommended_strategy"] = summary.apply(recommend_strategy, axis=1)

    ordered_cols = [
        "store_id",
        "store_name",
        "product_id",
        "product_name",
        "category",
        "total_days",
        "nonzero_sales_days",
        "nonzero_sales_day_ratio",
        "average_sales",
        "max_sales",
        "zero_sales_ratio",
        "std_sales",
        "coefficient_of_variation",
        "average_demand_interval",
        "cv_squared",
        "is_intermittent_demand",
        "demand_class",
        "formal_adi_cv_class",
        "mean_positive_sales_when_nonzero",
        "total_sales",
        "recommended_strategy",
    ]
    return summary[ordered_cols].sort_values(
        ["zero_sales_ratio", "average_sales", "store_id", "product_id"],
        ascending=[False, True, True, True],
    )


def count_line(count: int, total: int) -> str:
    return f"{count} 条，占 {count / total:.1%}"


def write_report(summary: pd.DataFrame, panel: pd.DataFrame) -> None:
    total = len(summary)
    zero50 = int((summary["zero_sales_ratio"] >= 0.50).sum())
    zero75 = int((summary["zero_sales_ratio"] >= 0.75).sum())
    avg05 = int((summary["average_sales"] < 0.50).sum())
    avg10 = int((summary["average_sales"] < 1.00).sum())
    intermittent = int(summary["is_intermittent_demand"].sum())
    formal_lumpy = int(
        (summary["formal_adi_cv_class"] == "块状/高度波动间歇需求").sum()
    )
    formal_erratic = int((summary["formal_adi_cv_class"] == "波动需求").sum())

    class_counts = (
        summary["demand_class"]
        .value_counts()
        .rename_axis("demand_class")
        .reset_index(name="sequence_count")
    )
    class_counts["sequence_share"] = class_counts["sequence_count"] / total

    formal_counts = (
        summary["formal_adi_cv_class"]
        .value_counts()
        .rename_axis("formal_adi_cv_class")
        .reset_index(name="sequence_count")
    )
    formal_counts["sequence_share"] = formal_counts["sequence_count"] / total

    strategy_counts = (
        summary["recommended_strategy"]
        .value_counts()
        .rename_axis("recommended_strategy")
        .reset_index(name="sequence_count")
    )

    quantiles = pd.DataFrame(
        {
            "metric": [
                "zero_sales_ratio",
                "average_sales",
                "coefficient_of_variation",
                "average_demand_interval",
            ],
            "p25": [
                summary["zero_sales_ratio"].quantile(0.25),
                summary["average_sales"].quantile(0.25),
                summary["coefficient_of_variation"].quantile(0.25),
                summary["average_demand_interval"].quantile(0.25),
            ],
            "median": [
                summary["zero_sales_ratio"].quantile(0.50),
                summary["average_sales"].quantile(0.50),
                summary["coefficient_of_variation"].quantile(0.50),
                summary["average_demand_interval"].quantile(0.50),
            ],
            "p75": [
                summary["zero_sales_ratio"].quantile(0.75),
                summary["average_sales"].quantile(0.75),
                summary["coefficient_of_variation"].quantile(0.75),
                summary["average_demand_interval"].quantile(0.75),
            ],
        }
    )

    sparse_examples = summary[
        [
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "average_sales",
            "max_sales",
            "nonzero_sales_day_ratio",
            "zero_sales_ratio",
            "coefficient_of_variation",
            "average_demand_interval",
            "demand_class",
        ]
    ].head(12)

    date_min = panel["date"].min().strftime("%Y-%m-%d")
    date_max = panel["date"].max().strftime("%Y-%m-%d")
    unique_days = panel["date"].nunique()

    report = f"""# 门店-商品粒度间歇性需求审查

## 1. 数据口径

本次审查读取 `data/processed/modeling_base_table.csv`，按 `store_id + product_id` 汇总为门店-商品编号日序列。商品编号 `11001020` 在原表中有两个商品名称，已按项目数据字典的处理原则合并为同一商品编号，避免把录入名称差异误判为两个序列。

销量口径使用 `positive_sales`，即顾客正向购买销量。负销量在前序阶段已确认为损耗或冲销类调整，不作为正常需求销量参与间歇性需求判断。

样本日期为 {date_min} 至 {date_max}，共 {unique_days} 个日历日；门店-商品编号序列共 {total} 条。

## 2. 指标定义

- 非零销售天数占比：序列中 `positive_sales > 0` 的天数 / 总天数。
- 平均销量：包含零销量日后的日均 `positive_sales`。
- 最大销量：序列日销量最大值，用于识别偶发大单。
- 零销量比例：序列中 `positive_sales <= 0` 的天数 / 总天数。
- 变异系数：日销量标准差 / 日均销量。均值越小、偶发大单越多，变异系数通常越高。
- ADI：平均需求间隔，计算为总天数 / 非零销售天数。ADI 越大，说明越久才出现一次非零需求。

## 3. 总体判断

当前门店-商品粒度存在明显的大量零销量和低销量序列：

- 零销量比例不低于 50% 的序列：{count_line(zero50, total)}。
- 零销量比例不低于 75% 的序列：{count_line(zero75, total)}。
- 平均日销量低于 0.5 的序列：{count_line(avg05, total)}。
- 平均日销量低于 1.0 的序列：{count_line(avg10, total)}。
- 按“零销量比例至少 50% 或 ADI 至少 2”的可解释规则判定为间歇性需求：{count_line(intermittent, total)}。

分位数统计如下：

{to_markdown_table(quantiles)}

按实用建模口径划分的序列类型如下：

{to_markdown_table(class_counts)}

最稀疏的若干序列如下，完整结果见 `tables/intermittent_demand_summary.csv`：

{to_markdown_table(sparse_examples, float_digits=3)}

## 4. 间歇性需求判定

本报告采用两套口径：

第一，本文采用更直观的实用口径：当一个门店-商品序列超过一半日子没有正向销售，即 `zero_sales_ratio >= 0.50`，可判定为间歇性需求。这一规则容易解释，相当于平均至少约隔 2 天才出现一次正向需求。

第二，作为方法诊断，补充 ADI 与 `CV^2` 的经典分类：ADI 以 1.32 为分界，`CV^2` 以 0.49 为分界。按该规则，本数据中 {formal_lumpy} 条属于“块状/高度波动间歇需求”，{formal_erratic} 条属于“波动需求”。结果说明：该粒度不仅有大量零销量，而且非零日销量也很不稳定。

ADI-CV 诊断分类如下：

{to_markdown_table(formal_counts)}

## 5. 稳健预测策略

本文不对所有门店-商品序列统一使用复杂模型。原因是：大量序列的有效非零销售日较少，复杂模型容易把偶发大单、短期活动或噪声学习成稳定规律，从而在未来 7 日预测中高估低销量商品。

分层策略如下：

1. 极低销量稀疏序列：以收缩预测为主。可将门店-商品自身均值向商品均值或类别均值收缩，例如
   `预测值 = w * 门店商品同星期均值 + (1 - w) * 商品或类别同星期均值`，
   其中 `w = 非零销售天数 / (非零销售天数 + 20)`。非零天数越少，越不相信该序列自身的偶然波动。
2. 低销量间歇序列：保留简单均值、近 28 日均值、同星期均值，并增加收缩均值兜底。评价时看 7 日总量，不宜过分关注单日是否正好为零。
3. 中等间歇序列：可以比较同星期均值、简单指数平滑和收缩均值，但必须使用时间切分验证，并与 baseline 比较。
4. 有销量但波动较大序列：综合 Ridge 或树模型可以保留为候选，但应设置非负裁剪，并检查偶发大单对模型的影响。
5. 相对稳定序列：可继续使用现有 baseline 或综合模型，但仍需报告 MAE、RMSE、WAPE，不能只报告复杂模型结果。

逐序列推荐策略计数如下：

{to_markdown_table(strategy_counts)}

## 6. Croston / TSB / 简化间歇需求方法取舍

结论：Croston 或 TSB 不作为当前论文主模型；“简化间歇需求处理”作为低销量序列的稳健策略。

理由如下：

1. Croston 方法主要分开估计“需求间隔”和“非零需求规模”，适合间歇性需求，但它本身不直接利用星期、节假日、活动日等解释变量。当前题目需要解释门店、商品、类别、外部因素，直接把 Croston 作为主模型不如同星期均值和收缩均值容易写清楚。
2. TSB 方法进一步估计“发生需求的概率”和“发生后的需求规模”，理论上更适合存在大量零销量的序列。但它比简单均值多了平滑参数，若没有单独验证其误差下降，放进本科建模论文会增加解释成本。
3. 当前最稳健的做法是：对低销量序列使用“需求发生概率 × 非零销量规模”的简化思想。例如先估计某门店-商品在同星期几出现正向销售的概率，再乘以正向销售日的平均销量，并向商品或类别层级收缩。该方法保留了间歇需求思想，但比 Croston/TSB 更容易解释和复现。

因此，论文表述口径为：门店-商品细粒度存在明显间歇性需求，低销量序列不宜依赖复杂模型单独拟合；本文采用简单均值、同星期均值和层级收缩预测作为低销量序列的稳健处理，并将 Croston/TSB 作为可选扩展而非主方案。
"""

    (OUTPUTS / "intermittent_demand_review.md").write_text(report, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    panel = build_store_product_panel(df)
    summary = make_summary(panel)

    summary.to_csv(
        TABLES / "intermittent_demand_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(summary, panel)

    total = len(summary)
    intermittent = int(summary["is_intermittent_demand"].sum())
    print(
        f"Wrote intermittent demand review for {total} store-product series; "
        f"{intermittent} flagged as intermittent."
    )


if __name__ == "__main__":
    main()
