from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_store_product_panel
from src.models import exp_smoothing_prediction, optimize_exp_smoothing_alpha
from src.stage5_q4_low_volume_strategy import classification_at_window

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"

TARGET = "positive_sales"
FUTURE_START = pd.Timestamp("2022-04-01")


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


def build_q1_future_predictions(panel: pd.DataFrame, future_template: pd.DataFrame) -> pd.DataFrame:
    combo_cols = ["store_id", "store_name", "product_id", "product_name", "category"]
    combos = (
        panel[combo_cols]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )
    history = {}
    alphas = {}
    for row in combos.itertuples():
        key = (int(row.store_id), int(row.product_id))
        values = (
            panel[
                (panel["store_id"] == row.store_id)
                & (panel["product_id"] == row.product_id)
            ]
            .sort_values("date")[TARGET]
            .astype(float)
            .tolist()
        )
        history[key] = values
        alphas[key] = optimize_exp_smoothing_alpha(pd.Series(values, dtype=float))

    ext_cols = [
        "date",
        "weather",
        "is_holiday",
        "is_weekend",
        "is_activity_day",
        "external_scenario_note",
    ]
    future_ext = future_template[ext_cols].drop_duplicates("date").sort_values("date")
    rows = []
    for ext in future_ext.itertuples():
        for combo in combos.itertuples():
            key = (int(combo.store_id), int(combo.product_id))
            pred = max(0.0, float(exp_smoothing_prediction(history[key], alphas[key])))
            rows.append(
                {
                    "date": pd.Timestamp(ext.date),
                    "store_id": int(combo.store_id),
                    "store_name": combo.store_name,
                    "product_id": int(combo.product_id),
                    "product_name": combo.product_name,
                    "category": combo.category,
                    "weather": ext.weather,
                    "is_holiday": int(ext.is_holiday),
                    "is_weekend": int(ext.is_weekend),
                    "is_activity_day": int(ext.is_activity_day),
                    "pred_q1_exp_smoothing": pred,
                    "external_scenario_note": ext.external_scenario_note,
                }
            )
            history[key].append(pred)
    return pd.DataFrame(rows)


def integerize_forecast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["predicted_sales"] = np.rint(np.maximum(0.0, out["predicted_sales"].astype(float))).astype(int)
    return out


def aggregate_integer_forecast(integer_forecast: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every displayed forecast total from the integer finest-grain table."""
    df = integer_forecast.copy()
    df["predicted_sales"] = df["predicted_sales"].astype(int)
    group_specs = {
        "store_total": ["store_id", "store_name"],
        "store_date_total": ["date", "store_id", "store_name"],
        "category_total": ["category"],
        "category_date_total": ["date", "category"],
        "product_total": ["product_id", "product_name", "category"],
        "store_product_total": [
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "category",
        ],
    }
    outputs: dict[str, pd.DataFrame] = {}
    for name, group_cols in group_specs.items():
        total = (
            df.groupby(group_cols, as_index=False, dropna=False)["predicted_sales"]
            .sum()
            .rename(columns={"predicted_sales": "predicted_7day_sales"})
        )
        if name.endswith("_date_total"):
            total = total.rename(columns={"predicted_7day_sales": "predicted_daily_sales"})
            sort_cols = group_cols
        elif name == "store_product_total":
            sort_cols = ["store_id", "product_id"]
        else:
            sort_cols = ["predicted_7day_sales"]
        ascending = [False] if sort_cols == ["predicted_7day_sales"] else True
        outputs[name] = total.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return outputs


def write_report(
    future: pd.DataFrame,
    store_total: pd.DataFrame,
    category_total: pd.DataFrame,
    integer_total: int,
    cls: pd.DataFrame,
) -> None:
    strategy_metrics = pd.read_csv(TABLES / "q4_low_volume_strategy_metrics.csv")
    sp = strategy_metrics[strategy_metrics["level"] == "store_product"].set_index("model")
    low_count = int(cls["is_low_volume"].sum())
    total_count = int(len(cls))
    report = f"""# 问题四低销量混合策略最终预测说明

执行日期：{date.today().isoformat()}

## 1. 生成目的

原最终预测文件 `outputs/final_7day_forecast.csv` 保留为完整综合 Ridge 条件预测。本文件新增一个不覆盖原结果的候选最终预测：`outputs/final_7day_forecast_hybrid_low_volume.csv`。该策略来自严格 7 日递推低销量鲁棒性检验：低销量门店--商品序列使用问题一简单指数平滑，非低销量序列使用完整综合 Ridge。

## 2. 验证依据

在 2022-03-01 至 2022-03-28 的 4 个完整严格 7 日递推窗口上，低销量指数平滑--常规 Ridge 混合策略门店--商品 WAPE 为 {sp.loc['hybrid_low_q1_regular_ridge','WAPE_pct']:.2f}%，问题一简单指数平滑为 {sp.loc['q1_store_product_exp_smoothing','WAPE_pct']:.2f}%，完整综合 Ridge 为 {sp.loc['ridge_full_external','WAPE_pct']:.2f}%。

## 3. 未来预测规则

以 2022-04-01 作为预测窗口起点，只使用 2022-03-31 及以前历史销量判定低销量序列。本次共有 {low_count}/{total_count} 条门店--商品组合被判为低销量。低销量组合使用简单指数平滑递推预测；其余组合沿用完整综合 Ridge 预测。未来天气、温度、风力和活动日仍沿用历史同期情景，因此预测仍是条件预测。

## 4. 预测汇总

未来 7 天混合策略连续预测总销量为 {future['predicted_sales'].sum():.3f}；按门店--商品--日期最细粒度非负约束并四舍五入后，整数预测总销量为 {integer_total}。所有门店、商品、类别和逐日汇总均由该整数明细表求和得到。

门店汇总：

{df_to_md(store_total, 20)}

类别汇总：

{df_to_md(category_total, 20)}

## 5. 论文表述边界

该混合策略可以作为最终预测候选，因为它在同一严格递推验证口径下略优于原强基准和完整 Ridge；但它是基于低销量诊断的规则化模型选择，不应被写成复杂机器学习模型带来显著突破。正式论文可表述为“在综合 Ridge 的基础上，对低销量稀疏序列采用简单指数平滑兜底，形成稳健混合预测策略”。
"""
    (OUTPUTS / "q4_hybrid_final_forecast_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for path in [TABLES, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)

    ridge_path = OUTPUTS / "final_7day_forecast.csv"
    if not ridge_path.exists():
        raise FileNotFoundError("缺少 outputs/final_7day_forecast.csv。")
    ridge = pd.read_csv(ridge_path)
    ridge["date"] = pd.to_datetime(ridge["date"])
    ridge = ridge.rename(columns={"predicted_sales": "pred_ridge_full"})

    raw = pd.read_csv(ROOT / "data" / "processed" / "modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    panel["date"] = pd.to_datetime(panel["date"])

    q1 = build_q1_future_predictions(panel, ridge)
    cls = classification_at_window(panel, FUTURE_START)
    save_csv(cls, "q4_hybrid_future_low_volume_classification.csv")

    merged = ridge.merge(
        q1[
            [
                "date",
                "store_id",
                "product_id",
                "pred_q1_exp_smoothing",
            ]
        ],
        on=["date", "store_id", "product_id"],
        how="left",
    ).merge(
        cls[["store_id", "product_id", "demand_class", "is_low_volume"]],
        on=["store_id", "product_id"],
        how="left",
    )
    merged["is_low_volume"] = merged["is_low_volume"].fillna(0).astype(int)
    merged["predicted_sales"] = np.where(
        merged["is_low_volume"].eq(1),
        merged["pred_q1_exp_smoothing"],
        merged["pred_ridge_full"],
    )
    merged["predicted_sales"] = np.where(
        np.abs(merged["predicted_sales"].astype(float)) < 1e-6,
        0.0,
        merged["predicted_sales"].astype(float),
    )
    merged["model"] = "hybrid_low_q1_regular_ridge"
    merged["model_label"] = "低销量指数平滑-常规Ridge混合"
    merged["external_scenario_note"] = (
        merged["external_scenario_note"]
        + "；低销量组合使用问题一指数平滑，非低销量组合使用综合Ridge"
    )
    out_cols = [
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
        "demand_class",
        "is_low_volume",
        "external_scenario_note",
    ]
    final = merged[out_cols].sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
    final["date"] = final["date"].dt.date.astype(str)
    save_csv(final, "final_7day_forecast_hybrid_low_volume.csv")
    integer_final = integerize_forecast(final)
    save_csv(integer_final, "final_7day_forecast_hybrid_low_volume_integer.csv")

    summaries = aggregate_integer_forecast(integer_final)
    save_csv(summaries["store_total"], "q4_hybrid_forecast_7day_total_by_store.csv")
    save_csv(summaries["store_date_total"], "q4_hybrid_forecast_daily_by_store.csv")
    save_csv(summaries["category_total"], "q4_hybrid_forecast_7day_total_by_category.csv")
    save_csv(summaries["category_date_total"], "q4_hybrid_forecast_daily_by_category.csv")
    save_csv(summaries["product_total"], "q4_hybrid_forecast_7day_total_by_product.csv")
    save_csv(summaries["store_product_total"], "q4_hybrid_forecast_7day_total_by_store_product.csv")

    integer_total = int(integer_final["predicted_sales"].sum())
    write_report(
        final,
        summaries["store_total"],
        summaries["category_total"],
        integer_total,
        cls,
    )
    summary = {
        "model": "hybrid_low_q1_regular_ridge",
        "future_rows": int(len(final)),
        "low_volume_series": int(cls["is_low_volume"].sum()),
        "total_series": int(len(cls)),
        "future_total_sales": float(final["predicted_sales"].sum()),
        "future_integer_total_sales": integer_total,
        "original_ridge_total_sales": float(ridge["pred_ridge_full"].sum()),
        "integer_aggregation_rule": "only round nonnegative store-product-date cells; all upper totals are sums of that integer matrix",
    }
    (OUTPUTS / "q4_hybrid_final_forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    row = (
        f"| {date.today().isoformat()} | 问题四低销量混合策略最终预测 | "
        "processed: modeling_base_table.csv；输入原 final_7day_forecast.csv；未覆盖原预测表 | "
        "低销量指数平滑 + 常规序列综合 Ridge 混合 | "
        "2022-04-01 前历史销量判定低销量；低销量组合用指数平滑，非低销量组合沿用综合 Ridge；外部变量仍为历史同期情景 | "
        f"未来7天混合预测总量={summary['future_total_sales']:.3f}；整数化总量={summary['future_integer_total_sales']}；"
        f"低销量组合={summary['low_volume_series']}/{summary['total_series']} | "
        "新增不覆盖原结果的候选最终预测表；严格递推验证依据来自 q4_low_volume_strategy_metrics.csv | "
        "混合策略为规则化稳健处理，不应写成复杂模型显著突破；未来外部变量仍为条件情景 | "
        "论文问题四和模型评价中说明最终预测表取舍 |"
    )
    prepend_result_log(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
