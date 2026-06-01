from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calculate_metrics
from src.features import add_leakage_safe_sales_features, build_store_product_panel
from src.models import build_ridge_forecaster, clipped_predict

TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"

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
DETERMINISTIC_NUMERIC = [
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "is_activity_day",
]
WEATHER_NUMERIC = ["max_temperature", "min_temperature", "wind_power"]
BASE_CATEGORICAL = ["store_id_str", "product_id_str", "category"]
WEATHER_CATEGORICAL = ["weather"]


@dataclass(frozen=True)
class SensitivitySpec:
    variant: str
    model_label: str
    weather_setting: str
    numeric_features: list[str]
    categorical_features: list[str]
    use_weather_scenario: bool


SPECS = [
    SensitivitySpec(
        variant="full_external_actual_weather",
        model_label="完整外部变量Ridge",
        weather_setting="验证期使用附件真实天气、温度、风力",
        numeric_features=HISTORY_NUMERIC + DETERMINISTIC_NUMERIC + WEATHER_NUMERIC,
        categorical_features=BASE_CATEGORICAL + WEATHER_CATEGORICAL,
        use_weather_scenario=False,
    ),
    SensitivitySpec(
        variant="deterministic_without_weather",
        model_label="去除天气变量Ridge",
        weather_setting="去除天气、最高温、最低温、风力，保留节假日、活动日、星期等确定性变量",
        numeric_features=HISTORY_NUMERIC + DETERMINISTIC_NUMERIC,
        categorical_features=BASE_CATEGORICAL,
        use_weather_scenario=False,
    ),
    SensitivitySpec(
        variant="historical_weather_scenario",
        model_label="历史同期天气情景Ridge",
        weather_setting="训练仍使用历史真实天气，验证期天气、温度、风力替换为窗口开始日前历史同月日情景",
        numeric_features=HISTORY_NUMERIC + DETERMINISTIC_NUMERIC + WEATHER_NUMERIC,
        categorical_features=BASE_CATEGORICAL + WEATHER_CATEGORICAL,
        use_weather_scenario=True,
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


def prepare_model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_leakage_safe_sales_features(panel, TARGET)
    out["store_id_str"] = out["store_id"].astype(str)
    out["product_id_str"] = out["product_id"].astype(str)
    out["category"] = out["category"].astype(str)
    out["weather"] = out["weather"].fillna("未知").astype(str)
    numeric_cols = HISTORY_NUMERIC + DETERMINISTIC_NUMERIC + WEATHER_NUMERIC
    for col in numeric_cols:
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
        has_all_sales = all(date in data_dates for date in dates)
        has_all_external = all(date in external_dates for date in dates)
        if has_all_sales and has_all_external:
            windows.append((start, end))
        else:
            excluded.append(
                {
                    "window_start": start.date().isoformat(),
                    "window_end": end.date().isoformat(),
                    "reason": "不足完整 7 日销售或外部变量覆盖，未纳入敏感性检验",
                }
            )
    return windows, excluded


def history_feature_row(values: list[float]) -> dict[str, float]:
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


def historical_weather_scenario(
    feature_df: pd.DataFrame,
    target_date: pd.Timestamp,
    window_start: pd.Timestamp,
) -> dict:
    history = feature_df[
        (feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)
    ].copy()
    history["month_day"] = history["date"].dt.strftime("%m-%d")
    same_day = history[history["month_day"] == target_date.strftime("%m-%d")]
    if same_day.empty:
        same_day = history[history["date"].dt.month == target_date.month]
    if same_day.empty:
        same_day = history
    weather_mode = same_day["weather"].dropna().mode()
    return {
        "weather": str(weather_mode.iloc[0]) if not weather_mode.empty else "未知",
        "max_temperature": float(same_day["max_temperature"].mean()),
        "min_temperature": float(same_day["min_temperature"].mean()),
        "wind_power": float(same_day["wind_power"].mean()),
        "scenario_source_days": int(same_day["date"].nunique()),
    }


def external_lookup_from_feature(feature_df: pd.DataFrame) -> dict:
    cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "weekday",
        "month",
        "is_weekend",
        "is_holiday",
        "is_activity_day",
        "has_external_data",
    ]
    return feature_df[cols].drop_duplicates("date").set_index("date").to_dict("index")


def build_batch(
    combos: pd.DataFrame,
    history: dict[tuple[int, int], list[float]],
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


def validate_spec(
    panel: pd.DataFrame,
    feature_df: pd.DataFrame,
    combos: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    spec: SensitivitySpec,
) -> pd.DataFrame:
    ext_lookup = external_lookup_from_feature(feature_df)
    actual_lookup = panel.set_index(["date", "store_id", "product_id"])[TARGET].astype(float).to_dict()
    rows = []
    for window_start, window_end in windows:
        train = feature_df[
            (feature_df["date"] < window_start) & (feature_df["has_external_data"] == 1)
        ].copy()
        model = build_ridge_forecaster(
            spec.numeric_features,
            spec.categorical_features,
            alpha=10.0,
        )
        features = spec.numeric_features + spec.categorical_features
        model.fit(train[features], train[TARGET])

        history = initial_history(panel, combos, window_start)
        for target_date in pd.date_range(window_start, window_end, freq="D"):
            ext = dict(ext_lookup[pd.Timestamp(target_date)])
            scenario_source_days = np.nan
            if spec.use_weather_scenario:
                scenario = historical_weather_scenario(feature_df, target_date, window_start)
                ext.update(
                    {
                        "weather": scenario["weather"],
                        "max_temperature": scenario["max_temperature"],
                        "min_temperature": scenario["min_temperature"],
                        "wind_power": scenario["wind_power"],
                    }
                )
                scenario_source_days = scenario["scenario_source_days"]
            batch = build_batch(combos, history, target_date, ext)
            for col in spec.numeric_features:
                batch[col] = pd.to_numeric(batch[col], errors="coerce").fillna(0.0)
            batch["prediction"] = clipped_predict(model, batch[features])
            for pred_row in batch.itertuples():
                key = (pd.Timestamp(pred_row.date), int(pred_row.store_id), int(pred_row.product_id))
                actual = float(actual_lookup.get(key, 0.0))
                rows.append(
                    {
                        "validation_mode": "strict_recursive_7day",
                        "window_start": window_start,
                        "window_end": window_end,
                        "date": pd.Timestamp(pred_row.date),
                        "horizon": int((pd.Timestamp(pred_row.date) - window_start).days + 1),
                        "store_id": int(pred_row.store_id),
                        "store_name": pred_row.store_name,
                        "product_id": int(pred_row.product_id),
                        "product_name": pred_row.product_name,
                        "category": pred_row.category,
                        "actual": actual,
                        "prediction": float(pred_row.prediction),
                        "variant": spec.variant,
                        "model_label": spec.model_label,
                        "weather_setting": spec.weather_setting,
                        "validation_weather": pred_row.weather,
                        "validation_max_temperature": float(pred_row.max_temperature),
                        "validation_min_temperature": float(pred_row.min_temperature),
                        "validation_wind_power": float(pred_row.wind_power),
                        "scenario_source_days": scenario_source_days,
                    }
                )
            for pred_row in batch.itertuples():
                history[(int(pred_row.store_id), int(pred_row.product_id))].append(
                    float(pred_row.prediction)
                )
    return pd.DataFrame(rows)


def aggregate_for_level(preds: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "store_product":
        return preds.copy()
    if level == "store":
        group_cols = ["window_start", "date", "variant", "model_label", "store_id", "store_name"]
    elif level == "product":
        group_cols = [
            "window_start",
            "date",
            "variant",
            "model_label",
            "product_id",
            "product_name",
            "category",
        ]
    elif level == "category":
        group_cols = ["window_start", "date", "variant", "model_label", "category"]
    else:
        raise ValueError(level)
    return (
        preds.groupby(group_cols, as_index=False, dropna=False)
        .agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        .sort_values(group_cols)
    )


def metric_table(preds: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    setting_lookup = preds.groupby("variant")["weather_setting"].first().to_dict()
    for level in ["store_product", "store", "product", "category"]:
        agg = aggregate_for_level(preds, level)
        for (variant, label), group in agg.groupby(["variant", "model_label"], dropna=False):
            metrics = calculate_metrics(group["actual"], group["prediction"])
            rows.append(
                {
                    "validation_mode": "strict_recursive_7day",
                    "validation_start": min(start for start, _ in windows).date().isoformat(),
                    "validation_end": max(end for _, end in windows).date().isoformat(),
                    "level": level,
                    "variant": variant,
                    "model_label": label,
                    "weather_setting": setting_lookup[variant],
                    **metrics,
                    "WAPE_pct": metrics["WAPE"] * 100,
                    "actual_sum": float(group["actual"].sum()),
                    "prediction_sum": float(group["prediction"].sum()),
                    "n": int(len(group)),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["level", "WAPE", "MAE", "RMSE"]).reset_index(drop=True)
    full = metrics[metrics["variant"] == "full_external_actual_weather"][
        ["level", "MAE", "RMSE", "WAPE", "WAPE_pct"]
    ].rename(
        columns={
            "MAE": "full_MAE",
            "RMSE": "full_RMSE",
            "WAPE": "full_WAPE",
            "WAPE_pct": "full_WAPE_pct",
        }
    )
    metrics = metrics.merge(full, on="level", how="left")
    metrics["MAE_change_vs_full"] = metrics["MAE"] - metrics["full_MAE"]
    metrics["RMSE_change_vs_full"] = metrics["RMSE"] - metrics["full_RMSE"]
    metrics["WAPE_pct_point_change_vs_full"] = metrics["WAPE_pct"] - metrics["full_WAPE_pct"]
    metrics["relative_WAPE_change_vs_full_pct"] = (
        (metrics["WAPE"] - metrics["full_WAPE"]) / metrics["full_WAPE"] * 100
    )
    return metrics.drop(columns=["full_MAE", "full_RMSE", "full_WAPE", "full_WAPE_pct"])


def make_judgement(store_product_metrics: pd.DataFrame) -> tuple[str, str]:
    rows = store_product_metrics.set_index("variant")
    full = rows.loc["full_external_actual_weather"]
    no_weather = rows.loc["deterministic_without_weather"]
    scenario = rows.loc["historical_weather_scenario"]
    full_gain_vs_no_weather = no_weather["WAPE_pct"] - full["WAPE_pct"]
    scenario_gap_vs_full = scenario["WAPE_pct"] - full["WAPE_pct"]
    scenario_gain_vs_no_weather = no_weather["WAPE_pct"] - scenario["WAPE_pct"]

    if scenario_gap_vs_full >= 0:
        scenario_gap_text = f"历史同期天气情景比真实天气验证高 {scenario_gap_vs_full:.2f} 个百分点"
    else:
        scenario_gap_text = f"历史同期天气情景比真实天气验证低 {abs(scenario_gap_vs_full):.2f} 个百分点"

    if full_gain_vs_no_weather >= 1.0 and scenario_gap_vs_full <= 0.5:
        label = "天气变量提升明显且情景替换后仍较稳健"
        explanation = (
            f"完整模型相对去天气模型 WAPE 下降 {full_gain_vs_no_weather:.2f} 个百分点，"
            f"{scenario_gap_text}。"
        )
    elif abs(full_gain_vs_no_weather) < 1.0:
        label = "天气变量提升有限"
        explanation = (
            f"完整模型相对去天气模型 WAPE 仅变化 {full_gain_vs_no_weather:.2f} 个百分点，"
            "在门店-商品主粒度上不足以支持“天气显著提升预测精度”的表述。"
        )
    elif scenario_gap_vs_full >= 1.0 and scenario_gain_vs_no_weather <= 0:
        label = "天气变量可能引入验证乐观偏差"
        explanation = (
            f"真实天气验证下完整模型比去天气模型好 {full_gain_vs_no_weather:.2f} 个百分点，"
            f"但换成历史同期情景后反而比完整模型高 {scenario_gap_vs_full:.2f} 个百分点，"
            "说明验证期真实天气带来的收益不一定能在未来天气未知时复现。"
        )
    else:
        label = "天气变量有一定贡献，但稳健性证据不足"
        explanation = (
            f"完整模型相对去天气模型 WAPE 下降 {full_gain_vs_no_weather:.2f} 个百分点；"
            f"历史同期情景相对去天气模型下降 {scenario_gain_vs_no_weather:.2f} 个百分点，"
            f"相对真实天气验证高 {scenario_gap_vs_full:.2f} 个百分点。"
        )
    return label, explanation


def build_report(
    metrics: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    excluded: list[dict],
    judgement_label: str,
    judgement_explanation: str,
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
    store_product_metrics = metrics[metrics["level"] == "store_product"][
        [
            "model_label",
            "weather_setting",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "WAPE_pct_point_change_vs_full",
            "relative_WAPE_change_vs_full_pct",
            "actual_sum",
            "prediction_sum",
            "n",
        ]
    ].copy()
    level_metrics = metrics[
        [
            "level",
            "model_label",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "WAPE_pct_point_change_vs_full",
            "n",
        ]
    ].copy()
    excluded_df = pd.DataFrame(excluded)
    return f"""# 问题四天气变量敏感性检验报告

执行日期：2026-05-03

## 1. 检验目的

本检验回答两个问题：第一，天气类型、最高温、最低温、风力是否真正降低问题四综合预测误差；第二，当未来真实天气不可知、只能使用历史同期天气情景时，含天气模型是否仍然稳健。

为隔离天气变量的作用，三组模型均使用 Ridge 回归和同一严格 7 日递推验证口径，区别只在天气相关变量的处理方式。

## 2. 验证窗口

采用完整 7 日递推窗口如下：

{df_to_md(window_df)}

未纳入窗口如下：

{df_to_md(excluded_df)}

说明：每个窗口只用窗口开始日前的真实销量初始化滞后和滚动特征；窗口内后续日期使用前面日期的预测值递推。这比日滚动一步预测更接近“未来 7 天一次性预测”的实际使用场景。

## 3. 三组模型设置

| 模型 | 天气变量处理 | 目的 |
|---|---|---|
| 完整外部变量Ridge | 使用验证期附件真实天气、最高温、最低温、风力 | 衡量原含天气方案在验证集上的表现 |
| 去除天气变量Ridge | 去除天气、最高温、最低温、风力，保留节假日、活动日、星期、月份、周末等变量 | 判断天气变量是否带来额外预测收益 |
| 历史同期天气情景Ridge | 训练仍使用历史真实天气，验证期天气、温度、风力替换为窗口开始日前历史同月日情景 | 模拟未来真实天气不可知时的预测条件 |

## 4. 门店-商品主粒度指标

{df_to_md(store_product_metrics, 10)}

## 5. 分层指标

{df_to_md(level_metrics, 20)}

完整指标表已保存至 `tables/q4_weather_sensitivity_metrics.csv`。

## 6. 结论

判断：**{judgement_label}**。

{judgement_explanation}

因此，论文中应把天气、温度、风力写成“可作为外部情景变量的补充特征”，而不是写成稳定、必然的精度提升来源。若正式预测时无法获得未来天气预报或不想引入天气情景假设，应优先报告去天气模型作为稳健对照；若使用含天气模型，则必须说明预测结果是在给定天气情景下得到的条件预测。
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
        lines.insert(insert_idx, row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [TABLES, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "processed" / "modeling_base_table.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    panel = build_store_product_panel(raw, TARGET)
    panel["date"] = pd.to_datetime(panel["date"])
    feature_df = prepare_model_frame(panel)

    windows, excluded = validation_windows(feature_df)
    if not windows:
        raise RuntimeError("没有可用于天气敏感性检验的完整 7 日验证窗口。")

    combos = (
        panel[["store_id", "store_name", "product_id", "product_name", "category"]]
        .drop_duplicates()
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )
    preds = pd.concat(
        [validate_spec(panel, feature_df, combos, windows, spec) for spec in SPECS],
        ignore_index=True,
    )
    preds["abs_error"] = (preds["actual"] - preds["prediction"]).abs()

    metrics = metric_table(preds, windows)
    metrics.to_csv(TABLES / "q4_weather_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")

    store_product_metrics = metrics[metrics["level"] == "store_product"].copy()
    judgement_label, judgement_explanation = make_judgement(store_product_metrics)
    report = build_report(metrics, windows, excluded, judgement_label, judgement_explanation)
    (OUTPUTS / "q4_weather_sensitivity_report.md").write_text(report, encoding="utf-8")

    sp = store_product_metrics.set_index("variant")
    row = (
        "| 2026-05-03 | 问题四天气变量敏感性检验 | processed: modeling_base_table.csv | "
        "Ridge 三组对照：完整天气、去天气、历史同期天气情景 | "
        "lag/rolling、门店、商品、类别、星期、节假日、活动日；天气/温度/风力按三种情景处理 | "
        f"严格7日递推门店-商品 WAPE：完整={sp.loc['full_external_actual_weather','WAPE_pct']:.2f}%，"
        f"去天气={sp.loc['deterministic_without_weather','WAPE_pct']:.2f}%，"
        f"历史同期天气情景={sp.loc['historical_weather_scenario','WAPE_pct']:.2f}% | "
        f"{judgement_label}；已输出 tables/q4_weather_sensitivity_metrics.csv 和 outputs/q4_weather_sensitivity_report.md | "
        "验证期真实天气可能高估未来可用性；历史同期情景只是一种可复现假设，不等于真实天气预报 | "
        "同步修订问题三、问题四报告中天气变量表述 |"
    )
    prepend_result_log(row)

    summary = {
        "validation_mode": "strict_recursive_7day",
        "windows": [
            {"window_start": str(start.date()), "window_end": str(end.date())}
            for start, end in windows
        ],
        "store_product_metrics": store_product_metrics[
            ["variant", "MAE", "RMSE", "WAPE", "WAPE_pct"]
        ].to_dict("records"),
        "judgement": judgement_label,
    }
    (OUTPUTS / "q4_weather_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
