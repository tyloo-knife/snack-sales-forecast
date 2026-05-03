from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "modeling_base_table.csv"
OUTPUTS = ROOT / "outputs" / "method_search"
TABLES = ROOT / "tables" / "method_search"
FIGURES = ROOT / "figures" / "method_search"
NOTEBOOKS = ROOT / "notebooks" / "method_search"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")


WEATHER_GROUP_DISPLAY = {
    "no_precip": "无降水",
    "light_rain": "小雨/阵雨",
    "moderate_heavy_rain": "中大雨及以上",
    "sleet_rare": "雨夹雪",
}


def ensure_dirs() -> None:
    for path in [OUTPUTS, TABLES, FIGURES, NOTEBOOKS]:
        path.mkdir(parents=True, exist_ok=True)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None, digits: int = 4) -> str:
    if df is None or df.empty:
        return "无。"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}"
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


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else math.nan


def metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, level: str) -> dict[str, float | str]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "analysis_level": level,
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "WAPE": wape(y, pred),
    }


def weather_to_group(weather: str) -> str:
    no_precip = {"晴", "多云", "阴"}
    light = {"小雨", "小到中雨", "阵雨", "雷阵雨"}
    heavy = {"中雨", "中到大雨", "大雨", "大到暴雨", "暴雨"}
    if weather in no_precip:
        return "no_precip"
    if weather in light:
        return "light_rain"
    if weather in heavy:
        return "moderate_heavy_rain"
    if weather == "雨夹雪":
        return "sleet_rare"
    return "other"


def make_temp_bins(series: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    quantiles = series.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy()
    quantiles = np.unique(np.round(quantiles, 6))
    labels = [f"T{i}" for i in range(1, len(quantiles))]
    binned = pd.cut(series, bins=quantiles, labels=labels, include_lowest=True)
    meta_rows = []
    for i, label in enumerate(labels):
        meta_rows.append(
            {
                "temp_bin": label,
                "avg_temperature_min": float(quantiles[i]),
                "avg_temperature_max": float(quantiles[i + 1]),
            }
        )
    return binned.astype(str), pd.DataFrame(meta_rows)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)

    required = [
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_holiday",
        "is_weekend",
        "is_activity_day",
        "weekday",
        "month",
        "store_id",
        "product_id",
        "product_name",
        "category",
        TARGET,
    ]
    analysis = df[df["has_external_data"] == 1].dropna(subset=required).copy()
    analysis["avg_temperature"] = (
        analysis["max_temperature"] + analysis["min_temperature"]
    ) / 2
    analysis["temperature_range"] = (
        analysis["max_temperature"] - analysis["min_temperature"]
    )
    analysis["log_positive_sales"] = np.log1p(analysis[TARGET])
    analysis["weather_group"] = analysis["weather"].astype(str).map(weather_to_group)
    analysis["temp_bin"], temp_bin_meta = make_temp_bins(analysis["avg_temperature"])

    analysis["store_id_str"] = analysis["store_id"].astype(int).astype(str)
    analysis["product_id_str"] = analysis["product_id"].astype(int).astype(str)
    analysis["weekday_str"] = analysis["weekday"].astype(int).astype(str)
    analysis["month_str"] = analysis["month"].astype(int).astype(str)
    analysis["category_str"] = analysis["category"].astype(str)
    analysis["combo_id"] = (
        analysis["store_id_str"]
        + "|"
        + analysis["product_id_str"]
        + "|"
        + analysis["product_name"].astype(str)
    )
    analysis = analysis.sort_values(["combo_id", "date"]).copy()
    analysis["lag_7"] = analysis.groupby("combo_id")[TARGET].shift(7)
    analysis["rolling_28_prev"] = analysis.groupby("combo_id")[TARGET].transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    )

    combo_stats = analysis.groupby("combo_id")[TARGET].agg(["mean", "std"])
    analysis = analysis.join(combo_stats, on="combo_id", rsuffix="_combo")
    analysis["std_combo"] = analysis["std"].replace(0, np.nan)
    analysis["standardized_sales"] = (
        analysis[TARGET] - analysis["mean"]
    ) / analysis["std_combo"]
    analysis["standardized_sales"] = analysis["standardized_sales"].fillna(0.0)
    analysis = analysis.drop(columns=["mean", "std", "std_combo"])

    daily = (
        analysis.groupby(
            [
                "date",
                "weather",
                "weather_group",
                "max_temperature",
                "min_temperature",
                "avg_temperature",
                "temperature_range",
                "temp_bin",
                "wind_power",
                "is_holiday",
                "is_weekend",
                "is_activity_day",
                "weekday",
                "month",
            ],
            as_index=False,
        )[TARGET]
        .sum()
        .rename(columns={TARGET: "daily_total_sales"})
        .sort_values("date")
    )
    q1, q3 = daily["daily_total_sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(0.0, q1 - 1.5 * iqr)
    daily["is_daily_total_outlier"] = (
        (daily["daily_total_sales"] < lower)
        | (daily["daily_total_sales"] > upper)
    ).astype(int)
    outlier_dates = set(daily.loc[daily["is_daily_total_outlier"] == 1, "date"])
    analysis["is_daily_total_outlier"] = analysis["date"].isin(outlier_dates).astype(int)

    # Fix category levels before fitting formulas so validation rows cannot introduce
    # an unseen categorical level relative to the training window.
    for col in [
        "weather",
        "weather_group",
        "temp_bin",
        "store_id_str",
        "product_id_str",
        "weekday_str",
        "month_str",
        "category_str",
    ]:
        levels = sorted(analysis[col].dropna().astype(str).unique().tolist())
        analysis[col] = pd.Categorical(analysis[col].astype(str), categories=levels)
        if col in daily.columns:
            daily[col] = pd.Categorical(daily[col].astype(str), categories=levels)

    return analysis, daily, temp_bin_meta


def fit_cluster_ols(data: pd.DataFrame, formula: str):
    groups = data["date"].astype("category").cat.codes
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_plain_ols(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit()


def evaluate_formula(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    formula: str,
    target_col: str,
    level: str,
    inverse_log: bool = False,
) -> tuple[object, dict[str, float | str]]:
    model = fit_plain_ols(train, formula)
    pred = np.asarray(model.predict(valid), dtype=float)
    if inverse_log:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    return model, metrics(valid[target_col].to_numpy(), pred, level)


def extract_level(term: str) -> str:
    match = re.search(r"\[T\.(.*)\]", term)
    return match.group(1) if match else term


def extract_external_coefficients(
    model,
    model_name: str,
    iqr_map: dict[str, float],
    target_scale: str,
) -> pd.DataFrame:
    rows = []
    for term, coef in model.params.items():
        factor = None
        variable = None
        comparable = abs(float(coef))
        if term.startswith("C(weather_group"):
            factor = "weather_group"
            variable = WEATHER_GROUP_DISPLAY.get(extract_level(term), extract_level(term))
        elif term.startswith("C(weather,"):
            factor = "weather_detail"
            variable = extract_level(term)
        elif term.startswith("C(temp_bin"):
            factor = "temp_bin"
            variable = extract_level(term)
        elif term in ["avg_temperature", "temperature_range", "max_temperature", "min_temperature", "wind_power"]:
            factor = term
            variable = term
            comparable = abs(float(coef) * iqr_map.get(term, 1.0))
        elif term in ["is_holiday", "is_activity_day", "is_weekend"]:
            factor = term
            variable = term
        if factor is None:
            continue
        rows.append(
            {
                "model": model_name,
                "target_scale": target_scale,
                "term": term,
                "factor": factor,
                "variable": variable,
                "coef": float(coef),
                "std_err": float(model.bse.get(term, np.nan)),
                "p_value": float(model.pvalues.get(term, np.nan)),
                "direction": "正关联" if coef > 0 else "负关联" if coef < 0 else "近零",
                "comparable_abs_effect": comparable,
            }
        )
    return pd.DataFrame(rows).sort_values("comparable_abs_effect", ascending=False)


def group_binary_diff(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("weather_group", "天气合并类别"),
        ("is_holiday", "是否节假日"),
        ("is_weekend", "是否周末"),
        ("is_activity_day", "是否活动日"),
        ("temp_bin", "平均温度分箱"),
    ]
    for col, label in specs:
        stats = (
            daily.groupby(col, observed=False)["daily_total_sales"]
            .agg(n_days="count", mean_sales="mean", median_sales="median", std_sales="std")
            .reset_index()
        )
        for _, row in stats.iterrows():
            value = row[col]
            display = WEATHER_GROUP_DISPLAY.get(str(value), str(value))
            rows.append(
                {
                    "factor": label,
                    "level": display,
                    "n_days": int(row["n_days"]),
                    "mean_daily_total_sales": float(row["mean_sales"]),
                    "median_daily_total_sales": float(row["median_sales"]),
                    "std_daily_total_sales": float(row["std_sales"]) if pd.notna(row["std_sales"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def holiday_weekend_crosstab(daily: pd.DataFrame) -> pd.DataFrame:
    tmp = daily.copy()
    tmp["holiday_weekend_group"] = np.select(
        [
            (tmp["is_holiday"] == 0) & (tmp["is_weekend"] == 0),
            (tmp["is_holiday"] == 0) & (tmp["is_weekend"] == 1),
            (tmp["is_holiday"] == 1) & (tmp["is_weekend"] == 0),
            (tmp["is_holiday"] == 1) & (tmp["is_weekend"] == 1),
        ],
        ["非节假日工作日", "非节假日周末", "节假日工作日", "节假日周末"],
        default="其他",
    )
    return (
        tmp.groupby("holiday_weekend_group")["daily_total_sales"]
        .agg(n_days="count", mean_daily_total_sales="mean", median_daily_total_sales="median")
        .reset_index()
        .sort_values("mean_daily_total_sales", ascending=False)
    )


def activity_diagnostics(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekday = (
        daily.groupby(["weekday", "is_activity_day"])["daily_total_sales"]
        .agg(n_days="count", mean_daily_total_sales="mean")
        .reset_index()
        .sort_values(["weekday", "is_activity_day"])
    )
    top_days = daily.sort_values("daily_total_sales", ascending=False).head(20)[
        [
            "date",
            "daily_total_sales",
            "weather",
            "is_holiday",
            "is_weekend",
            "is_activity_day",
            "weekday",
            "is_daily_total_outlier",
        ]
    ]
    return weekday, top_days


def run_random_forest(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = analysis.dropna(subset=["lag_7", "rolling_28_prev"]).copy()
    train = model_df[model_df["date"] < VALIDATION_START].copy()
    valid = model_df[model_df["date"] >= VALIDATION_START].copy()
    numeric = [
        "avg_temperature",
        "temperature_range",
        "wind_power",
        "is_holiday",
        "is_weekend",
        "is_activity_day",
        "lag_7",
        "rolling_28_prev",
    ]
    categorical = [
        "weather_group",
        "store_id_str",
        "product_id_str",
        "weekday_str",
        "month_str",
        "category_str",
        "temp_bin",
    ]
    X_train = train[numeric + categorical]
    y_train = train[TARGET]
    X_valid = valid[numeric + categorical]
    y_valid = valid[TARGET]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", pre), ("model", rf)])
    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_valid), 0, None)
    metric = metrics(y_valid.to_numpy(), pred, "row_level_store_product")
    metric.update(
        {
            "method": "RandomForest grouped weather + history",
            "train_rows": len(train),
            "validation_rows": len(valid),
        }
    )
    result = permutation_importance(
        pipe,
        X_valid,
        y_valid,
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": X_valid.columns,
            "importance_mean_mae_increase": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_mae_increase", ascending=False)
    return pd.DataFrame([metric]), imp


def import_status() -> pd.DataFrame:
    rows = []
    for package in ["lightgbm", "shap"]:
        try:
            __import__(package)
            available = True
            note = "环境可导入，但本问以可解释回归为主；如需使用，应只作补充。"
        except Exception as exc:  # pragma: no cover - diagnostic only
            available = False
            note = f"当前环境不可导入：{type(exc).__name__}"
        rows.append({"package": package, "available": available, "note": note})
    return pd.DataFrame(rows)


def run_regression_suite(
    analysis: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = analysis.dropna(subset=["lag_7", "rolling_28_prev"]).copy()
    train = model_df[model_df["date"] < VALIDATION_START].copy()
    valid = model_df[model_df["date"] >= VALIDATION_START].copy()

    iqr_map = {
        col: float(model_df[col].quantile(0.75) - model_df[col].quantile(0.25))
        for col in [
            "avg_temperature",
            "temperature_range",
            "max_temperature",
            "min_temperature",
            "wind_power",
            "lag_7",
            "rolling_28_prev",
        ]
    }

    formulas = {
        "原方案详细天气FE回归": {
            "formula": "positive_sales ~ C(weather, Treatment(reference='晴')) + max_temperature + min_temperature + wind_power + is_holiday + is_activity_day + C(store_id_str) + C(product_id_str) + C(weekday_str) + C(month_str)",
            "target": TARGET,
            "scale": "原始正向销量",
            "inverse_log": False,
            "data": analysis.copy(),
        },
        "合并天气_对数销量_历史控制FE回归": {
            "formula": "log_positive_sales ~ C(weather_group, Treatment(reference='no_precip')) + avg_temperature + temperature_range + wind_power + is_holiday + is_activity_day + lag_7 + rolling_28_prev + C(store_id_str) + C(product_id_str) + C(weekday_str) + C(month_str)",
            "target": "log_positive_sales",
            "scale": "log1p销量",
            "inverse_log": True,
            "data": model_df,
        },
        "温度分箱_对数销量_历史控制FE回归": {
            "formula": "log_positive_sales ~ C(weather_group, Treatment(reference='no_precip')) + C(temp_bin) + wind_power + is_holiday + is_activity_day + lag_7 + rolling_28_prev + C(store_id_str) + C(product_id_str) + C(weekday_str) + C(month_str)",
            "target": "log_positive_sales",
            "scale": "log1p销量",
            "inverse_log": True,
            "data": model_df,
        },
        "周末补充_无星期FE回归": {
            "formula": "log_positive_sales ~ C(weather_group, Treatment(reference='no_precip')) + avg_temperature + temperature_range + wind_power + is_holiday + is_weekend + is_activity_day + lag_7 + rolling_28_prev + C(store_id_str) + C(product_id_str) + C(month_str)",
            "target": "log_positive_sales",
            "scale": "log1p销量",
            "inverse_log": True,
            "data": model_df,
        },
        "去异常日_对数销量_历史控制FE回归": {
            "formula": "log_positive_sales ~ C(weather_group, Treatment(reference='no_precip')) + avg_temperature + temperature_range + wind_power + is_holiday + is_activity_day + lag_7 + rolling_28_prev + C(store_id_str) + C(product_id_str) + C(weekday_str) + C(month_str)",
            "target": "log_positive_sales",
            "scale": "log1p销量；去日总销量异常日",
            "inverse_log": True,
            "data": model_df[model_df["is_daily_total_outlier"] == 0].copy(),
        },
        "标准化销量_历史控制FE回归": {
            "formula": "standardized_sales ~ C(weather_group, Treatment(reference='no_precip')) + avg_temperature + temperature_range + wind_power + is_holiday + is_activity_day + lag_7 + rolling_28_prev + C(store_id_str) + C(product_id_str) + C(weekday_str) + C(month_str)",
            "target": "standardized_sales",
            "scale": "门店-商品内标准化销量",
            "inverse_log": False,
            "data": model_df,
        },
    }

    coefficient_frames = []
    metric_rows = []
    summary_rows = []
    for name, spec in formulas.items():
        data = spec["data"].copy()
        full_model = fit_cluster_ols(data, spec["formula"])
        coefficient_frames.append(
            extract_external_coefficients(full_model, name, iqr_map, spec["scale"])
        )
        summary_rows.append(
            {
                "method": name,
                "nobs": int(full_model.nobs),
                "r_squared": float(full_model.rsquared),
                "adj_r_squared": float(full_model.rsquared_adj),
                "target_scale": spec["scale"],
                "note": "标准误按日期聚类；系数用于统计关联解释。",
            }
        )

        train_i = data[data["date"] < VALIDATION_START].copy()
        valid_i = data[data["date"] >= VALIDATION_START].copy()
        if len(train_i) > 0 and len(valid_i) > 0:
            _, m = evaluate_formula(
                train_i,
                valid_i,
                spec["formula"],
                TARGET if spec["inverse_log"] else spec["target"],
                "row_level_store_product",
                inverse_log=spec["inverse_log"],
            )
            m.update(
                {
                    "method": name,
                    "train_rows": len(train_i),
                    "validation_rows": len(valid_i),
                }
            )
            metric_rows.append(m)

    coef_table = pd.concat(coefficient_frames, ignore_index=True)
    metric_table = pd.DataFrame(metric_rows)
    summary_table = pd.DataFrame(summary_rows)

    daily_train = daily[daily["date"] < VALIDATION_START].copy()
    daily_valid = daily[daily["date"] >= VALIDATION_START].copy()
    desc_mean = daily_train["daily_total_sales"].mean()
    desc_pred = np.repeat(desc_mean, len(daily_valid))
    desc_metric = metrics(
        daily_valid["daily_total_sales"].to_numpy(), desc_pred, "daily_total"
    )
    desc_metric.update(
        {
            "method": "简单描述性分析：训练期日均销量",
            "train_rows": len(daily_train),
            "validation_rows": len(daily_valid),
        }
    )
    metric_table = pd.concat([pd.DataFrame([desc_metric]), metric_table], ignore_index=True)

    return summary_table, coef_table, metric_table, iqr_map


def robustness_by_group(analysis: pd.DataFrame, group_col: str) -> pd.DataFrame:
    model_df = analysis.dropna(subset=["lag_7", "rolling_28_prev"]).copy()
    rows = []
    for group_value, sub in model_df.groupby(group_col, observed=False):
        if len(sub) < 500 or sub["date"].nunique() < 100:
            continue
        controls = [
            "C(weather_group, Treatment(reference='no_precip'))",
            "avg_temperature",
            "temperature_range",
            "wind_power",
            "is_holiday",
            "is_activity_day",
            "lag_7",
            "rolling_28_prev",
            "C(weekday_str)",
            "C(month_str)",
        ]
        if group_col != "store_id":
            controls.append("C(store_id_str)")
        if group_col != "category":
            controls.append("C(product_id_str)")
        formula = "log_positive_sales ~ " + " + ".join(controls)
        try:
            model = fit_cluster_ols(sub, formula)
        except Exception as exc:  # pragma: no cover - diagnostic only
            rows.append(
                {
                    group_col: group_value,
                    "nobs": len(sub),
                    "status": f"模型失败：{type(exc).__name__}",
                }
            )
            continue
        def get(term: str, attr: str = "params") -> float:
            obj = getattr(model, attr)
            return float(obj.get(term, np.nan))

        weather_terms = [t for t in model.params.index if t.startswith("C(weather_group")]
        rows.append(
            {
                group_col: group_value,
                "nobs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "activity_coef": get("is_activity_day"),
                "activity_p": get("is_activity_day", "pvalues"),
                "holiday_coef": get("is_holiday"),
                "holiday_p": get("is_holiday", "pvalues"),
                "avg_temperature_coef": get("avg_temperature"),
                "avg_temperature_p": get("avg_temperature", "pvalues"),
                "weather_max_abs_coef": max([abs(float(model.params[t])) for t in weather_terms], default=np.nan),
                "weather_terms_count": len(weather_terms),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def build_method_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "方法": "分组均值对比（日总销量）",
                "是否控制门店": "否",
                "是否控制商品": "否",
                "是否控制星期": "否",
                "是否控制月份": "否",
                "是否能解释方向": "能，粗略",
                "是否能比较影响大小": "能，均值差；不稳健",
                "是否适合论文": "适合作为现象描述",
                "主要局限": "不控制混杂因素，不能作为主结论。",
            },
            {
                "方法": "原方案详细天气固定效应回归",
                "是否控制门店": "是",
                "是否控制商品": "是",
                "是否控制星期": "是",
                "是否控制月份": "是",
                "是否能解释方向": "能",
                "是否能比较影响大小": "能",
                "是否适合论文": "可写，但需降级为补充",
                "主要局限": "详细天气类别稀疏，雨夹雪、暴雨等样本天数过少，排序不稳定。",
            },
            {
                "方法": "合并天气 + 对数销量 + 历史控制固定效应回归",
                "是否控制门店": "是",
                "是否控制商品": "是",
                "是否控制星期": "是",
                "是否控制月份": "是",
                "是否能解释方向": "能",
                "是否能比较影响大小": "能",
                "是否适合论文": "建议作为主分析",
                "主要局限": "仍然是观察数据，不能证明因果；活动日可能存在内生性。",
            },
            {
                "方法": "温度分箱固定效应回归",
                "是否控制门店": "是",
                "是否控制商品": "是",
                "是否控制星期": "是",
                "是否控制月份": "是",
                "是否能解释方向": "能，按温度区间",
                "是否能比较影响大小": "能",
                "是否适合论文": "适合作为非线性检查",
                "主要局限": "分箱边界依赖样本分位数，不能过度解释为精确阈值。",
            },
            {
                "方法": "去除异常销量日后回归",
                "是否控制门店": "是",
                "是否控制商品": "是",
                "是否控制星期": "是",
                "是否控制月份": "是",
                "是否能解释方向": "能",
                "是否能比较影响大小": "能",
                "是否适合论文": "适合作为稳健性检查",
                "主要局限": "异常日可能是真实业务高峰，删除后结论只能作敏感性分析。",
            },
            {
                "方法": "分门店/分类别稳健性回归",
                "是否控制门店": "分门店时不需要；分类别时控制",
                "是否控制商品": "视分组内商品数而定",
                "是否控制星期": "是",
                "是否控制月份": "是",
                "是否能解释方向": "能",
                "是否能比较影响大小": "能，主要比较符号稳定性",
                "是否适合论文": "适合作为辅助表",
                "主要局限": "分组后样本变少，个别类别估计不稳定。",
            },
            {
                "方法": "随机森林置换重要性",
                "是否控制门店": "是，作为特征",
                "是否控制商品": "是，作为特征",
                "是否控制星期": "是，作为特征",
                "是否控制月份": "是，作为特征",
                "是否能解释方向": "不能",
                "是否能比较影响大小": "能，比较预测贡献",
                "是否适合论文": "适合作为补充",
                "主要局限": "只能说明预测贡献，不能说明方向或因果。",
            },
            {
                "方法": "SHAP / LightGBM",
                "是否控制门店": "可控制",
                "是否控制商品": "可控制",
                "是否控制星期": "可控制",
                "是否控制月份": "可控制",
                "是否能解释方向": "可解释局部方向，但表达复杂",
                "是否能比较影响大小": "能",
                "是否适合论文": "本项目不建议作为主方法",
                "主要局限": "增加论文表达难度；当前题目更需要稳健统计关联而非黑箱解释。",
            },
        ]
    )


def build_framework_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "方法类别": "简单描述性分析",
                "方法目的": "观察不同外部状态下销量均值是否存在差异，作为现象展示。",
                "输入变量": "日总销量、天气合并类别、是否节假日、是否周末、是否活动日、温度分箱。",
                "控制了哪些因素": "不控制，仅按外部变量分组。",
                "没控制哪些因素": "门店、商品、星期、月份、历史销量、未观测客流和库存。",
                "可解释性": "高，均值差直观。",
                "局限性": "容易把周末、节假日、活动日和季节差异混在一起。",
                "是否适合写入论文主结论": "不适合主结论，适合作为引入和描述性证据。",
            },
            {
                "方法类别": "控制变量固定效应回归",
                "方法目的": "在控制基础需求差异后估计外部因素与销量的条件统计关联。",
                "输入变量": "门店-商品日销量、天气合并类别、温度、风力、节假日、活动日、星期、月份、历史销量。",
                "控制了哪些因素": "门店、商品、星期、月份、历史销量水平；标准误按日期聚类。",
                "没控制哪些因素": "客流、库存、竞品、价格策略细节、活动安排内生性。",
                "可解释性": "高，可解释方向和可比效应。",
                "局限性": "观察数据仍不能证明因果；类别合并会损失细节。",
                "是否适合写入论文主结论": "适合，建议作为主分析。",
            },
            {
                "方法类别": "机器学习特征重要性分析",
                "方法目的": "检验非线性模型中外部变量是否对验证集预测误差有贡献。",
                "输入变量": "天气、温度、风力、节假日、周末、活动日、门店、商品、类别、星期、月份、历史销量。",
                "控制了哪些因素": "以特征形式纳入门店、商品、类别、星期、月份和历史销量。",
                "没控制哪些因素": "同样不能控制未观测混杂；不提供清晰参数方向。",
                "可解释性": "中，只能解释预测贡献。",
                "局限性": "置换重要性可能受特征相关性影响，不能说明因果和方向。",
                "是否适合写入论文主结论": "不适合主结论，适合作为补充。",
            },
        ]
    )


def build_practical_evaluation(metric_table: pd.DataFrame) -> pd.DataFrame:
    meta = {
        "简单描述性分析：训练期日均销量": ("低", "高", "高", "仅日总量预测参考，不控制混杂。"),
        "原方案详细天气FE回归": ("中", "高", "较高", "可解释方向，但详细天气样本稀疏。"),
        "合并天气_对数销量_历史控制FE回归": ("中", "高", "高", "建议主分析；控制更充分，表达难度可控。"),
        "温度分箱_对数销量_历史控制FE回归": ("中", "高", "高", "适合检查非线性温度关系。"),
        "周末补充_无星期FE回归": ("中", "高", "较高", "只能作为周末效应补充，不能与星期固定效应同放。"),
        "去异常日_对数销量_历史控制FE回归": ("中", "高", "高", "稳健性检查，验证活动日结论是否依赖异常日。"),
        "标准化销量_历史控制FE回归": ("中", "中", "中", "目标为标准化销量，指标不可与原始销量模型直接比较。"),
        "RandomForest grouped weather + history": ("中高", "中", "中", "适合作为预测贡献补充，不解释方向。"),
    }
    rows = []
    for _, row in metric_table.iterrows():
        method = row["method"]
        complexity, interpretability, writability, note = meta.get(
            method, ("中", "中", "中", "")
        )
        rows.append(
            {
                "方法": method,
                "MAE": row.get("MAE"),
                "RMSE": row.get("RMSE"),
                "WAPE": row.get("WAPE"),
                "运行复杂度": complexity,
                "可解释性": interpretability,
                "论文可写性": writability,
                "说明": note,
            }
        )
    return pd.DataFrame(rows)


def build_importance_figure(coef_table: pd.DataFrame, rf_importance: pd.DataFrame) -> pd.DataFrame:
    factor_map = {
        "weather_group": "天气",
        "avg_temperature": "平均温度",
        "temperature_range": "昼夜温差",
        "wind_power": "风力",
        "is_holiday": "节假日",
        "is_activity_day": "活动日",
        "is_weekend": "周末",
        "temp_bin": "温度分箱",
    }
    main = coef_table[
        coef_table["model"] == "合并天气_对数销量_历史控制FE回归"
    ].copy()
    reg = (
        main.assign(factor_cn=main["factor"].map(factor_map))
        .dropna(subset=["factor_cn"])
        .groupby("factor_cn")["comparable_abs_effect"]
        .max()
        .reset_index(name="regression_comparable_effect")
    )
    rf = (
        rf_importance.assign(
            factor_cn=rf_importance["feature"].map(factor_map),
            importance_clipped=rf_importance["importance_mean_mae_increase"].clip(lower=0),
        )
        .dropna(subset=["factor_cn"])
        .groupby("factor_cn")["importance_clipped"]
        .sum()
        .reset_index(name="rf_permutation_importance")
    )
    comp = pd.merge(reg, rf, on="factor_cn", how="outer").fillna(0.0)
    for col in ["regression_comparable_effect", "rf_permutation_importance"]:
        max_value = comp[col].max()
        comp[col + "_scaled"] = comp[col] / max_value if max_value > 0 else 0.0
    comp = comp.sort_values(
        ["regression_comparable_effect_scaled", "rf_permutation_importance_scaled"],
        ascending=False,
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(comp))
    width = 0.36
    ax.barh(
        y - width / 2,
        comp["regression_comparable_effect_scaled"],
        width,
        label="控制回归可比效应（归一化）",
        color="#3A78B7",
    )
    ax.barh(
        y + width / 2,
        comp["rf_permutation_importance_scaled"],
        width,
        label="随机森林置换重要性（归一化）",
        color="#CC6677",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(comp["factor_cn"])
    ax.invert_yaxis()
    ax.set_xlabel("归一化强度")
    ax.set_title("问题三外部因素重要性比较")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_factor_importance_comparison.png", dpi=180)
    plt.close(fig)
    return comp


def build_notebook() -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 问题三方法优化探索\n",
                "\n",
                "本 Notebook 复现 `src/experimental/q3_method_search.py` 的核心流程。目标不是直接证明因果，而是在描述性分析、控制变量回归、稳健性检查和机器学习特征重要性之间比较哪种方法更适合作为问题三主分析。",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 读取数据与构造变量\n",
                "\n",
                "外部变量来自 `modeling_base_table.csv`。天气、节假日、活动日是日期层面变量；销量是门店-商品层面变量，因此要特别注意同一天外部变量被重复用于多行样本。标准误需要按日期聚类。",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import sys\n",
                "ROOT = Path.cwd()\n",
                "while not (ROOT / 'AGENTS.md').exists():\n",
                "    ROOT = ROOT.parent\n",
                "sys.path.insert(0, str(ROOT))\n",
                "from src.experimental.q3_method_search import prepare_data\n",
                "analysis, daily, temp_bin_meta = prepare_data()\n",
                "analysis.shape, daily.shape, temp_bin_meta\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 描述性分组比较\n",
                "\n",
                "描述性均值用于观察现象，但没有控制门店、商品、星期和月份，因此不能作为因果结论。",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.experimental.q3_method_search import group_binary_diff, holiday_weekend_crosstab\n",
                "group_binary_diff(daily).head(20)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["holiday_weekend_crosstab(daily)\n"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 控制变量和固定效应回归\n",
                "\n",
                "主推荐模型使用合并天气类别、`log1p` 销量、门店/商品/星期/月固定效应，并加入只使用过去信息构造的 `lag_7` 与 `rolling_28_prev`。这样做的目的不是提高预测精度，而是降低基础需求差异和历史销量水平的干扰。",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.experimental.q3_method_search import run_regression_suite\n",
                "summary, coef_table, metric_table, iqr_map = run_regression_suite(analysis, daily)\n",
                "summary\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "coef_table[coef_table['model'].eq('合并天气_对数销量_历史控制FE回归')].head(20)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 稳健性检查\n",
                "\n",
                "分门店、分类别和去除日总销量异常日后的结果用于判断结论是否稳定。若一个因素只在少数组别显著，论文中应降低结论强度。",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.experimental.q3_method_search import robustness_by_group\n",
                "store_robustness = robustness_by_group(analysis, 'store_id')\n",
                "category_robustness = robustness_by_group(analysis, 'category')\n",
                "store_robustness\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["category_robustness\n"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 机器学习特征重要性\n",
                "\n",
                "随机森林置换重要性用于判断非线性预测贡献。它不能说明方向，也不能证明因果，因此只适合作为辅助分析。",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.experimental.q3_method_search import run_random_forest\n",
                "rf_metrics, rf_importance = run_random_forest(analysis)\n",
                "rf_metrics, rf_importance.head(15)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 结论口径\n",
                "\n",
                "本问更适合写成“统计关联分析”。若使用“影响”一词，应明确限定为观察数据中的条件关联，不写成“导致”。",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "q3_method_search.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_reports(
    analysis: pd.DataFrame,
    daily: pd.DataFrame,
    temp_bin_meta: pd.DataFrame,
    descriptive_table: pd.DataFrame,
    holiday_weekend: pd.DataFrame,
    activity_weekday: pd.DataFrame,
    top_activity_days: pd.DataFrame,
    regression_summary: pd.DataFrame,
    coef_table: pd.DataFrame,
    metric_table: pd.DataFrame,
    store_robustness: pd.DataFrame,
    category_robustness: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    rf_importance: pd.DataFrame,
    package_status: pd.DataFrame,
    importance_comparison: pd.DataFrame,
    framework_comparison: pd.DataFrame,
    practical_evaluation: pd.DataFrame,
) -> None:
    original_report = (ROOT / "outputs" / "stage4_q3_factor_analysis_report.md").read_text(
        encoding="utf-8"
    )
    direct_causal_terms = [
        line.strip()
        for line in original_report.splitlines()
        if any(word in line for word in ["导致", "因果影响", "证明"])
    ]

    current_summary = """## 当前问题三已采用的方法概括

1. 外部变量构造方式：使用天气类别、最高温、最低温、风力、节假日、周末、活动日、星期、月份、门店、商品和类别字段；另构造平均温度与昼夜温差。湿度字段不存在，原方案没有补造湿度。
2. 描述性分析方法：先将门店-商品销量汇总到日期层面，比较天气类型、节假日、周末、活动日下的平均日总销量；并计算温度、风力、昼夜温差与日总销量的 Pearson 相关。
3. 回归或机器学习模型：原方案使用带门店、商品、星期、月份控制变量的 OLS 回归，并按日期聚类标准误；同时使用随机森林置换重要性作为非线性预测贡献补充。
4. 影响大小排序：原方案按回归可比效应排序时，天气排第一、活动日第二、最低温第三、最高温第四；但天气第一主要受雨夹雪等稀有天气类别影响。
5. 主要结论：活动日与销量上升的统计关联较稳定；温度方向在控制后存在一定关联；节假日和周末在预测贡献上较弱；风力不稳定；天气类别存在样本不均衡。
6. 可能存在的因果夸大风险：原报告主体已经反复使用“统计关联”限制语，直接因果化风险较低；但题目和章节中“影响”“效应”等词仍可能被读者理解成因果，需要在论文中统一改成“统计关联”或“条件关联”。
"""

    causal_review = f"""# 问题三因果语言与核心任务审查

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{current_summary}

## 本问性质判断

1. 本问更接近解释问题，而不是纯预测问题。预测模型只回答“哪些变量有助于预测”，不能直接回答“某因素是否造成销量变化”。
2. 在当前数据条件下，“影响”应写成统计关联或条件统计关联。数据不是随机实验，也没有断点、双重差分、工具变量等准实验设计。
3. 当前方案已经控制了门店、商品、星期、月份，并用日期聚类标准误降低同一天外部变量重复出现带来的显著性夸大；但仍不能控制客流、竞品、价格调整、库存、商圈事件等未观测混杂因素。
4. 当前方案区分了天气、节假日、活动日和星期效应：主模型用星期固定效应吸收周内规律，周末变量只在补充模型中估计，这是合理的。
5. 节假日和周末确实存在混淆风险。当前数据中节假日既可能落在周末，也可能落在工作日，因此论文应使用“控制星期后节假日变量的条件关联”，不能把简单节假日均值差直接解释为节假日效应。
6. 活动日也存在与高销量日混淆的风险。数据字典说明活动日表示促销活动日，但如果促销安排本身选择在预期高销量日，则活动日系数会混合促销、预期需求和其他活动安排因素。

## 原报告中直接因果表述扫描

自动检索到的含因果关键词句子如下。需要说明：其中多数是原报告中的限制性说明，不是违规表述；真正需要避免的是把“统计关联”写成“导致”。

{df_to_md(pd.DataFrame({'句子': direct_causal_terms}) if direct_causal_terms else pd.DataFrame(columns=['句子']))}

## 关键数据审查结果

| 项目 | 结果 |
| --- | --- |
| 外部变量完整样本行数 | {len(analysis)} |
| 外部变量完整日期数 | {analysis['date'].nunique()} |
| 日期层面异常销量日数 | {int(daily['is_daily_total_outlier'].sum())} |
| 湿度字段 | 不存在，不能分析湿度分箱 |
| 活动日天数 | {int(daily['is_activity_day'].sum())} |
| 节假日天数 | {int(daily['is_holiday'].sum())} |
| 周末天数 | {int(daily['is_weekend'].sum())} |

## 节假日与周末关系

{df_to_md(holiday_weekend)}

## 活动日与星期分布

{df_to_md(activity_weekday, max_rows=20)}

## 高销量日与活动日关系

{df_to_md(top_activity_days, max_rows=20)}

结论：活动日与销量上升的关系值得保留，但不能写成促销活动“导致”销量增加。更严谨的写法是：在控制门店、商品、星期、月份和历史销量水平后，活动日与销量上升之间仍存在较稳定的统计关联。
"""

    rewrite_rows = [
        {
            "来源": "题目/报告任务表述",
            "原句或倾向": "分析并比较天气、节假日、活动日等因素对零食销量的影响。",
            "风险": "容易被理解为因果影响。",
            "建议改写": "分析并比较天气、节假日、活动日等因素与零食销量之间的统计关联。",
        },
        {
            "来源": "原报告章节标题",
            "原句或倾向": "如何控制混杂因素。",
            "风险": "控制变量回归只能缓解已观测混杂，不能完全控制所有混杂。",
            "建议改写": "如何降低已观测混杂因素的干扰。",
        },
        {
            "来源": "活动日结论",
            "原句或倾向": "活动日销量明显更高。",
            "风险": "若直接解释为促销带来销量，存在内生性风险。",
            "建议改写": "活动日的平均销量较高；在控制门店、商品、星期、月份和历史销量水平后，活动日仍与销量上升存在统计关联。",
        },
        {
            "来源": "天气结论",
            "原句或倾向": "天气是影响最大的因素。",
            "风险": "详细天气类别样本极不均衡，最大系数来自少数稀有天气。",
            "建议改写": "详细天气类别的回归系数差异较大，但部分天气样本天数很少；合并天气类别后，天气变量更适合作为辅助结论。",
        },
        {
            "来源": "节假日结论",
            "原句或倾向": "节假日使销量增加。",
            "风险": "节假日与周末、星期效应重叠，不能从均值差直接推出节假日因果作用。",
            "建议改写": "节假日样本的销量均值高于非节假日；但在控制星期后，节假日变量的稳定性弱于活动日。",
        },
        {
            "来源": "随机森林解释",
            "原句或倾向": "随机森林证明某变量影响销量。",
            "风险": "置换重要性只度量预测贡献，不给出方向和因果解释。",
            "建议改写": "随机森林置换重要性显示该变量对验证集预测误差有贡献，但不能说明销量变化方向或因果关系。",
        },
        {
            "来源": "新分析温度分箱",
            "原句或倾向": "某温度区间导致销量变化。",
            "风险": "温度区间来自样本分位数，且与季节、节假日、天气共同变化。",
            "建议改写": "在样本分位数温度区间下，不同温度段与销量存在条件关联，不能解释为精确温度阈值。",
        },
    ]
    rewrite_table = pd.DataFrame(rewrite_rows)

    main_coef = coef_table[
        coef_table["model"] == "合并天气_对数销量_历史控制FE回归"
    ].copy()
    no_outlier_coef = coef_table[
        coef_table["model"] == "去异常日_对数销量_历史控制FE回归"
    ].copy()
    temp_coef = coef_table[
        coef_table["model"] == "温度分箱_对数销量_历史控制FE回归"
    ].copy()

    stable_activity_store = int(
        ((store_robustness["activity_coef"] > 0) & (store_robustness["activity_p"] < 0.05)).sum()
    )
    stable_activity_category = int(
        ((category_robustness["activity_coef"] > 0) & (category_robustness["activity_p"] < 0.05)).sum()
    )

    report = f"""# 问题三方法优化探索报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 原问题三方案的不足

原方案已经包含描述性统计、控制变量回归和随机森林置换重要性，方向是合理的；主要不足不在于“方法太简单”，而在于主排序仍使用详细天气类别，容易让少数稀有天气支配结论。例如雨夹雪只有 3 天，暴雨只有极少天数，这类系数不宜作为主结论排序依据。

第二个不足是原方案使用原始正向销量作为回归目标。销量分布高度偏斜，日总销量存在异常高峰，原始销量回归容易受极端值影响。第三个不足是原方案虽然控制了门店、商品、星期和月份，但没有把历史销量水平作为控制变量；本次补充了 `lag_7` 与 `rolling_28_prev`，且它们只使用过去销量，不使用未来信息。

## 2. 哪种方法最适合做主分析

建议主分析采用“合并天气 + 对数销量 + 历史控制固定效应回归”。该方法控制门店、商品、星期、月份和历史销量水平，同时把详细天气合并为更稳定的天气组，减少稀有天气类别造成的排序不稳。

主模型外部变量系数节选如下：

{df_to_md(main_coef[['factor','variable','coef','p_value','direction','comparable_abs_effect']].head(15))}

## 3. 哪种方法适合做辅助分析

1. 分组均值对比适合作为现象描述，帮助读者直观看到天气、节假日、周末、活动日的均值差异。
2. 温度分箱回归适合作为非线性检查，说明温度与销量不一定是简单线性关系，但分箱边界不能写成精确业务阈值。
3. 去除异常销量日后的回归适合作为稳健性检查。
4. 随机森林置换重要性适合作为非线性预测贡献补充，不适合作为主结论。

## 三类方法比较框架

{df_to_md(framework_comparison, max_rows=10)}

## 4. 哪些变量影响较稳定

活动日是本次最稳定的外部变量。分门店稳健性中，活动日系数为正且显著的门店数为 {stable_activity_store} / {len(store_robustness)}；分类别稳健性中，活动日系数为正且显著的类别数为 {stable_activity_category} / {len(category_robustness)}。这支持写成“活动日与销量上升之间存在较稳定的统计关联”。

温度变量有一定关联，但方向和表达要谨慎。连续温度、温度分箱和月份控制之间存在重叠，温度结论适合作为辅助而非最强结论。

## 5. 哪些变量结论不够稳

详细天气类别不够稳。原方案中“天气”排序靠前，主要来自少数稀有天气的较大系数；合并天气类别后更适合表达为“天气条件存在一定关联，但具体天气类型差异受样本量限制”。

节假日和周末结论也不够稳。它们与星期效应、休息日安排高度相关，随机森林置换重要性不高，因此不宜写成强结论。风力在回归和机器学习中的稳定性也弱，不建议作为主要影响因素。

## 6. 是否建议替换原问题三方案

建议部分替换。不是推翻原方案，而是把“详细天气固定效应回归 + 原始销量”从主分析降为补充，把“合并天气 + 对数销量 + 历史控制固定效应回归”提升为主分析。

推荐论文结构为：描述性统计用于展示现象；主回归用于给出条件统计关联；温度分箱、去异常日、分门店/分类别结果用于稳健性检查；随机森林置换重要性用于辅助说明预测贡献。

## 7. 论文中应该如何表述“影响”

建议统一写成“统计关联”“条件关联”“与销量变化相关”。如果保留“影响”一词，需要在首次出现时说明：本文所称影响是观察数据中的统计关联，不代表严格因果效应。

## 8. 哪些结论必须避免因果化

1. 不能写“促销活动导致销量显著增加”。
2. 不能写“某天气导致销量上升/下降”。
3. 不能写“节假日导致销量增加”。
4. 不能把随机森林重要性解释为因果影响。
5. 不能分析湿度影响，因为数据中没有湿度字段。

## 9. 是否建议加入固定效应或控制变量回归

建议加入，而且应作为问题三主方法。固定效应思想适合本题：门店固定效应控制门店基础客流差异，商品固定效应控制商品基础需求差异，星期固定效应控制周内规律，月份固定效应控制季节性。历史销量水平控制可进一步减少基础需求状态差异。

## 10. 是否建议加入机器学习特征重要性作为补充

建议加入随机森林置换重要性作为补充，但不要把它放在主结论前面。置换重要性可以说明某个变量对预测误差的贡献，但不能给出方向，更不能证明因果。

## 方法预测指标参考

这些指标用于比较方法稳定性和预测贡献，不是问题三主目标。

{df_to_md(metric_table[['method','analysis_level','train_rows','validation_rows','MAE','RMSE','WAPE']], max_rows=20)}

## 指标、复杂度与论文可写性

{df_to_md(practical_evaluation, max_rows=20)}

## 回归模型摘要

{df_to_md(regression_summary, max_rows=20)}

## 去异常日稳健性系数节选

{df_to_md(no_outlier_coef[['factor','variable','coef','p_value','direction','comparable_abs_effect']].head(12))}

## 温度分箱非线性检查

温度分箱边界如下：

{df_to_md(temp_bin_meta)}

温度分箱回归系数节选如下：

{df_to_md(temp_coef[['factor','variable','coef','p_value','direction','comparable_abs_effect']].head(12))}

## 随机森林置换重要性

随机森林验证指标：

{df_to_md(rf_metrics)}

置换重要性前 15 项：

{df_to_md(rf_importance.head(15))}

## 环境可用性

{df_to_md(package_status)}
"""

    causal_rewrite = f"""# 问题三因果化表述改写建议

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

当前阶段 4 报告整体已经比较克制，直接“导致”类表述很少。以下列出仍可能被误解为因果化的句子或写作倾向，以及建议改写。

{df_to_md(rewrite_table, max_rows=20)}

## 写作原则

1. 使用“统计关联”“条件关联”“相关”替代“导致”“带来”“促成”。
2. 描述性均值只能写“均值更高/更低”，不能写“因素使销量变化”。
3. 控制变量回归可以写“在控制若干因素后仍有关联”，不能写“已经证明影响”。
4. 随机森林置换重要性只能写“预测贡献”，不能写“影响方向”。
"""

    OUTPUTS.joinpath("q3_causal_language_review.md").write_text(
        causal_review, encoding="utf-8"
    )
    OUTPUTS.joinpath("q3_causal_rewrite_suggestions.md").write_text(
        causal_rewrite, encoding="utf-8"
    )
    OUTPUTS.joinpath("q3_method_search_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    analysis, daily, temp_bin_meta = prepare_data()

    descriptive_table = group_binary_diff(daily)
    holiday_weekend = holiday_weekend_crosstab(daily)
    activity_weekday, top_activity_days = activity_diagnostics(daily)
    regression_summary, coef_table, metric_table, _ = run_regression_suite(analysis, daily)
    store_robustness = robustness_by_group(analysis, "store_id")
    category_robustness = robustness_by_group(analysis, "category")
    rf_metrics, rf_importance = run_random_forest(analysis)
    package_status = import_status()
    method_comparison = build_method_comparison()
    framework_comparison = build_framework_comparison()
    metric_with_rf = pd.concat([metric_table, rf_metrics], ignore_index=True)
    practical_evaluation = build_practical_evaluation(metric_with_rf)
    importance_comparison = build_importance_figure(coef_table, rf_importance)

    descriptive_table.to_csv(TABLES / "q3_descriptive_group_comparison.csv", index=False, encoding="utf-8-sig")
    holiday_weekend.to_csv(TABLES / "q3_holiday_weekend_crosstab.csv", index=False, encoding="utf-8-sig")
    activity_weekday.to_csv(TABLES / "q3_activity_weekday_distribution.csv", index=False, encoding="utf-8-sig")
    top_activity_days.to_csv(TABLES / "q3_top_sales_days_activity_check.csv", index=False, encoding="utf-8-sig")
    regression_summary.to_csv(TABLES / "q3_regression_robustness_summary.csv", index=False, encoding="utf-8-sig")
    coef_table.to_csv(TABLES / "q3_regression_coefficients_robustness.csv", index=False, encoding="utf-8-sig")
    metric_with_rf.to_csv(TABLES / "q3_method_validation_metrics.csv", index=False, encoding="utf-8-sig")
    store_robustness.to_csv(TABLES / "q3_store_robustness.csv", index=False, encoding="utf-8-sig")
    category_robustness.to_csv(TABLES / "q3_category_robustness.csv", index=False, encoding="utf-8-sig")
    rf_metrics.to_csv(TABLES / "q3_rf_method_search_metrics.csv", index=False, encoding="utf-8-sig")
    rf_importance.to_csv(TABLES / "q3_rf_method_search_permutation_importance.csv", index=False, encoding="utf-8-sig")
    package_status.to_csv(TABLES / "q3_ml_explainability_environment.csv", index=False, encoding="utf-8-sig")
    method_comparison.to_csv(TABLES / "q3_factor_method_comparison.csv", index=False, encoding="utf-8-sig")
    framework_comparison.to_csv(TABLES / "q3_analysis_framework_comparison.csv", index=False, encoding="utf-8-sig")
    practical_evaluation.to_csv(TABLES / "q3_method_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    importance_comparison.to_csv(TABLES / "q3_factor_importance_comparison.csv", index=False, encoding="utf-8-sig")
    temp_bin_meta.to_csv(TABLES / "q3_temperature_bin_definition.csv", index=False, encoding="utf-8-sig")

    build_reports(
        analysis=analysis,
        daily=daily,
        temp_bin_meta=temp_bin_meta,
        descriptive_table=descriptive_table,
        holiday_weekend=holiday_weekend,
        activity_weekday=activity_weekday,
        top_activity_days=top_activity_days,
        regression_summary=regression_summary,
        coef_table=coef_table,
        metric_table=metric_with_rf,
        store_robustness=store_robustness,
        category_robustness=category_robustness,
        rf_metrics=rf_metrics,
        rf_importance=rf_importance,
        package_status=package_status,
        importance_comparison=importance_comparison,
        framework_comparison=framework_comparison,
        practical_evaluation=practical_evaluation,
    )
    build_notebook()

    summary = {
        "analysis_rows": int(len(analysis)),
        "analysis_dates": int(analysis["date"].nunique()),
        "daily_outlier_days": int(daily["is_daily_total_outlier"].sum()),
        "recommended_primary_method": "合并天气 + 对数销量 + 历史控制固定效应回归",
        "replace_original": "建议部分替换",
        "key_output": str(OUTPUTS / "q3_method_search_report.md"),
    }
    (OUTPUTS / "q3_method_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
