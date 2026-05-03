from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import wape

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
PAPER = ROOT / "paper"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")


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


def extract_weather_level(term: str) -> str:
    match = re.search(r"\[T\.(.*)\]", term)
    return match.group(1) if match else term


def regression_external_coefficients(model, iqr_map: dict[str, float]) -> pd.DataFrame:
    rows = []
    for term, coef in model.params.items():
        if term == "Intercept":
            continue
        if term.startswith("C(weather"):
            factor = "weather"
            variable = extract_weather_level(term)
            direction = "正关联" if coef > 0 else "负关联" if coef < 0 else "近零"
            comparable_effect = abs(float(coef))
            interpretation = f"相对基准天气的销量差异"
        elif term in ["max_temperature", "min_temperature", "wind_power"]:
            factor = term
            variable = term
            direction = "正关联" if coef > 0 else "负关联" if coef < 0 else "近零"
            comparable_effect = abs(float(coef) * iqr_map.get(term, 1.0))
            interpretation = "连续变量按四分位距变化折算"
        elif term in ["is_holiday", "is_activity_day", "is_weekend"]:
            factor = term
            variable = term
            direction = "正关联" if coef > 0 else "负关联" if coef < 0 else "近零"
            comparable_effect = abs(float(coef))
            interpretation = "0/1 变量从 0 到 1 的关联差异"
        else:
            continue
        rows.append(
            {
                "term": term,
                "factor": factor,
                "variable": variable,
                "coef": float(coef),
                "std_err": float(model.bse.get(term, np.nan)),
                "t_value": float(model.tvalues.get(term, np.nan)),
                "p_value": float(model.pvalues.get(term, np.nan)),
                "direction": direction,
                "comparable_abs_effect": comparable_effect,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows).sort_values("comparable_abs_effect", ascending=False)


def build_notebook() -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 06 外部因素统计关联分析\n",
                    "\n",
                    "本 Notebook 对应阶段 4：问题三建模。目标是分析天气、节假日、活动日等因素与销量之间的统计关联，不直接证明因果关系。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 1：读取建模基础表\n",
                    "\n",
                    "这段代码解决“外部因素字段是否真实存在”的问题。特别注意：原始附件没有湿度字段，所以后续不分析湿度。",
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
                    "df = pd.read_csv(ROOT / 'data/processed/modeling_base_table.csv')\n",
                    "df['date'] = pd.to_datetime(df['date'])\n",
                    "external_cols = ['weather','max_temperature','min_temperature','wind_power','is_holiday','is_weekend','is_activity_day','weekday','month','store_id','product_id','category']\n",
                    "df[external_cols + ['positive_sales']].head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：如果上述字段存在，就可以进入问题三变量构造。`humidity` 不在字段中，不能补造湿度变量。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 2：构造日总销量表\n",
                    "\n",
                    "这段代码解决描述性分析的粒度问题。由于天气、节假日、活动日是日期层面的变量，先把所有门店商品汇总成日总销量，再比较不同外部因素下的均值。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "daily = pd.read_csv(ROOT / 'tables/q3_daily_external_factor_table.csv')\n",
                    "daily.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：每一行是一日总销量及该日外部因素。这个表用于天气、节假日、周末、活动日的均值比较。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 3：查看天气描述性统计\n",
                    "\n",
                    "这段代码解决“不同天气下销量是否有差异”的问题，只是描述性比较，不控制混杂因素。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "weather_stats = pd.read_csv(ROOT / 'tables/q3_weather_descriptive_stats.csv')\n",
                    "weather_stats.head(10)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：`mean_daily_sales` 是该天气下平均日销量。样本天数少的天气结论更不稳定，不能过度解释。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 4：读取控制变量回归结果\n",
                    "\n",
                    "这段代码解决“控制门店、商品、星期、月份后，外部因素是否仍有关联”的问题。回归系数表示统计关联，不表示因果影响。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "reg = pd.read_csv(ROOT / 'tables/q3_regression_coefficients_external.csv')\n",
                    "reg[['factor','variable','coef','p_value','direction','comparable_abs_effect']].head(15)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：`coef` 为控制变量后的回归系数。连续变量的大小需结合四分位距折算；天气系数是相对基准天气的差异。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 5：读取随机森林置换重要性\n",
                    "\n",
                    "这段代码解决“非线性模型下哪些特征对预测更重要”的问题。置换重要性表示打乱某个特征后模型误差上升多少，不能说明销量变化方向。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "rf_imp = pd.read_csv(ROOT / 'tables/q3_random_forest_permutation_importance.csv')\n",
                    "rf_imp.head(12)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：重要性越高，说明该特征对随机森林验证集预测贡献越大。门店、商品等控制变量可能排名较高，这反映基础需求差异。",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 代码 6：读取统计关联强度排序\n",
                    "\n",
                    "这段代码解决论文最终如何排序的问题。排序综合考虑回归可比效应和随机森林重要性，并明确稳定性与局限。",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "summary = pd.read_csv(ROOT / 'tables/q3_factor_summary.csv')\n",
                    "summary\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "输出理解：该表可直接作为问题三“统计关联强度排序”的依据。论文中应使用“关联”表述，不写“导致”。",
                ],
            },
        ],
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
    (NOTEBOOKS / "06_external_factor_analysis.ipynb").write_text(
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

    df = pd.read_csv(ROOT / "data/processed/modeling_base_table.csv")
    df["date"] = pd.to_datetime(df["date"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)

    analysis_df = df[df["has_external_data"] == 1].copy()
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
        "category",
        TARGET,
    ]
    analysis_df = analysis_df.dropna(subset=required)
    analysis_df["avg_temperature"] = (
        analysis_df["max_temperature"] + analysis_df["min_temperature"]
    ) / 2
    analysis_df["temperature_range"] = (
        analysis_df["max_temperature"] - analysis_df["min_temperature"]
    )
    analysis_df["store_id_str"] = analysis_df["store_id"].astype(str)
    analysis_df["product_id_str"] = analysis_df["product_id"].astype(str)
    analysis_df["weekday_str"] = analysis_df["weekday"].astype(int).astype(str)
    analysis_df["month_str"] = analysis_df["month"].astype(int).astype(str)
    analysis_df["weather"] = analysis_df["weather"].astype(str)
    analysis_df["category"] = analysis_df["category"].astype(str)

    variable_rows = [
        ("天气类型", "weather", "存在", "附件二 `天气` 字段，分类变量"),
        ("温度", "max_temperature/min_temperature/avg_temperature", "存在", "最高温、最低温及平均温"),
        ("湿度", "humidity", "不存在", "原始附件和处理表均无湿度字段，不能分析"),
        ("风力", "wind_power", "存在", "附件二 `风力` 字段，单位待确认"),
        ("是否节假日", "is_holiday", "存在", "由节日字段转换"),
        ("是否周末", "is_weekend", "存在", "由日期星期转换"),
        ("是否活动日", "is_activity_day", "存在", "人工确认为门店促销活动日"),
        ("星期", "weekday", "存在", "周一=1，周日=7"),
        ("月份", "month", "存在", "日期月份"),
        ("门店", "store_id", "存在", "门店编号控制变量"),
        ("商品", "product_id", "存在", "商品编号控制变量"),
        ("商品类别", "category", "存在", "类别控制或分组变量"),
    ]
    variable_availability = pd.DataFrame(
        variable_rows, columns=["requested_variable", "field", "availability", "note"]
    )
    save_csv(variable_availability, "q3_variable_availability.csv", True)

    daily_external = (
        analysis_df.groupby(
            [
                "date",
                "weather",
                "max_temperature",
                "min_temperature",
                "avg_temperature",
                "temperature_range",
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
    save_csv(daily_external, "q3_daily_external_factor_table.csv")

    weather_stats = (
        daily_external.groupby("weather")["daily_total_sales"]
        .agg(
            n_days="count",
            mean_daily_sales="mean",
            median_daily_sales="median",
            std_daily_sales="std",
            min_daily_sales="min",
            max_daily_sales="max",
            total_sales="sum",
        )
        .reset_index()
        .sort_values(["mean_daily_sales", "n_days"], ascending=[False, False])
    )
    save_csv(weather_stats, "q3_weather_descriptive_stats.csv", True)

    binary_rows = []
    binary_specs = [
        ("is_holiday", "节假日", "非节假日", "节假日"),
        ("is_weekend", "周末", "工作日", "周末"),
        ("is_activity_day", "活动日", "非活动日", "活动日"),
    ]
    for col, factor_label, zero_label, one_label in binary_specs:
        grouped = (
            daily_external.groupby(col)["daily_total_sales"]
            .agg(n_days="count", mean_daily_sales="mean", median_daily_sales="median", total_sales="sum")
            .reset_index()
        )
        value_map = {0: zero_label, 1: one_label}
        for _, row in grouped.iterrows():
            binary_rows.append(
                {
                    "factor": factor_label,
                    "value": int(row[col]),
                    "label": value_map.get(int(row[col]), str(row[col])),
                    "n_days": int(row["n_days"]),
                    "mean_daily_sales": float(row["mean_daily_sales"]),
                    "median_daily_sales": float(row["median_daily_sales"]),
                    "total_sales": float(row["total_sales"]),
                }
            )
    binary_stats = pd.DataFrame(binary_rows)
    binary_diff_rows = []
    for factor in binary_stats["factor"].unique():
        g = binary_stats[binary_stats["factor"] == factor]
        if set(g["value"]) >= {0, 1}:
            mean0 = float(g.loc[g["value"] == 0, "mean_daily_sales"].iloc[0])
            mean1 = float(g.loc[g["value"] == 1, "mean_daily_sales"].iloc[0])
            binary_diff_rows.append(
                {
                    "factor": factor,
                    "mean_when_0": mean0,
                    "mean_when_1": mean1,
                    "mean_diff_1_minus_0": mean1 - mean0,
                    "relative_diff_pct": (mean1 - mean0) / mean0 * 100 if mean0 else np.nan,
                }
            )
    binary_diff = pd.DataFrame(binary_diff_rows)
    save_csv(binary_stats, "q3_binary_factor_descriptive_stats.csv", True)
    save_csv(binary_diff, "q3_binary_factor_mean_differences.csv", True)

    corr_rows = []
    for col, label in [
        ("max_temperature", "最高温"),
        ("min_temperature", "最低温"),
        ("avg_temperature", "平均温"),
        ("temperature_range", "昼夜温差"),
        ("wind_power", "风力"),
    ]:
        corr_rows.append(
            {
                "factor": label,
                "field": col,
                "pearson_corr_with_daily_sales": float(
                    daily_external[col].corr(daily_external["daily_total_sales"])
                ),
                "q1": float(daily_external[col].quantile(0.25)),
                "q3": float(daily_external[col].quantile(0.75)),
                "iqr": float(
                    daily_external[col].quantile(0.75)
                    - daily_external[col].quantile(0.25)
                ),
            }
        )
    continuous_corr = pd.DataFrame(corr_rows)
    save_csv(continuous_corr, "q3_continuous_factor_correlations.csv", True)

    weather_reference = "晴" if "晴" in set(analysis_df["weather"]) else analysis_df["weather"].mode().iloc[0]
    formula_controlled = (
        f"{TARGET} ~ C(weather, Treatment(reference='{weather_reference}')) "
        "+ max_temperature + min_temperature + wind_power "
        "+ is_holiday + is_activity_day "
        "+ C(weekday_str) + C(month_str) + C(store_id_str) + C(product_id_str)"
    )
    controlled_model = smf.ols(formula_controlled, data=analysis_df).fit(
        cov_type="cluster", cov_kwds={"groups": analysis_df["date"]}
    )

    formula_weekend = (
        f"{TARGET} ~ C(weather, Treatment(reference='{weather_reference}')) "
        "+ max_temperature + min_temperature + wind_power "
        "+ is_holiday + is_weekend + is_activity_day "
        "+ C(month_str) + C(store_id_str) + C(product_id_str)"
    )
    weekend_model = smf.ols(formula_weekend, data=analysis_df).fit(
        cov_type="cluster", cov_kwds={"groups": analysis_df["date"]}
    )

    formula_category = (
        f"{TARGET} ~ C(weather, Treatment(reference='{weather_reference}')) "
        "+ max_temperature + min_temperature + wind_power "
        "+ is_holiday + is_activity_day "
        "+ C(weekday_str) + C(month_str) + C(store_id_str) + C(category)"
    )
    category_model = smf.ols(formula_category, data=analysis_df).fit(
        cov_type="cluster", cov_kwds={"groups": analysis_df["date"]}
    )

    iqr_map = {
        col: float(analysis_df[col].quantile(0.75) - analysis_df[col].quantile(0.25))
        for col in ["max_temperature", "min_temperature", "wind_power"]
    }
    reg_external = regression_external_coefficients(controlled_model, iqr_map)
    weekend_external = regression_external_coefficients(weekend_model, iqr_map)
    weekend_row = weekend_external[weekend_external["factor"] == "is_weekend"].copy()
    if not weekend_row.empty:
        weekend_row["term"] = "is_weekend_from_weekend_model"
        reg_external = pd.concat([reg_external, weekend_row], ignore_index=True)
    weather_day_counts = weather_stats.set_index("weather")["n_days"].to_dict()
    reg_external["level_n_days"] = np.nan
    weather_mask = reg_external["factor"] == "weather"
    reg_external.loc[weather_mask, "level_n_days"] = reg_external.loc[
        weather_mask, "variable"
    ].map(weather_day_counts)
    reg_external["sample_size_note"] = ""
    reg_external.loc[
        weather_mask & (reg_external["level_n_days"] < 10), "sample_size_note"
    ] = "该天气样本天数少，系数需谨慎解释"
    reg_external = reg_external.sort_values("comparable_abs_effect", ascending=False)
    save_csv(reg_external, "q3_regression_coefficients_external.csv", True)

    regression_model_summary = pd.DataFrame(
        [
            {
                "model": "controlled_product_fixed_effect",
                "description": "天气、温度、风力、节假日、活动日 + 星期、月份、门店、商品控制变量",
                "nobs": int(controlled_model.nobs),
                "r_squared": float(controlled_model.rsquared),
                "adj_r_squared": float(controlled_model.rsquared_adj),
                "weather_reference": weather_reference,
                "note": "主要解释模型；标准误按日期聚类",
            },
            {
                "model": "weekend_effect_model",
                "description": "估计周末变量；不再加入星期固定效应，避免完全共线",
                "nobs": int(weekend_model.nobs),
                "r_squared": float(weekend_model.rsquared),
                "adj_r_squared": float(weekend_model.rsquared_adj),
                "weather_reference": weather_reference,
                "note": "用于周末/工作日关联估计",
            },
            {
                "model": "category_control_model",
                "description": "用商品类别替代商品固定效应的补充模型",
                "nobs": int(category_model.nobs),
                "r_squared": float(category_model.rsquared),
                "adj_r_squared": float(category_model.rsquared_adj),
                "weather_reference": weather_reference,
                "note": "产品固定效应会吸收类别差异，因此单独给出类别控制模型",
            },
        ]
    )
    save_csv(regression_model_summary, "q3_regression_model_summary.csv", True)

    train = analysis_df[analysis_df["date"] < VALIDATION_START].copy()
    valid = analysis_df[analysis_df["date"] >= VALIDATION_START].copy()
    rf_features = [
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_holiday",
        "is_weekend",
        "is_activity_day",
        "weekday",
        "month",
        "store_id_str",
        "product_id_str",
        "category",
    ]
    numeric_features = [
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_holiday",
        "is_weekend",
        "is_activity_day",
        "weekday",
        "month",
    ]
    categorical_features = ["weather", "store_id_str", "product_id_str", "category"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_model.fit(train[rf_features], train[TARGET])
    rf_pred = np.maximum(0, rf_model.predict(valid[rf_features]))
    rf_metrics = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "validation_start": str(VALIDATION_START.date()),
                "MAE": float(mean_absolute_error(valid[TARGET], rf_pred)),
                "RMSE": float(mean_squared_error(valid[TARGET], rf_pred) ** 0.5),
                "WAPE": float(wape(valid[TARGET], rf_pred)),
                "R2": float(r2_score(valid[TARGET], rf_pred)),
                "note": "用于非线性特征重要性分析，不用于证明因果",
            }
        ]
    )
    save_csv(rf_metrics, "q3_random_forest_metrics.csv", True)

    perm = permutation_importance(
        rf_model,
        valid[rf_features],
        valid[TARGET],
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_importance = pd.DataFrame(
        {
            "feature": rf_features,
            "importance_mean_mae_increase": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_mae_increase", ascending=False)
    rf_importance["feature_group"] = rf_importance["feature"].replace(
        {
            "store_id_str": "门店",
            "product_id_str": "商品",
            "category": "商品类别",
            "weather": "天气",
            "max_temperature": "最高温",
            "min_temperature": "最低温",
            "wind_power": "风力",
            "is_holiday": "节假日",
            "is_weekend": "周末",
            "is_activity_day": "活动日",
            "weekday": "星期",
            "month": "月份",
        }
    )
    save_csv(rf_importance, "q3_random_forest_permutation_importance.csv", True)

    def descriptive_diff(field: str) -> float | None:
        row = binary_diff[binary_diff["factor"] == field]
        if row.empty:
            return None
        return float(row["mean_diff_1_minus_0"].iloc[0])

    factor_rows = []
    factor_specs = [
        ("天气", "weather"),
        ("最高温", "max_temperature"),
        ("最低温", "min_temperature"),
        ("风力", "wind_power"),
        ("节假日", "is_holiday"),
        ("周末", "is_weekend"),
        ("活动日", "is_activity_day"),
    ]
    for display, factor in factor_specs:
        main_limitation = "存在混杂因素，只能解释为统计关联；天气和日历变量为日期层面变量"
        if factor == "weather":
            sub = reg_external[reg_external["factor"] == "weather"]
            effect = float(sub["comparable_abs_effect"].max()) if not sub.empty else 0.0
            direction = "分天气类型不同"
            p_value = float(sub["p_value"].min()) if not sub.empty else np.nan
            top_weather_level = (
                str(sub.sort_values("comparable_abs_effect", ascending=False)["variable"].iloc[0])
                if not sub.empty
                else ""
            )
            top_weather_days = weather_day_counts.get(top_weather_level, np.nan)
            if pd.notna(top_weather_days) and top_weather_days < 10:
                main_limitation = (
                    f"天气类别样本不均衡；最大系数来自 {top_weather_level}，仅 {int(top_weather_days)} 天"
                )
        else:
            sub = reg_external[reg_external["factor"] == factor]
            if sub.empty:
                effect, direction, p_value = 0.0, "无法估计", np.nan
            else:
                effect = float(sub["comparable_abs_effect"].iloc[0])
                direction = str(sub["direction"].iloc[0])
                p_value = float(sub["p_value"].iloc[0])
        rf_row = rf_importance[rf_importance["feature"] == factor]
        rf_imp = (
            float(rf_row["importance_mean_mae_increase"].iloc[0])
            if not rf_row.empty
            else np.nan
        )
        if factor in ["is_holiday", "is_activity_day"]:
            desc = descriptive_diff("节假日" if factor == "is_holiday" else "活动日")
        elif factor == "is_weekend":
            desc = descriptive_diff("周末")
        else:
            desc = None
        if factor == "weather" and "样本不均衡" in main_limitation:
            stable = "不稳定/部分天气样本少"
        elif pd.isna(p_value) or p_value >= 0.05:
            stable = "不稳定/统计不显著"
        elif pd.isna(rf_imp) or rf_imp <= 0:
            stable = "不稳定/预测贡献弱"
        else:
            stable = "较稳定"
        factor_rows.append(
            {
                "factor": display,
                "regression_direction": direction,
                "regression_comparable_abs_effect": effect,
                "min_p_value": p_value,
                "rf_permutation_importance": rf_imp,
                "descriptive_mean_diff_if_binary": desc,
                "stability_judgement": stable,
                "causal_interpretation_allowed": "否",
                "main_limitation": main_limitation,
            }
        )
    factor_summary = pd.DataFrame(factor_rows).sort_values(
        ["regression_comparable_abs_effect", "rf_permutation_importance"],
        ascending=[False, False],
    )
    factor_summary.insert(0, "rank_by_regression_effect", range(1, len(factor_summary) + 1))
    save_csv(factor_summary, "q3_factor_summary.csv", True)

    # Figures.
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_weather = weather_stats.sort_values("mean_daily_sales", ascending=True)
    ax.barh(plot_weather["weather"], plot_weather["mean_daily_sales"], color="#3A78B7")
    ax.set_title("图14 不同天气下平均日销量")
    ax.set_xlabel("平均日销量")
    for i, row in enumerate(plot_weather.itertuples()):
        ax.text(row.mean_daily_sales, i, f"{row.mean_daily_sales:.1f} ({row.n_days}天)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_weather_mean_sales.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, factor in zip(axes, ["节假日", "周末", "活动日"]):
        sub = binary_stats[binary_stats["factor"] == factor].sort_values("value")
        ax.bar(sub["label"], sub["mean_daily_sales"], color=["#88CCEE", "#CC6677"])
        ax.set_title(f"{factor}与平均日销量")
        ax.set_ylabel("平均日销量")
        for i, row in enumerate(sub.itertuples()):
            ax.text(i, row.mean_daily_sales, f"{row.mean_daily_sales:.1f}\n{row.n_days}天", ha="center", va="bottom", fontsize=8)
    fig.suptitle("图15 节假日、周末、活动日描述性比较", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_binary_factor_mean_sales.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(
        axes,
        ["avg_temperature", "wind_power", "temperature_range"],
        ["平均温度", "风力", "昼夜温差"],
    ):
        ax.scatter(daily_external[col], daily_external["daily_total_sales"], s=14, alpha=0.55, color="#4477AA")
        coef = np.polyfit(daily_external[col], daily_external["daily_total_sales"], 1)
        xs = np.linspace(daily_external[col].min(), daily_external[col].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#CC6677", linewidth=2)
        ax.set_title(f"{title}与日销量")
        ax.set_xlabel(title)
        ax.set_ylabel("日总销量")
    fig.suptitle("图16 连续外部变量与日总销量关系", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_continuous_factor_scatter.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_reg = factor_summary.sort_values("regression_comparable_abs_effect", ascending=True)
    ax.barh(plot_reg["factor"], plot_reg["regression_comparable_abs_effect"], color="#66AA55")
    ax.set_title("图17 控制变量回归的外部因素可比关联强度")
    ax.set_xlabel("可比绝对效应")
    for i, row in enumerate(plot_reg.itertuples()):
        ax.text(row.regression_comparable_abs_effect, i, f"{row.regression_comparable_abs_effect:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_regression_factor_effect_ranking.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_rf = rf_importance.head(12).sort_values("importance_mean_mae_increase", ascending=True)
    ax.barh(plot_rf["feature_group"], plot_rf["importance_mean_mae_increase"], color="#AA4499")
    ax.set_title("图18 随机森林置换重要性")
    ax.set_xlabel("打乱该特征后的 MAE 增加")
    for i, row in enumerate(plot_rf.itertuples()):
        ax.text(row.importance_mean_mae_increase, i, f"{row.importance_mean_mae_increase:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_random_forest_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    weather_coeffs = reg_external[reg_external["factor"] == "weather"].copy()
    if not weather_coeffs.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_wc = weather_coeffs.sort_values("coef")
        colors = ["#CC6677" if v < 0 else "#4477AA" for v in plot_wc["coef"]]
        ax.barh(plot_wc["variable"], plot_wc["coef"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"图19 天气类别回归系数（相对 {weather_reference}）")
        ax.set_xlabel("控制变量后的系数")
        fig.tight_layout()
        fig.savefig(FIGURES / "q3_weather_regression_coefficients.png", bbox_inches="tight")
        plt.close(fig)

    # Markdown report.
    top_weather = weather_stats.head(6)[
        ["weather", "n_days", "mean_daily_sales", "median_daily_sales", "std_daily_sales"]
    ]
    reg_report = reg_external[
        [
            "factor",
            "variable",
            "coef",
            "p_value",
            "direction",
            "comparable_abs_effect",
            "level_n_days",
            "sample_size_note",
        ]
    ].head(15)
    rf_report = rf_importance[
        ["feature_group", "feature", "importance_mean_mae_increase", "importance_std"]
    ].head(12)
    factor_report = factor_summary[
        [
            "rank_by_regression_effect",
            "factor",
            "regression_direction",
            "regression_comparable_abs_effect",
            "rf_permutation_importance",
            "stability_judgement",
            "main_limitation",
        ]
    ]
    binary_diff_report = binary_diff.copy()
    continuous_report = continuous_corr.copy()

    report = f"""# 阶段 4：问题三外部因素统计关联分析报告

执行日期：2026-05-02

## 1. 问题三核心任务

题目要求分析并比较天气、节假日、活动日等因素对零食销量的影响。由于当前数据不具备因果识别条件，本阶段将这一要求界定为外部因素与销量之间的统计关联分析：先进行描述性比较，再建立带门店、商品、星期、月份控制变量的回归模型，并用随机森林置换重要性作为非线性补充。

本阶段仍使用 `positive_sales` 作为销量口径。负销量已在前序阶段确认为损耗或冲销类调整，不代表顾客正向购买需求。

## 2. 什么是外部因素统计关联分析

外部因素统计关联分析的直觉是：在销量变化时，检查天气、节假日、活动日等变量是否也发生系统性变化。数学上，本阶段使用两类方法：一是描述性统计，比较不同外部状态下的平均日销量；二是回归模型，在控制门店、商品、星期、月份后，估计外部变量与销量之间的条件统计关联。

## 3. 为什么不能轻易说因果影响

当前数据不是随机实验，也没有明确的准实验识别设计。天气、节假日、促销活动和客流变化可能同时出现，门店和商品本身也存在固定需求差异。因此，即使某个外部变量的回归系数显著，也只能写为“在控制若干因素后呈现统计关联”，不能写为“该因素导致销量增加”。

## 4. 如何控制混杂因素

本阶段在主要回归中加入门店固定效应、商品固定效应、星期固定效应和月份固定效应：

1. 门店固定效应用于控制门店位置、规模、客流等长期差异。
2. 商品固定效应用于控制商品基础需求、价格带、品类属性等长期差异。由于每个商品属于一个类别，商品固定效应会吸收类别差异；类别控制模型作为补充模型单独给出。
3. 星期固定效应用于控制周内规律。周末变量是星期变量的组合，因此主要控制模型中不单独估计周末系数；周末效应用不含星期固定效应的补充模型估计。
4. 月份固定效应用于控制季节和月份需求差异。

## 5. 变量构造

变量可用性如下：

{df_to_md(variable_availability, 20)}

需要特别说明：附件二没有湿度字段，因此本阶段无法分析湿度与销量的关系，不能在论文中伪造湿度变量或湿度结论。

## 6. 描述性分析

不同天气下平均日销量最高的若干天气类型如下：

{df_to_md(top_weather, 10)}

节假日、周末、活动日的描述性均值差异如下：

{df_to_md(binary_diff_report, 10)}

连续变量与日销量的 Pearson 相关系数如下：

{df_to_md(continuous_report, 10)}

图14展示不同天气下平均日销量，用于观察天气类型与销量均值差异；图15展示节假日、周末、活动日的均值差异；图16展示平均温度、风力、昼夜温差与日总销量的散点关系。描述性统计没有控制门店、商品、星期、月份等因素，因此只作为现象观察。

## 7. 回归模型

主要控制变量回归模型为：

$$
y_{{i,t}}=\\beta_0+\\beta_1Weather_t+\\beta_2Temp_t+\\beta_3Wind_t+\\beta_4Holiday_t+\\beta_5Activity_t+\\gamma_s+\\delta_p+\\eta_w+\\mu_m+\\varepsilon_{{i,t}}
$$

其中，$y_{{i,t}}$ 表示第 $t$ 天某门店-商品组合的正向销量，$Weather_t$ 为天气类型，$Temp_t$ 包括最高温和最低温，$Wind_t$ 为风力，$Holiday_t$ 为节假日变量，$Activity_t$ 为活动日变量，$\\gamma_s$、$\\delta_p$、$\\eta_w$、$\\mu_m$ 分别表示门店、商品、星期和月份控制变量。标准误按日期聚类，以降低同一天外部变量重复出现造成的显著性夸大。

回归模型摘要：

{df_to_md(regression_model_summary, 10)}

外部变量回归系数节选如下：

{df_to_md(reg_report, 15)}

## 8. 随机森林特征重要性

本阶段补充随机森林模型，原因是外部因素与销量之间可能存在非线性统计关系，例如温度过高或过低时，销量与温度的关联不一定保持线性。随机森林使用的特征包括天气、最高温、最低温、风力、节假日、周末、活动日、星期、月份、门店、商品和商品类别。该模型用于特征重要性分析，不用于证明因果关系。

LightGBM 在当前环境中不可用；SHAP 也不可用。考虑到本题本科数学建模论文需要可解释、可复现，且回归模型已经提供关联方向解释，因此本阶段不用 SHAP，改用 sklearn 的置换重要性。置换重要性表示打乱某一特征后验证集 MAE 增加多少，只能说明预测贡献，不能说明销量变化方向。

随机森林验证集指标：

{df_to_md(rf_metrics, 5)}

随机森林置换重要性前若干项：

{df_to_md(rf_report, 12)}

## 9. 统计关联强度排序

综合控制变量回归的可比效应和随机森林置换重要性，外部因素统计关联强度排序如下：

{df_to_md(factor_report, 20)}

解释时应注意：回归可比效应用于判断统计关联方向和大致强度，随机森林重要性用于补充判断非线性预测贡献。两者一致时，结论相对稳定；两者不一致时，应写为“不稳定或需要进一步验证”。

## 10. 结论和限制

1. 当前可以支持的表述是“在控制门店、商品、星期、月份后，部分外部因素与销量呈现统计关联”。
2. 不能写“天气导致销量增加”或“活动日导致销量增加”，因为没有随机实验或准实验识别设计。
3. 湿度字段不存在，当前数据无法分析湿度。
4. 周末效应与星期固定效应存在确定关系，因此主要模型中用星期固定效应控制周内规律，周末变量只在补充模型中估计。
5. 商品固定效应会吸收商品类别差异，因此类别变量不能与商品固定效应同时解释为独立类别效应；类别控制模型只作为补充。
"""
    (OUTPUTS / "stage4_q3_factor_analysis_report.md").write_text(report, encoding="utf-8")

    q3_model_building = """## 问题三模型建立

问题三分析天气、节假日、活动日等外部因素与零食销量之间的统计关联。本文将外部因素构造为分类变量和连续变量：天气类型为分类变量，最高温、最低温和风力为连续变量，节假日、周末、活动日为 0-1 变量，星期、月份、门店、商品和类别作为控制或分组变量。原始附件中没有湿度字段，因此不构造湿度变量。

### 描述性统计模型

对日期层面总销量 $Y_t$，比较不同外部状态下的均值：

$$
\\bar Y_g=\\frac{1}{n_g}\\sum_{t\\in g}Y_t
$$

其中，$g$ 表示天气、节假日、周末或活动日分组。该方法直观，但不能控制门店、商品和星期等混杂因素。

### 控制变量回归模型

主要回归模型设为：

$$
y_{i,t}=\\beta_0+\\beta_1Weather_t+\\beta_2Temp_t+\\beta_3Wind_t+\\beta_4Holiday_t+\\beta_5Activity_t+\\gamma_s+\\delta_p+\\eta_w+\\mu_m+\\varepsilon_{i,t}
$$

其中，$y_{i,t}$ 表示第 $t$ 天门店-商品组合 $i$ 的销量，$Weather_t$ 为天气类型，$Temp_t$ 为温度变量，$Wind_t$ 为风力，$Holiday_t$ 和 $Activity_t$ 分别为节假日和活动日变量，$\\gamma_s$、$\\delta_p$、$\\eta_w$、$\\mu_m$ 分别控制门店、商品、星期和月份差异。

商品固定效应会吸收类别差异，因此类别变量不在主要模型中与商品固定效应同时解释；本文另建立类别控制模型作为补充。周末变量由星期决定，因此在主要模型中通过星期固定效应控制周内规律，并在补充模型中单独估计周末关联。

### 随机森林特征重要性

为补充线性回归无法表达的非线性统计关系，本文建立随机森林回归模型，并用置换重要性衡量各特征的预测贡献。随机森林重要性只表示预测贡献，不表示因果影响，也不提供关联方向解释。
"""

    q3_model_solution = f"""## 问题三模型求解

本阶段使用 `modeling_base_table.csv` 中成功匹配外部变量的样本。由于 2022-03-31 没有附件二外部变量，本阶段外部因素统计关联分析仅使用外部变量完整的日期。

变量可用性如下：

{df_to_md(variable_availability, 20)}

控制变量回归结果摘要如下：

{df_to_md(regression_model_summary, 10)}

外部变量回归系数节选如下：

{df_to_md(reg_report, 15)}

随机森林模型以 2022-03-01 为验证期起点，验证结果如下：

{df_to_md(rf_metrics, 5)}

随机森林置换重要性前若干项如下：

{df_to_md(rf_report, 12)}
"""

    q3_result_analysis = f"""## 问题三结果分析

描述性统计显示，不同天气、节假日、周末和活动日下的平均日销量存在差异。但这些差异可能与门店结构、商品结构、星期规律和月份季节性同时相关，因此不能仅凭描述性均值作因果解释。

控制变量回归将门店、商品、星期和月份纳入模型后，外部因素的统计关联强度排序如下：

{df_to_md(factor_report, 20)}

从解释边界看，本文只能得到统计关联结论。例如，可以写“在控制门店、商品、星期和月份后，某外部变量与销量变化呈现统计关联”；不能写“天气导致销量增加”。此外，湿度字段不存在，当前数据无法支持湿度相关结论。周末变量与星期固定效应存在共线关系，因此周末关联只在补充模型中估计。

随机森林置换重要性用于补充判断非线性预测贡献。它能说明打乱某一特征后预测误差增加多少，但不能说明销量变化方向，也不能证明因果关系。因此，问题三最终结论以回归的统计关联方向为主，以随机森林重要性作为辅助排序依据。
"""

    upsert_section(PAPER / "model_building.md", "## 问题三模型建立", q3_model_building)
    upsert_section(PAPER / "model_solution.md", "## 问题三模型求解", q3_model_solution)
    upsert_section(PAPER / "result_analysis.md", "## 问题三结果分析", q3_result_analysis)
    build_notebook()

    best_factor = factor_summary.iloc[0]
    row = (
        f"| 2026-05-02 | 阶段 4 问题三外部因素统计关联分析 | processed: modeling_base_table.csv | "
        f"描述性统计、控制变量回归、随机森林置换重要性 | 天气、最高温、最低温、风力、节假日、周末、活动日、星期、月份、门店、商品、类别 | "
        f"控制变量回归 R2={controlled_model.rsquared:.3f}；随机森林验证 WAPE={rf_metrics['WAPE'].iloc[0]*100:.2f}% | "
        f"已完成外部因素变量构造、描述性分析、回归分析和特征重要性排序；按回归可比关联强度排序首位为 {best_factor['factor']}，但需结合稳定性解释 | "
        f"湿度字段不存在；结论只能表述为统计关联，不能写因果；周末与星期固定效应存在共线关系 | "
        f"停止在阶段 4，等待确认后进入阶段 5 问题四综合模型 |"
    )
    prepend_result_log(row)

    summary = {
        "analysis_rows": int(len(analysis_df)),
        "analysis_dates": int(analysis_df["date"].nunique()),
        "excluded_rows_no_external_data": int(len(df) - len(analysis_df)),
        "humidity_available": False,
        "weather_reference": weather_reference,
        "controlled_regression_r2": float(controlled_model.rsquared),
        "controlled_regression_adj_r2": float(controlled_model.rsquared_adj),
        "random_forest_wape": float(rf_metrics["WAPE"].iloc[0]),
        "top_factor_by_regression_effect": str(best_factor["factor"]),
        "top_rf_feature": str(rf_importance.iloc[0]["feature_group"]),
        "factor_summary_rows": int(len(factor_summary)),
    }
    (OUTPUTS / "stage4_q3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
