"""
Preprocessing helpers for the snack sales forecasting project.

The functions in this module convert transaction-level sales details into a
daily store-product panel and merge date-level external variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SALES_RENAME = {
    "门店编号": "store_id",
    "门店名称": "store_name",
    "商品代码": "product_id",
    "商品名称": "product_name",
    "规格": "spec",
    "单位": "unit",
    "数量": "quantity",
    "实际零售价": "actual_retail_price",
    "实际销售额": "actual_sales_amount",
    "销售金额": "sales_amount",
    "折扣金额": "discount_amount",
    "收款时间": "payment_time",
    "类别代码": "category_code",
    "类别名称": "category",
}

WEATHER_RENAME = {
    "日期": "date",
    "天气": "weather",
    "温度": "max_temperature",
    "温度.1": "min_temperature",
    "风力": "wind_power",
    "节日": "holiday_name",
    "星期几": "weekday_raw",
    "活动日": "is_activity_day",
    "工休日": "is_rest_day",
}


PANEL_KEYS = [
    "date",
    "store_id",
    "store_name",
    "product_id",
    "product_name",
    "category_code",
    "category",
    "spec",
    "unit",
]


@dataclass
class QualitySummary:
    """Compact quality summary used in stage reports."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_by_column: Dict[str, int]
    dtypes: Dict[str, str]


def summarize_frame(df: pd.DataFrame) -> QualitySummary:
    """Build a simple row/column/missing/duplicate summary."""
    return QualitySummary(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_by_column={col: int(df[col].isna().sum()) for col in df.columns},
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )


def standardize_sales(raw_sales: pd.DataFrame) -> pd.DataFrame:
    """Standardize sales column names and add date fields."""
    missing = [col for col in SALES_RENAME if col not in raw_sales.columns]
    if missing:
        raise KeyError(f"Sales attachment is missing required columns: {missing}")

    sales = raw_sales.rename(columns=SALES_RENAME).copy()
    sales["payment_time"] = pd.to_datetime(sales["payment_time"], errors="coerce")
    sales["date"] = sales["payment_time"].dt.normalize()
    sales["weekday"] = sales["date"].dt.dayofweek + 1
    sales["is_weekend"] = sales["weekday"].isin([6, 7]).astype(int)
    sales["month"] = sales["date"].dt.month
    sales["is_negative_quantity"] = sales["quantity"] < 0
    return sales


def standardize_weather(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Standardize weather and calendar columns.

    The original workbook has two columns both named 温度. pandas renames the
    second one to 温度.1. Human review confirmed that they mean daily maximum
    and minimum temperature.
    """
    missing = [col for col in WEATHER_RENAME if col not in raw_weather.columns]
    if missing:
        raise KeyError(f"Weather attachment is missing required columns: {missing}")

    weather = raw_weather.rename(columns=WEATHER_RENAME).copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather["weekday_from_date"] = weather["date"].dt.dayofweek + 1
    weather["month_from_date"] = weather["date"].dt.month

    holiday = weather["holiday_name"].astype(str).str.strip()
    weather["holiday_name"] = holiday.where(~holiday.isin(["0", "0.0", "", "nan"]), "")
    weather["is_holiday"] = (weather["holiday_name"] != "").astype(int)

    for col in ["is_activity_day", "is_rest_day"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce").astype("Int64")

    return weather


def build_daily_store_product_panel(
    standardized_sales: pd.DataFrame,
    complete_dates: bool = True,
) -> pd.DataFrame:
    """Aggregate transactions to a date-store-product daily panel.

    daily_sales is net quantity, so negative transaction quantities reduce the
    daily total. The human review confirmed that negative quantities should be
    treated as loss/write-off style adjustments such as damaged or expired goods,
    not ordinary customer demand. To keep the data auditable, the output also
    records positive quantity, negative adjustment quantity, and negative record
    count.
    """
    sales = standardized_sales.copy()

    grouped = (
        sales.groupby(PANEL_KEYS, dropna=False)
        .agg(
            daily_sales=("quantity", "sum"),
            daily_amount=("actual_sales_amount", "sum"),
            sales_amount=("sales_amount", "sum"),
            discount_amount=("discount_amount", "sum"),
            transaction_count=("quantity", "size"),
            positive_sales=("quantity", lambda s: s[s > 0].sum()),
            negative_adjustment_qty=("quantity", lambda s: -s[s < 0].sum()),
            negative_record_count=("quantity", lambda s: int((s < 0).sum())),
            actual_retail_price_mean=("actual_retail_price", "mean"),
        )
        .reset_index()
    )

    grouped["price"] = np.where(
        grouped["daily_sales"] > 0,
        grouped["daily_amount"] / grouped["daily_sales"],
        np.nan,
    )

    if not complete_dates:
        panel = grouped
    else:
        date_frame = pd.DataFrame(
            {
                "date": pd.date_range(
                    sales["date"].min(),
                    sales["date"].max(),
                    freq="D",
                )
            }
        )
        combos = sales[PANEL_KEYS[1:]].drop_duplicates().reset_index(drop=True)
        date_frame["_key"] = 1
        combos["_key"] = 1
        full_index = date_frame.merge(combos, on="_key").drop(columns="_key")
        panel = full_index.merge(grouped, on=PANEL_KEYS, how="left")

        zero_cols = [
            "daily_sales",
            "daily_amount",
            "sales_amount",
            "discount_amount",
            "transaction_count",
            "positive_sales",
            "negative_adjustment_qty",
            "negative_record_count",
        ]
        panel[zero_cols] = panel[zero_cols].fillna(0)

    panel["daily_sales"] = panel["daily_sales"].astype(float)
    for col in [
        "transaction_count",
        "positive_sales",
        "negative_adjustment_qty",
        "negative_record_count",
    ]:
        panel[col] = panel[col].astype(int)

    panel["weekday"] = panel["date"].dt.dayofweek + 1
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(int)
    panel["month"] = panel["date"].dt.month
    panel["has_sales_record"] = (panel["transaction_count"] > 0).astype(int)
    return panel.sort_values(["date", "store_id", "product_id", "product_name"])


def mark_daily_sales_outliers(panel: pd.DataFrame) -> pd.DataFrame:
    """Mark extreme daily_sales values by product using the IQR rule.

    The function only adds flags and thresholds. It does not delete outliers.
    """
    df = panel.copy()
    stats = (
        df.groupby(["product_id", "product_name"], dropna=False)["daily_sales"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
        .reset_index()
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["outlier_lower"] = stats["q1"] - 1.5 * stats["iqr"]
    stats["outlier_upper"] = stats["q3"] + 1.5 * stats["iqr"]
    df = df.merge(stats, on=["product_id", "product_name"], how="left")
    df["daily_sales_outlier"] = (
        (df["daily_sales"] < df["outlier_lower"])
        | (df["daily_sales"] > df["outlier_upper"])
    ).astype(int)
    return df


def merge_external_variables(
    daily_panel: pd.DataFrame,
    standardized_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge date-level weather/calendar variables into the daily panel."""
    weather_cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "holiday_name",
        "weekday_raw",
        "weekday_from_date",
        "month_from_date",
        "is_activity_day",
        "is_rest_day",
        "is_holiday",
    ]
    weather = standardized_weather[weather_cols].drop_duplicates("date")
    merged = daily_panel.merge(weather, on="date", how="left")
    merged["has_external_data"] = merged["weather"].notna().astype(int)
    return merged


def one_to_one_violations(
    df: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    """Find values in left that map to more than one value in right."""
    counts = df.groupby(left, dropna=False)[right].nunique(dropna=False)
    bad_keys = counts[counts > 1].index
    if len(bad_keys) == 0:
        return pd.DataFrame(columns=[left, right])
    return (
        df[df[left].isin(bad_keys)][[left, right]]
        .drop_duplicates()
        .sort_values([left, right])
        .reset_index(drop=True)
    )


def date_coverage_gap(
    sales_dates: Iterable[pd.Timestamp],
    weather_dates: Iterable[pd.Timestamp],
) -> List[pd.Timestamp]:
    """Return sales dates that have no matching external-variable date."""
    sales_set = {pd.Timestamp(date).normalize() for date in sales_dates}
    weather_set = {pd.Timestamp(date).normalize() for date in weather_dates}
    return sorted(sales_set - weather_set)
