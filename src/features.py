"""
Feature engineering helpers for the snack-sales forecasting project.

The early helper functions are kept for notebook compatibility. Stage 5 adds
store-product panel construction and leakage-safe lag/rolling features.
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """添加时间相关特征。

    Args:
        df: DataFrame（必须包含日期列）
        date_col: 日期列名

    Returns:
        添加了时间特征的 DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["dayofweek"] = df[date_col].dt.dayofweek       # 0=周一, 6=周日
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["quarter"] = df[date_col].dt.quarter
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = None,
    lags: list = None,
) -> pd.DataFrame:
    """添加滞后特征。

    Args:
        df: DataFrame（必须按时间排序）
        target_col: 目标列名（如 'sales'）
        group_col: 分组列名（如 'product_id'），None 表示不分组
        lags: 滞后天数列表，默认 [1, 7, 14]

    Returns:
        添加了滞后特征的 DataFrame
    """
    if lags is None:
        lags = [1, 7, 14]

    df = df.copy()

    for lag in lags:
        col_name = f"lag_{lag}"
        if group_col:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = None,
    windows: list = None,
) -> pd.DataFrame:
    """添加滚动统计特征。

    Args:
        df: DataFrame（必须按时间排序）
        target_col: 目标列名
        group_col: 分组列名
        windows: 窗口大小列表，默认 [7, 14]

    Returns:
        添加了滚动特征的 DataFrame
    """
    if windows is None:
        windows = [7, 14]

    df = df.copy()

    for window in windows:
        if group_col:
            df[f"rolling_mean_{window}"] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        else:
            df[f"rolling_mean_{window}"] = df[target_col].rolling(window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df[target_col].rolling(window, min_periods=1).std()

    return df


def combine_unique_names(values: pd.Series) -> str:
    """Return a stable display name for possibly duplicated product names."""
    names = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    return " / ".join(names)


def build_store_product_panel(
    df: pd.DataFrame,
    target_col: str = "positive_sales",
) -> pd.DataFrame:
    """Aggregate the modeling table to date-store-product_id grain.

    Product code is used as the canonical product key. This avoids treating the
    two names under product code 11001020 as two different products.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    product_names = data.groupby("product_id")["product_name"].apply(combine_unique_names)
    meta_cols = {
        "store_name": "first",
        "category": lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "待确认",
        "category_code": lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan,
    }
    panel = (
        data.groupby(["date", "store_id", "product_id"], as_index=False)
        .agg({target_col: "sum", **meta_cols})
        .sort_values(["store_id", "product_id", "date"])
    )
    panel["product_name"] = panel["product_id"].map(product_names)
    panel["weekday"] = panel["date"].dt.isocalendar().day.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["is_weekend"] = panel["weekday"].isin([6, 7]).astype(int)

    # Merge one row per date of external variables.
    ext_cols = [
        "date",
        "weather",
        "max_temperature",
        "min_temperature",
        "wind_power",
        "is_holiday",
        "is_activity_day",
        "has_external_data",
    ]
    existing_ext_cols = [col for col in ext_cols if col in data.columns]
    external = data[existing_ext_cols].drop_duplicates("date")
    panel = panel.merge(external, on="date", how="left", suffixes=("", "_external"))
    for col in ["is_holiday", "is_activity_day", "has_external_data"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)
    return panel.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def add_leakage_safe_sales_features(
    panel: pd.DataFrame,
    target_col: str = "positive_sales",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add lag and rolling features shifted by one day to avoid leakage."""
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    out = panel.copy().sort_values([*group_cols, "date"])
    group = out.groupby(group_cols, dropna=False)[target_col]
    for lag in [1, 7, 14]:
        out[f"lag_{lag}"] = group.shift(lag)
    shifted = group.shift(1)
    out["rolling_mean_7"] = shifted.groupby([out[col] for col in group_cols]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["rolling_mean_14"] = shifted.groupby([out[col] for col in group_cols]).transform(
        lambda s: s.rolling(14, min_periods=1).mean()
    )
    out["rolling_std_7"] = shifted.groupby([out[col] for col in group_cols]).transform(
        lambda s: s.rolling(7, min_periods=2).std()
    )
    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
    ]
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out.reset_index(drop=True)


def build_future_external_scenario(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a documented future external-variable scenario.

    Future weather and promotion variables are not observed in the attachment.
    Weather, temperature, wind and activity day are therefore approximated from
    historical same month-day records. Qingming holiday dates are manually
    encoded from the contest context confirmed by the user.
    """
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month_day"] = data["date"].dt.strftime("%m-%d")
    rows = []
    for date in pd.to_datetime(future_dates):
        month_day = date.strftime("%m-%d")
        same_day = data[(data["month_day"] == month_day) & (data["has_external_data"] == 1)]
        if same_day.empty:
            same_day = data[data["has_external_data"] == 1]
        weather = (
            same_day["weather"].dropna().mode().iloc[0]
            if not same_day["weather"].dropna().empty
            else "未知"
        )
        is_holiday = 1 if date in pd.to_datetime(["2022-04-03", "2022-04-04", "2022-04-05"]) else 0
        rows.append(
            {
                "date": date,
                "weather": weather,
                "max_temperature": float(same_day["max_temperature"].mean()),
                "min_temperature": float(same_day["min_temperature"].mean()),
                "wind_power": float(same_day["wind_power"].mean()),
                "is_holiday": is_holiday,
                "is_activity_day": int(round(float(same_day["is_activity_day"].mean()))),
                "weekday": int(date.isocalendar().weekday),
                "month": int(date.month),
                "is_weekend": int(date.isocalendar().weekday in [6, 7]),
                "external_scenario_note": "天气、温度、风力、活动日为历史同月日参考；节假日按2022清明日历设定",
            }
        )
    return pd.DataFrame(rows)
