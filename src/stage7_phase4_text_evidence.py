from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import standardize_weather

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"

TARGET = "positive_sales"


def save(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUTS / name, index=False, encoding="utf-8-sig")


def product_lookup(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product_id, group in base.groupby("product_id", dropna=False):
        names = sorted(str(x) for x in group["product_name"].dropna().unique())
        categories = sorted(str(x) for x in group["category"].dropna().unique())
        rows.append(
            {
                "product_id": product_id,
                "product_name": " / ".join(names),
                "category": " / ".join(categories),
            }
        )
    return pd.DataFrame(rows)


def trend_slope(values: pd.Series) -> float:
    y = values.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def panel_descriptive(base: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    daily = (
        base.groupby(["date", *id_cols], dropna=False)[TARGET]
        .sum()
        .reset_index()
        .sort_values([*id_cols, "date"])
    )
    rows = []
    for keys, group in daily.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[TARGET].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        row = dict(zip(id_cols, keys))
        row.update(
            {
                "total_sales": float(values.sum()),
                "mean_daily_sales": mean,
                "std_daily_sales": std,
                "cv": float(std / mean) if mean > 0 else np.nan,
                "zero_sales_days": int((values == 0).sum()),
                "n_days": int(len(values)),
                "zero_sales_ratio": float((values == 0).mean()),
                "trend_slope_per_day": trend_slope(values),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_q1_descriptive(base: pd.DataFrame) -> None:
    store = panel_descriptive(base, ["store_id"])
    store_names = base[["store_id", "store_name"]].drop_duplicates("store_id")
    store = store.merge(store_names, on="store_id", how="left")
    store = store[
        [
            "store_id",
            "store_name",
            "total_sales",
            "mean_daily_sales",
            "cv",
            "zero_sales_ratio",
            "trend_slope_per_day",
        ]
    ].sort_values("total_sales", ascending=False)
    save(store, "q1_store_descriptive_phase4.csv")

    product = panel_descriptive(base, ["product_id"])
    product = product.merge(product_lookup(base), on="product_id", how="left")
    product = product[
        [
            "product_id",
            "product_name",
            "category",
            "total_sales",
            "mean_daily_sales",
            "cv",
            "zero_sales_ratio",
            "trend_slope_per_day",
        ]
    ].sort_values("total_sales", ascending=False)
    save(product, "q1_product_descriptive_phase4.csv")


def build_q2_pair_evidence(base: pd.DataFrame) -> None:
    lookup = product_lookup(base).set_index("product_id")
    daily_product = (
        base.groupby(["date", "product_id"], dropna=False)[TARGET]
        .sum()
        .reset_index()
        .sort_values(["date", "product_id"])
    )
    matrix = daily_product.pivot(index="date", columns="product_id", values=TARGET).fillna(0.0)
    corr = matrix.corr()
    rows = []
    for a, b in combinations(corr.columns, 2):
        a_series = matrix[a]
        b_series = matrix[b]
        rows.append(
            {
                "product_a_id": a,
                "product_a_name": lookup.loc[a, "product_name"],
                "product_b_id": b,
                "product_b_name": lookup.loc[b, "product_name"],
                "corr": float(corr.loc[a, b]),
                "nonzero_both_days": int(((a_series > 0) & (b_series > 0)).sum()),
                "product_a_nonzero_days": int((a_series > 0).sum()),
                "product_b_nonzero_days": int((b_series > 0).sum()),
                "n_days": int(len(matrix)),
            }
        )
    pairs = pd.DataFrame(rows).sort_values("corr", ascending=False)
    save(pairs, "q2_product_correlation_pairs_full_phase4.csv")
    save(pairs[pairs["corr"] >= 0.50].copy(), "q2_strong_positive_pairs_full_phase4.csv")
    weak = pairs[pairs["corr"].abs() <= 0.10].copy().sort_values("corr")
    save(weak, "q2_weak_correlation_pairs_full_phase4.csv")


def load_raw_weather() -> pd.DataFrame:
    weather_files = list(RAW.glob("*天气*.xlsx"))
    if not weather_files:
        raise FileNotFoundError("Cannot find raw weather attachment under data/raw")
    raw_weather = pd.read_excel(weather_files[0])
    return standardize_weather(raw_weather)


def build_q3_overlap(base: pd.DataFrame) -> None:
    daily = (
        base[base["has_external_data"] == 1]
        .drop_duplicates("date")
        .sort_values("date")
        .copy()
    )
    for col in ["is_holiday", "is_weekend", "is_rest_day", "is_activity_day"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0).astype(int)
    total = int(len(daily))
    holiday = int(daily["is_holiday"].sum())
    weekend = int(daily["is_weekend"].sum())
    rest = int(daily["is_rest_day"].sum())
    activity = int(daily["is_activity_day"].sum())
    overlap_rows = [
        {
            "metric": "建模销售期有外部变量日期",
            "n_days": total,
            "denominator": total,
            "share": 1.0,
        },
        {
            "metric": "节假日",
            "n_days": holiday,
            "denominator": total,
            "share": holiday / total,
        },
        {
            "metric": "周末",
            "n_days": weekend,
            "denominator": total,
            "share": weekend / total,
        },
        {
            "metric": "工休日",
            "n_days": rest,
            "denominator": total,
            "share": rest / total,
        },
        {
            "metric": "活动日",
            "n_days": activity,
            "denominator": total,
            "share": activity / total,
        },
        {
            "metric": "节假日且周末",
            "n_days": int(((daily["is_holiday"] == 1) & (daily["is_weekend"] == 1)).sum()),
            "denominator": holiday,
            "share": int(((daily["is_holiday"] == 1) & (daily["is_weekend"] == 1)).sum()) / holiday,
        },
        {
            "metric": "节假日且工休日",
            "n_days": int(((daily["is_holiday"] == 1) & (daily["is_rest_day"] == 1)).sum()),
            "denominator": holiday,
            "share": int(((daily["is_holiday"] == 1) & (daily["is_rest_day"] == 1)).sum()) / holiday,
        },
        {
            "metric": "周末且工休日",
            "n_days": int(((daily["is_weekend"] == 1) & (daily["is_rest_day"] == 1)).sum()),
            "denominator": weekend,
            "share": int(((daily["is_weekend"] == 1) & (daily["is_rest_day"] == 1)).sum()) / weekend,
        },
        {
            "metric": "活动日且周末",
            "n_days": int(((daily["is_activity_day"] == 1) & (daily["is_weekend"] == 1)).sum()),
            "denominator": activity,
            "share": int(((daily["is_activity_day"] == 1) & (daily["is_weekend"] == 1)).sum()) / activity,
        },
        {
            "metric": "活动日且节假日",
            "n_days": int(((daily["is_activity_day"] == 1) & (daily["is_holiday"] == 1)).sum()),
            "denominator": activity,
            "share": int(((daily["is_activity_day"] == 1) & (daily["is_holiday"] == 1)).sum()) / activity,
        },
    ]
    save(pd.DataFrame(overlap_rows), "q3_calendar_overlap_phase4.csv")

    combo = (
        daily.groupby(["is_holiday", "is_weekend", "is_rest_day"], dropna=False)
        .size()
        .reset_index(name="n_days")
        .sort_values(["is_holiday", "is_weekend", "is_rest_day"])
    )
    combo["share"] = combo["n_days"] / total
    save(combo, "q3_calendar_combination_phase4.csv")

    raw_weather = load_raw_weather()
    raw_weather["is_activity_day"] = pd.to_numeric(
        raw_weather["is_activity_day"], errors="coerce"
    ).fillna(0).astype(int)
    raw_summary = pd.DataFrame(
        [
            {
                "scope": "附件二全日期",
                "date_min": raw_weather["date"].min().date().isoformat(),
                "date_max": raw_weather["date"].max().date().isoformat(),
                "n_days": int(len(raw_weather)),
                "activity_days": int(raw_weather["is_activity_day"].sum()),
            },
            {
                "scope": "建模销售期且有外部变量日期",
                "date_min": daily["date"].min().date().isoformat(),
                "date_max": daily["date"].max().date().isoformat(),
                "n_days": total,
                "activity_days": activity,
            },
        ]
    )
    save(raw_summary, "q3_activity_day_scope_phase4.csv")


def main() -> None:
    base = pd.read_csv(PROCESSED / "modeling_base_table.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    build_q1_descriptive(base)
    build_q2_pair_evidence(base)
    build_q3_overlap(base)
    print("Phase 4 evidence tables generated.")


if __name__ == "__main__":
    main()
