from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import exp_smoothing_prediction, prepare_series_table

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

TARGET = "positive_sales"
VALIDATION_START = pd.Timestamp("2022-03-01")
ALPHAS = [0.1, 0.3, 0.5]


LEVEL_CONFIGS = {
    "store": {
        "label": "门店",
        "id_cols": ["store_id"],
    },
    "product": {
        "label": "商品",
        "id_cols": ["product_id"],
    },
    "store_product": {
        "label": "门店--商品",
        "id_cols": ["store_id", "product_id"],
    },
}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual"].astype(float).to_numpy()
    pred = df["prediction"].astype(float).to_numpy()
    err = actual - pred
    actual_sum = float(np.abs(actual).sum())
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "WAPE": float(np.abs(err).sum() / actual_sum) if actual_sum else np.nan,
        "WAPE_pct": float(np.abs(err).sum() / actual_sum * 100.0) if actual_sum else np.nan,
        "actual_sum": float(actual.sum()),
        "prediction_sum": float(pred.sum()),
        "n": int(len(df)),
    }


def validate_fixed_alpha(
    series_df: pd.DataFrame,
    id_cols: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in series_df.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("date").reset_index(drop=True)
        validation = group[group["date"] >= VALIDATION_START]
        for _, row in validation.iterrows():
            target_date = row["date"]
            history = group[group["date"] < target_date][TARGET]
            if history.empty:
                continue
            record = dict(zip(id_cols, keys))
            record.update(
                {
                    "date": target_date,
                    "alpha": alpha,
                    "actual": float(row[TARGET]),
                    "prediction": max(0.0, exp_smoothing_prediction(history, alpha=alpha)),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def build_alpha_sensitivity() -> pd.DataFrame:
    base = pd.read_csv(PROCESSED / "modeling_base_table.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    rows: list[dict[str, object]] = []
    for level, config in LEVEL_CONFIGS.items():
        series = prepare_series_table(base, config["id_cols"], TARGET)
        for alpha in ALPHAS:
            pred = validate_fixed_alpha(series, config["id_cols"], alpha)
            metrics = calculate_metrics(pred)
            rows.append(
                {
                    "level": level,
                    "level_label": config["label"],
                    "validation_mode": "rolling_one_step",
                    "validation_start": VALIDATION_START.date().isoformat(),
                    "model": "exp_smoothing_fixed_alpha",
                    "alpha": alpha,
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "q1_exp_smoothing_alpha_sensitivity.csv", index=False, encoding="utf-8-sig")
    out.to_csv(
        OUTPUTS / "q1_exp_smoothing_alpha_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        2.25,
        0.78,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)
    ax.text(x + 1.125, y + 0.39, text, ha="center", va="center", fontsize=10.5)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.0,
        color="#4B5563",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_technical_route() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    boxes = [
        ((0.25, 2.95), "附件销售明细\n天气与日历变量", "#4C78A8"),
        ((3.05, 2.95), "日销量面板\n正向销量口径", "#4C78A8"),
        ((5.85, 2.95), "Q1 基准预测\n门店/商品/组合", "#59A14F"),
        ((8.35, 2.95), "Q4 综合融合\n严格7日递推验证", "#E15759"),
        ((3.05, 1.15), "Q2 类别聚合\n商品同步波动", "#F28E2B"),
        ((5.85, 1.15), "Q3 外部关联\n天气/节假日/活动日", "#B07AA1"),
        ((8.35, 1.15), "最终条件预测\n统一整数汇总", "#E15759"),
    ]
    for xy, text, color in boxes:
        add_box(ax, xy, text, color)

    add_arrow(ax, (2.50, 3.34), (3.05, 3.34))
    add_arrow(ax, (5.30, 3.34), (5.85, 3.34))
    add_arrow(ax, (8.10, 3.34), (8.35, 3.34))
    add_arrow(ax, (4.18, 2.95), (4.18, 1.93))
    add_arrow(ax, (5.30, 1.54), (5.85, 1.54))
    add_arrow(ax, (8.10, 1.54), (8.35, 1.54))
    add_arrow(ax, (6.98, 2.95), (6.98, 1.93))
    add_arrow(ax, (9.48, 2.95), (9.48, 1.93))

    ax.text(1.38, 0.32, "时间切分；不随机打乱；不使用未来信息", ha="center", fontsize=9.5, color="#374151")
    ax.text(6.75, 0.32, "相关性只作同步波动或条件关联解释，不写成因果", ha="center", fontsize=9.5, color="#374151")
    fig.tight_layout()
    fig.savefig(FIGURES / "technical_route_phase3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    sensitivity = build_alpha_sensitivity()
    draw_technical_route()
    print(
        sensitivity[
            ["level_label", "alpha", "WAPE_pct", "MAE", "RMSE", "n"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
