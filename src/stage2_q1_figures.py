from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


LEVEL_LABELS = {
    "store": "门店",
    "product": "商品",
    "store_product": "门店--商品",
}

WEEKDAY_LABELS = {
    1: "周一",
    2: "周二",
    3: "周三",
    4: "周四",
    5: "周五",
    6: "周六",
    7: "周日",
}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def read_table(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=parse_dates)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_store_total_sales() -> None:
    df = read_table("q1_store_sales_stats.csv").sort_values("total_sales", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = df["store_name"].astype(str)
    bars = ax.bar(labels, df["total_sales"], color="#4C78A8")
    ax.set_xlabel("门店")
    ax.set_ylabel("历史累计销量")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    save_figure(fig, "q1_store_total_sales.png")


def plot_product_total_sales() -> None:
    df = read_table("q1_product_sales_stats.csv").sort_values("total_sales", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    labels = df["product_name"].astype(str)
    bars = ax.barh(labels, df["total_sales"], color="#59A14F")
    ax.set_xlabel("历史累计销量")
    ax.set_ylabel("商品")
    ax.grid(axis="x", alpha=0.25)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f"{width:.0f}", va="center", fontsize=8)
    save_figure(fig, "q1_product_total_sales.png")


def plot_daily_total_trend() -> None:
    df = read_table("q1_daily_total_trend.csv", parse_dates=["date"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(df["date"], df["target_sales"], color="#4C78A8", linewidth=1.0, alpha=0.55, label="日销量")
    ax.plot(df["date"], df["rolling_7d"], color="#E15759", linewidth=1.8, label="7日滚动均值")
    ax.set_xlabel("日期")
    ax.set_ylabel("销量")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "q1_daily_total_trend.png")


def plot_weekday_effect() -> None:
    df = read_table("q1_weekday_sales_stats.csv").sort_values("weekday")
    labels = [WEEKDAY_LABELS.get(int(day), str(day)) for day in df["weekday"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, df["avg_daily_sales"], color="#F28E2B")
    ax.set_xlabel("星期")
    ax.set_ylabel("平均日销量")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars, fmt="{:.1f}")
    save_figure(fig, "q1_weekday_effect.png")


def plot_model_wape_comparison() -> None:
    df = read_table("q1_model_metrics.csv")
    level_order = ["store", "product", "store_product"]
    model_order = ["moving_average_7", "same_weekday_mean_8", "exp_smoothing"]
    pivot = (
        df.pivot_table(index="level", columns="model", values="WAPE_pct", aggfunc="first")
        .reindex(level_order)
        .reindex(columns=model_order)
    )
    model_labels = df.drop_duplicates("model").set_index("model")["model_label"].reindex(model_order)
    model_labels = pd.Series(
        [label if pd.notna(label) else model for model, label in model_labels.items()],
        index=model_order,
    )
    x = np.arange(len(pivot.index))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = ["#4C78A8", "#F28E2B", "#59A14F"]
    for idx, model in enumerate(model_order):
        values = pivot[model].to_numpy(dtype=float)
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width=width, label=model_labels.loc[model], color=colors[idx])
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS.get(level, level) for level in pivot.index])
    ax.set_xlabel("预测层级")
    ax.set_ylabel("WAPE (%)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "q1_model_wape_comparison.png")


def plot_store_product_heatmap() -> None:
    df = read_table("q1_store_product_sales_stats.csv")
    pivot = df.pivot_table(
        index="store_name",
        columns="product_name",
        values="total_sales",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("商品")
    ax.set_ylabel("门店")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("历史累计销量")
    save_figure(fig, "q1_store_product_heatmap.png")


def main() -> None:
    setup_style()
    plot_store_total_sales()
    plot_product_total_sales()
    plot_daily_total_trend()
    plot_weekday_effect()
    plot_model_wape_comparison()
    plot_store_product_heatmap()
    print("Q1 figures regenerated without internal titles.")


if __name__ == "__main__":
    main()
