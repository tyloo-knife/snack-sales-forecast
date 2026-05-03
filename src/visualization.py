"""
Visualization helpers based on Pillow.

The bundled Python environment used for this project does not include
matplotlib, so these helpers draw simple PNG charts directly with Pillow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import FIGURE_DIR


WIDTH = 1400
HEIGHT = 820
MARGIN_LEFT = 150
MARGIN_RIGHT = 80
MARGIN_TOP = 90
MARGIN_BOTTOM = 180


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _canvas(width: int = WIDTH, height: int = HEIGHT) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def _draw_title(draw: ImageDraw.ImageDraw, title: str, width: int = WIDTH) -> None:
    font = _font(32, bold=True)
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 24), title, fill="#1F2937", font=font)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
    y_max: float,
    ylabel: str,
) -> None:
    axis_color = "#374151"
    grid_color = "#E5E7EB"
    label_font = _font(20)
    small_font = _font(16)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
    for i in range(6):
        y = plot_bottom - i * (plot_bottom - plot_top) / 5
        value = y_max * i / 5
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        draw.text((18, y - 10), f"{value:,.0f}", fill="#4B5563", font=small_font)
    draw.text((plot_left, 60), ylabel, fill="#111827", font=label_font)


def _scale(value: float, y_max: float, plot_top: int, plot_bottom: int) -> float:
    if y_max <= 0:
        return plot_bottom
    return plot_bottom - (float(value) / y_max) * (plot_bottom - plot_top)


def _save(image: Image.Image, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    image.save(path)
    return path


def save_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rotate: int = 30,
    color: str = "#4C78A8",
) -> Path:
    """Save a simple vertical bar chart."""
    data = df[[x, y]].copy()
    labels = data[x].astype(str).tolist()
    values = data[y].astype(float).to_numpy()
    width = max(WIDTH, 95 * len(labels) + MARGIN_LEFT + MARGIN_RIGHT)
    image, draw = _canvas(width=width)
    _draw_title(draw, title, width=width)
    plot_left, plot_top = MARGIN_LEFT, MARGIN_TOP
    plot_right, plot_bottom = width - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM
    y_max = max(float(np.nanmax(values)) * 1.12, 1.0)
    _draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, y_max, ylabel)

    gap = 12
    bar_width = max(18, ((plot_right - plot_left) - gap * (len(values) + 1)) / max(len(values), 1))
    label_font = _font(16)
    value_font = _font(15)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = plot_left + gap + idx * (bar_width + gap)
        x1 = x0 + bar_width
        y0 = _scale(value, y_max, plot_top, plot_bottom)
        draw.rectangle((x0, y0, x1, plot_bottom), fill=color)
        draw.text((x0, y0 - 24), f"{value:,.0f}", fill="#111827", font=value_font)
        short = label if len(label) <= 14 else label[:13] + "…"
        bbox = draw.textbbox((0, 0), short, font=label_font)
        draw.text((x0 + bar_width / 2 - (bbox[2] - bbox[0]) / 2, plot_bottom + 14), short, fill="#111827", font=label_font)
    draw.text((plot_left, HEIGHT - 56), xlabel, fill="#111827", font=_font(18))
    return _save(image, filename)


def save_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rolling_y: str | None = None,
) -> Path:
    """Save a line chart."""
    data = df[[x, y] + ([rolling_y] if rolling_y else [])].copy()
    values = data[y].astype(float).to_numpy()
    roll_values = data[rolling_y].astype(float).to_numpy() if rolling_y else None
    image, draw = _canvas()
    _draw_title(draw, title)
    plot_left, plot_top = MARGIN_LEFT, MARGIN_TOP
    plot_right, plot_bottom = WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM
    y_max = max(float(np.nanmax(values)) * 1.12, 1.0)
    if roll_values is not None:
        y_max = max(y_max, float(np.nanmax(roll_values)) * 1.12)
    _draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, y_max, ylabel)

    n = len(data)
    if n > 1:
        xs = np.linspace(plot_left, plot_right, n)
        raw_points = [(float(px), _scale(v, y_max, plot_top, plot_bottom)) for px, v in zip(xs, values)]
        draw.line(raw_points, fill="#9CA3AF", width=2)
        if roll_values is not None:
            roll_points = [(float(px), _scale(v, y_max, plot_top, plot_bottom)) for px, v in zip(xs, roll_values)]
            draw.line(roll_points, fill="#2563EB", width=4)
            draw.text((plot_left + 20, plot_top + 20), "gray: daily, blue: 7-day rolling", fill="#111827", font=_font(18))
    first_date = pd.to_datetime(data[x].iloc[0]).date()
    last_date = pd.to_datetime(data[x].iloc[-1]).date()
    draw.text((plot_left, plot_bottom + 16), str(first_date), fill="#111827", font=_font(16))
    draw.text((plot_right - 120, plot_bottom + 16), str(last_date), fill="#111827", font=_font(16))
    draw.text((plot_left, HEIGHT - 56), xlabel, fill="#111827", font=_font(18))
    return _save(image, filename)


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Save a basic heatmap."""
    rows, cols = matrix.shape
    cell_w = 105
    cell_h = 58
    width = max(WIDTH, MARGIN_LEFT + MARGIN_RIGHT + cols * cell_w)
    height = max(HEIGHT, MARGIN_TOP + MARGIN_BOTTOM + rows * cell_h)
    image, draw = _canvas(width=width, height=height)
    _draw_title(draw, title, width=width)
    values = matrix.to_numpy(dtype=float)
    max_val = max(float(np.nanmax(values)), 1.0)
    label_font = _font(15)
    small_font = _font(13)
    start_x, start_y = MARGIN_LEFT, MARGIN_TOP + 30
    for i, row_label in enumerate(matrix.index.astype(str)):
        draw.text((20, start_y + i * cell_h + 18), row_label, fill="#111827", font=label_font)
    for j, col_label in enumerate(matrix.columns.astype(str)):
        short = col_label if len(col_label) <= 8 else col_label[:7] + "…"
        draw.text((start_x + j * cell_w + 6, start_y - 30), short, fill="#111827", font=small_font)
    for i in range(rows):
        for j in range(cols):
            value = values[i, j]
            intensity = int(245 - 170 * (value / max_val))
            color = (intensity, min(255, intensity + 35), 255)
            x0 = start_x + j * cell_w
            y0 = start_y + i * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=color)
            draw.text((x0 + 8, y0 + 20), f"{value:,.0f}", fill="#111827", font=small_font)
    draw.text((start_x, height - 60), xlabel, fill="#111827", font=_font(18))
    draw.text((20, start_y - 35), ylabel, fill="#111827", font=_font(18))
    return _save(image, filename)


def save_grouped_bar(
    df: pd.DataFrame,
    category_col: str,
    series_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Save a grouped bar chart."""
    pivot = df.pivot(index=category_col, columns=series_col, values=value_col).fillna(0)
    categories = pivot.index.astype(str).tolist()
    series = pivot.columns.astype(str).tolist()
    values = pivot.to_numpy(dtype=float)
    image, draw = _canvas()
    _draw_title(draw, title)
    plot_left, plot_top = MARGIN_LEFT, MARGIN_TOP
    plot_right, plot_bottom = WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM
    y_max = max(float(np.nanmax(values)) * 1.12, 1.0)
    _draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, y_max, ylabel)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    group_w = (plot_right - plot_left) / max(len(categories), 1)
    bar_w = max(14, group_w / (len(series) + 1.2))
    font = _font(16)
    for i, cat in enumerate(categories):
        base_x = plot_left + i * group_w + 8
        for j, ser in enumerate(series):
            value = values[i, j]
            x0 = base_x + j * bar_w
            x1 = x0 + bar_w * 0.85
            y0 = _scale(value, y_max, plot_top, plot_bottom)
            draw.rectangle((x0, y0, x1, plot_bottom), fill=colors[j % len(colors)])
        draw.text((base_x, plot_bottom + 14), cat, fill="#111827", font=font)
    legend_x = plot_left + 20
    legend_y = plot_top + 18
    for j, ser in enumerate(series):
        draw.rectangle((legend_x, legend_y + j * 28, legend_x + 18, legend_y + 18 + j * 28), fill=colors[j % len(colors)])
        draw.text((legend_x + 26, legend_y + j * 28 - 2), ser, fill="#111827", font=_font(16))
    draw.text((plot_left, HEIGHT - 56), xlabel, fill="#111827", font=_font(18))
    return _save(image, filename)
