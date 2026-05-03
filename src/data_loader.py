"""
Data loading helpers for the snack sales forecasting project.

This module only reads files from data/raw. It never modifies the original
attachments. Processed outputs should be written through save_processed().
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}
SALES_KEYWORDS = ("附件一", "历史零售")
WEATHER_KEYWORDS = ("附件二", "天气")


def find_raw_file(
    keywords: Iterable[str],
    raw_dir: Path = RAW_DIR,
    extensions: Iterable[str] = EXCEL_EXTENSIONS,
) -> Path:
    """Find a raw attachment by filename keywords.

    Args:
        keywords: Keywords that should appear in the filename.
        raw_dir: Directory containing original attachments.
        extensions: Allowed file extensions.

    Returns:
        Path to the matched file.

    Raises:
        FileNotFoundError: If no file matches.
        ValueError: If more than one file matches.
    """
    normalized_exts = {ext.lower() for ext in extensions}
    keywords = tuple(keywords)
    matches = [
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_exts
        and all(keyword in path.name for keyword in keywords)
    ]
    if not matches:
        raise FileNotFoundError(f"No raw file matched keywords: {keywords}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple raw files matched {keywords}: {names}")
    return matches[0]


def read_excel_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """Read all sheets from an Excel workbook.

    The function lets pandas choose the engine from the file extension. If an
    old .xls file cannot be opened because xlrd is unavailable, the caller gets
    a clear ImportError instead of a silent partial read.
    """
    path = Path(path)
    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError(
            f"Unable to read {path.name}. Install the required Excel reader "
            "or convert the file to .xlsx without changing the raw content."
        ) from exc

    return {
        sheet_name: pd.read_excel(workbook, sheet_name=sheet_name)
        for sheet_name in workbook.sheet_names
    }


def first_non_empty_sheet(sheets: Dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Return the first worksheet that contains rows or columns."""
    for sheet_name, frame in sheets.items():
        if frame.shape[0] > 0 or frame.shape[1] > 0:
            return sheet_name, frame
    raise ValueError("Workbook contains no non-empty worksheets.")


def load_raw_sales(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the historical retail detail attachment."""
    source = Path(path) if path else find_raw_file(SALES_KEYWORDS)
    sheets = read_excel_workbook(source)
    _, frame = first_non_empty_sheet(sheets)
    return frame


def load_raw_weather(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the weather, holiday, and activity-day attachment."""
    source = Path(path) if path else find_raw_file(WEATHER_KEYWORDS)
    sheets = read_excel_workbook(source)
    _, frame = first_non_empty_sheet(sheets)
    return frame


def load_raw_workbooks() -> dict[str, dict[str, pd.DataFrame]]:
    """Load both raw workbooks with all worksheets."""
    sales_path = find_raw_file(SALES_KEYWORDS)
    weather_path = find_raw_file(WEATHER_KEYWORDS)
    return {
        "sales": read_excel_workbook(sales_path),
        "weather": read_excel_workbook(weather_path),
    }


def load_processed(filename: str, **kwargs) -> pd.DataFrame:
    """Load a processed CSV file from data/processed."""
    filepath = PROCESSED_DIR / filename
    return pd.read_csv(filepath, **kwargs)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a processed DataFrame as UTF-8 CSV.

    Args:
        df: DataFrame to save.
        filename: Output filename under data/processed.

    Returns:
        Path to the saved file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DIR / filename
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath
