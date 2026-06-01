"""
Evaluation metrics for sales forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error.

    WAPE = sum(|y - yhat|) / sum(|y|). It is more stable than MAPE when many
    daily sales values are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def calculate_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and WAPE for one prediction set."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


def metrics_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    actual_col: str = "actual",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """Calculate metrics for each group in a prediction DataFrame."""
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(calculate_metrics(group[actual_col], group[pred_col]))
        row["actual_sum"] = float(group[actual_col].sum())
        row["prediction_sum"] = float(group[pred_col].sum())
        row["n"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(metrics: pd.DataFrame, level_col: str = "level") -> pd.DataFrame:
    """Sort model metrics by level and WAPE."""
    sort_cols = [level_col, "WAPE", "MAE", "RMSE"]
    return metrics.sort_values(sort_cols).reset_index(drop=True)


def paired_error_tests(
    baseline_errors: pd.Series,
    candidate_errors: pd.Series,
    label: str,
) -> dict:
    """Run paired tests on matched absolute-error series.

    The tests evaluate whether the candidate error is lower than the baseline
    error on the same validation units. They do not prove practical or causal
    superiority; they only summarize validation error differences.
    """
    base = pd.Series(baseline_errors, dtype=float).reset_index(drop=True)
    cand = pd.Series(candidate_errors, dtype=float).reset_index(drop=True)
    mask = base.notna() & cand.notna()
    base = base[mask]
    cand = cand[mask]
    diff = base - cand
    row = {
        "comparison": label,
        "n_pairs": int(len(diff)),
        "mean_baseline_error": float(base.mean()) if len(base) else np.nan,
        "mean_candidate_error": float(cand.mean()) if len(cand) else np.nan,
        "mean_error_reduction": float(diff.mean()) if len(diff) else np.nan,
        "t_stat": np.nan,
        "t_p_value_less": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p_value_less": np.nan,
        "dm_stat": np.nan,
        "dm_p_value_less": np.nan,
        "note": "",
    }
    if len(diff) < 3:
        row["note"] = "样本对数不足，未做显著性检验"
        return row
    try:
        t_res = stats.ttest_rel(base, cand, alternative="greater")
        row["t_stat"] = float(t_res.statistic)
        row["t_p_value_less"] = float(t_res.pvalue)
    except Exception as exc:  # pragma: no cover - defensive for scipy variants
        row["note"] += f"配对t检验失败: {exc}; "
    try:
        if np.allclose(diff, 0):
            row["note"] += "误差差值全为0，Wilcoxon不适用; "
        else:
            w_res = stats.wilcoxon(base, cand, alternative="greater", zero_method="wilcox")
            row["wilcoxon_stat"] = float(w_res.statistic)
            row["wilcoxon_p_value_less"] = float(w_res.pvalue)
    except Exception as exc:  # pragma: no cover
        row["note"] += f"Wilcoxon检验失败: {exc}; "
    # Diebold-Mariano style test on loss differential, using h=1 and no
    # autocovariance correction. This is a simple diagnostic for daily losses.
    dm_denom = diff.std(ddof=1) / np.sqrt(len(diff))
    if dm_denom > 0:
        dm_stat = diff.mean() / dm_denom
        row["dm_stat"] = float(dm_stat)
        row["dm_p_value_less"] = float(1.0 - stats.t.cdf(dm_stat, df=len(diff) - 1))
    else:
        row["note"] += "DM检验差值方差为0，未计算; "
    return row
