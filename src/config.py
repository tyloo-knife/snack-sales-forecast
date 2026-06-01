"""
项目配置：路径、参数、常量。
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"

# ============================================================
# Model Parameters
# ============================================================

RANDOM_STATE = 42
TRAIN_RATIO = 0.8  # 时间序列按比例划分

# ============================================================
# Evaluation Metrics
# ============================================================

# 需要计算的指标列表
METRICS = ["MAE", "RMSE", "WAPE"]
