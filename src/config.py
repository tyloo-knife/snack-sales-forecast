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
# Feature Settings
# ============================================================

# 滞后特征的天数
LAG_DAYS = [1, 7, 14]

# 滚动窗口大小
ROLLING_WINDOWS = [7, 14, 30]

# ============================================================
# Evaluation Metrics
# ============================================================

# 需要计算的指标列表
METRICS = ["MAE", "RMSE", "MAPE"]
