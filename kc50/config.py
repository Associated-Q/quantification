"""固定参数。改这里的任何一项都等于改模型定义，必须先与用户确认。

江恩周期的起点由用户指定为 2026-09-01，所有 9/18/27/36/45/60 日窗口与
ATR 标准化角度线都以该日为第 0 天计算，不再从每个转折点重新投影。

其余固定参数的位置：
  kc50/structs.py  CYCLES / RETRACE / EXTEND / TIME_RATIOS
  kc50/core.py     atr_n=14, ma_n=20, 分形 k=2, ATR过滤 1.0
  kc50/model.py    GAP=5（标签隔离）, WARMUP=60, ANCHORS=5
"""
from __future__ import annotations

import pandas as pd

# 江恩周期起点（用户 2026-11-11 指定）
GANN_ANCHOR = pd.Timestamp("2026-09-01")

# 回测口径
OOS_START = "2025-01-01"
TRAIN_START = "2024-01-01"
