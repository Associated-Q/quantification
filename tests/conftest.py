"""pytest 公共配置：把项目各模块加到 sys.path，并统一 UTF-8 输出。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("quant", "web", os.path.join("_distill", "rag")):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
