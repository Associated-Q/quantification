"""一键启动：校验数据 → 必要时重建回测 → 输出今日判断。

用法：
    python run.py             # 数据有更新才重建回测
    python run.py --rebuild   # 强制重建回测
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

BT = os.path.join(BASE, "kc50", "data", "backtest.json")
DATA = os.path.join(BASE, "kc50", "data", "kc50_daily.csv")
OUT = os.path.join(BASE, "kc50", "data", "daily_log", "latest.txt")


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def need_backtest(force: bool) -> bool:
    if force or not os.path.exists(BT):
        return True
    return os.path.getmtime(BT) < os.path.getmtime(DATA)


def run(script: str) -> None:
    r = subprocess.run([sys.executable, "-I", os.path.join(BASE, script)],
                       capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"{script} 失败，退出码 {r.returncode}")


def main() -> None:
    _utf8()
    force = "--rebuild" in sys.argv

    if need_backtest(force):
        print("[1/2] 重建样本外回测（约 1~2 分钟）…")
        run("run_backtest.py")
    else:
        print("[1/2] 复用已有回测结果（--rebuild 可强制重建）")

    print("[2/2] 校验数据并生成今日输出 …\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import run_daily
        run_daily.main()
    text = buf.getvalue()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
