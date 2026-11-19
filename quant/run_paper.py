"""模拟盘复盘入口，供 Windows 任务计划（或 cron）在收盘后调用。

用法：
    C:/Users/AssociatedQuartz/.venvs/ta-gpu/Scripts/python.exe -I \
        "C:/Users/AssociatedQuartz/Desktop/技术分析/quant/run_paper.py" --period 1

只有距上次复盘 ≥ period 天才执行；未到期时输出 executed=false 并以 0 退出。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="模拟盘定时复盘")
    parser.add_argument("--period", type=int, default=1,
                        help="距上次复盘至少间隔的天数（默认 1）")
    args = parser.parse_args(argv)
    if args.period < 1:
        parser.error("--period 必须为正整数")
    result = paper.run_if_due(args.period)
    try:
        print(json.dumps(result, ensure_ascii=False))
    except UnicodeEncodeError:
        print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
