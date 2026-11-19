"""每日建议快照入口，供 Windows 任务计划（或 cron）在收盘后调用。

用法：
    C:/Users/AssociatedQuartz/.venvs/ta-gpu/Scripts/python.exe -I \
        "C:/Users/AssociatedQuartz/Desktop/技术分析/quant/run_snapshot.py"

退出码：0=全部成功；1=有标的分析失败（明细见输出 errors）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snapshot  # noqa: E402


def main():
    result = snapshot.run_snapshot()
    try:
        print(json.dumps(result, ensure_ascii=False))
    except UnicodeEncodeError:  # 极少数控制台编码不支持中文时退回转义输出
        print(json.dumps(result, ensure_ascii=True))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
