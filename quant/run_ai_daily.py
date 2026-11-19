"""计划任务入口：对全部自选执行一次 AI 读图研判。

用法：
    python -I quant/run_ai_daily.py                 # 刷新日线后跑全部自选
    python -I quant/run_ai_daily.py --no-refresh    # 只用本地数据
    python -I quant/run_ai_daily.py --symbols 600519 300759

结果写入 UTF-8 JSON 文件；控制台只输出 ASCII 摘要，避免 Windows GBK 乱码。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ai_analyst as ai  # noqa: E402
import realtime as rt  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="AI 读图研判每日任务")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="只跑指定代码；不传则跑全部分组自选")
    parser.add_argument("--no-refresh", action="store_true",
                        help="不重新拉取日线，直接用本地数据")
    parser.add_argument("--out", default=None, help="报告输出路径（UTF-8 JSON）")
    args = parser.parse_args(argv)

    refresh = not args.no_refresh
    if args.symbols:
        result = ai.run_symbols(args.symbols, refresh=refresh)
    else:
        result = ai.run_daily(refresh=refresh)

    out = args.out or os.path.join(HERE, f"ai_daily_{result['date'].replace('-', '')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print("AI daily date=%s count=%d errors=%d out=%s"
          % (result["date"], result["count"], len(result["errors"]), out))
    for err in result["errors"]:
        print("  ERROR %s: %s" % (err.get("symbol"), err.get("error")))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
