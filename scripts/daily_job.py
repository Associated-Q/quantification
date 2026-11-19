"""每日收盘后任务：写建议快照 + 模拟盘复盘。由 Windows 计划任务在 15:35 调用。

用法：python scripts/daily_job.py [--ai]
  --ai  同时跑 AI 研判（会调用模型，耗时数分钟并有费用，默认关闭）

日志写到 logs/daily-YYYYMMDD.log；任一步失败退出码非 0，计划任务里能看到。
"""
import argparse
import os
import sys
import time
import traceback
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "quant"))
LOG_DIR = os.path.join(ROOT, "logs")


def log(fh, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="store_true", help="同时生成 AI 研判（调用模型）")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"daily-{date.today():%Y%m%d}.log")
    failed = []

    with open(log_path, "a", encoding="utf-8") as fh:
        log(fh, f"开始每日任务（--ai={args.ai}）")

        if date.today().weekday() >= 5:
            log(fh, "今天是周末，跳过")
            return 0

        import paper
        import snapshot

        try:
            r = snapshot.run_snapshot()
            log(fh, f"建议快照：{r.get('date')} 共 {r.get('count')} 条，错误 {len(r.get('errors') or [])}")
        except Exception as e:
            failed.append("snapshot")
            log(fh, f"建议快照失败：{type(e).__name__}: {e}")
            fh.write(traceback.format_exc() + "\n")

        try:
            r = paper.rebalance()
            s = r.get("summary") or {}
            log(fh, f"模拟盘复盘：买 {s.get('buy_count')} 卖 {s.get('sell_count')} "
                    f"跳过 {s.get('skipped_count')} 持仓 {s.get('position_count')} "
                    f"权益 {s.get('equity')}")
            for o in r.get("orders") or []:
                log(fh, f"  {o['side']} {o['symbol']} {o['shares']}股 @{o['price']} 费用{o['fee']}")
        except Exception as e:
            failed.append("paper")
            log(fh, f"模拟盘复盘失败：{type(e).__name__}: {e}")
            fh.write(traceback.format_exc() + "\n")

        if args.ai:
            try:
                import ai_analyst
                r = ai_analyst.run_daily(refresh=True)
                log(fh, f"AI 研判：{r.get('date')} 生成 {r.get('count')} 条，错误 {len(r.get('errors') or [])}")
            except Exception as e:
                failed.append("ai")
                log(fh, f"AI 研判失败：{type(e).__name__}: {e}")
                fh.write(traceback.format_exc() + "\n")

        log(fh, "结束" + (f"，失败环节：{failed}" if failed else "，全部成功"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
