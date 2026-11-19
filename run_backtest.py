"""样本外回测：模型 vs 随机 / 恒涨 / 恒跌 / MA20趋势。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kc50.core import add_indicators, find_pivots, load            # noqa: E402
from kc50.model import (baseline_ma20, build_features, evaluate,  # noqa: E402
                        random_baseline, walk_forward)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kc50", "data")


def fmt_ci(ci):
    return f"[{ci[0]:.2%}, {ci[1]:.2%}]"


def main():
    df, rep = load("2023-01-01")
    df = add_indicators(df)
    piv = find_pivots(df)
    feat = build_features(df, piv)
    feat.to_csv(os.path.join(OUT, "features.csv"), index=False)

    last = df["date"].max()
    oos_start = "2025-01-01"
    results = {}
    print(f"\n样本外区间 {oos_start} ~ {last:%Y-%m-%d}｜初始训练 2024 年｜标签隔离 {5} 日\n")

    for h in (3, 5):
        pred = walk_forward(feat, horizon=h, oos_start=oos_start)
        pred.to_csv(os.path.join(OUT, f"pred_{h}.csv"), index=False)
        y = pred["y"].to_numpy(bool)
        base_p = float(np.mean(y))
        ev = evaluate(pred, df["close"].to_numpy(float), base_p, h)
        rnd = random_baseline(y)
        ma = baseline_ma20(df, oos_start, last.strftime("%Y-%m-%d"))
        rnd_sig = np.random.default_rng(3).integers(0, 2, len(y))
        rnd_trade = None
        # 随机方向的交易基准：与模型相同的持仓频率
        from kc50.model import trade_stats
        rnd_trade = trade_stats(pred["close"].to_numpy(float), rnd_sig)

        # MA20 方向准确率（与模型同一批日期）
        m = df[["date", "close", "ma20"]].copy()
        m["ma_sig"] = (m["close"] > m["ma20"]).astype(float)
        j = pred.merge(m[["date", "ma_sig"]], on="date", how="left")
        ma_acc = float(np.mean((j["ma_sig"] > 0.5).to_numpy() == y))
        ma_long = float(np.mean(y[j["ma_sig"].to_numpy() > 0.5]))
        ma_short = float(np.mean(~y[j["ma_sig"].to_numpy() <= 0.5]))
        ma_bacc = float(np.nanmean([ma_long, ma_short]))
        from kc50.model import bootstrap_ci
        ma_ci = bootstrap_ci(((j["ma_sig"].to_numpy() > 0.5) == y).astype(float))

        results[h] = {
            "n": ev["n"], "base_rate": base_p,
            "acc": ev["acc"], "acc_ci": ev["acc_ci"],
            "long_acc": ev["long_acc"], "short_acc": ev["short_acc"],
            "balanced": ev["balanced"], "balanced_ci": ev["balanced_ci"],
            "coverage": ev["coverage"],
            "brier": ev["brier"], "base_brier": ev["base_brier"],
            "random": rnd, "ma20": ma["trade"],
            "ma20_acc": ma_acc, "ma20_bacc": ma_bacc, "ma20_ci": ma_ci,
            "trade": ev["trade"], "random_trade": rnd_trade,
        }

        print(f"===== 未来 {h} 日 =====")
        print(f"样本数 {ev['n']}｜基率(上涨占比) {base_p:.2%}")
        print(f"模型准确率 {ev['acc']:.2%}  CI {fmt_ci(ev['acc_ci'])}")
        print(f"  多头准确率 {ev['long_acc']:.2%}｜空头准确率 {ev['short_acc']:.2%}"
              f"｜平衡准确率 {ev['balanced']:.2%}  CI {fmt_ci(ev['balanced_ci'])}")
        print(f"  信号覆盖率(±5pp) {ev['coverage']:.2%}")
        print(f"  Brier {ev['brier']:.4f}｜基率Brier {ev['base_brier']:.4f}"
              f"｜恒涨Brier {ev['always_up_brier']:.4f}｜恒跌Brier {ev['always_down_brier']:.4f}")
        print(f"基准｜随机 {rnd['acc']:.2%} CI {fmt_ci(rnd['ci'])} ({rnd['n_sim']}次模拟)"
              f"｜恒涨 {ev['always_up_acc']:.2%}｜恒跌 {ev['always_down_acc']:.2%}"
              f"｜MA20 {ma_acc:.2%} CI {fmt_ci(ma_ci)}（平衡 {ma_bacc:.2%}）")
        print(f"交易｜模型 累计 {ev['trade']['cum']:.2%} 年化 {ev['trade']['ann']:.2%} "
              f"日均 {ev['trade']['avg_daily']:.4%} 最大回撤 {ev['trade']['mdd']:.2%} "
              f"持仓占比 {ev['trade']['exposure']:.0%}")
        print(f"     ｜MA20 累计 {ma['trade']['cum']:.2%} 年化 {ma['trade']['ann']:.2%} "
              f"最大回撤 {ma['trade']['mdd']:.2%}")
        print(f"     ｜随机 累计 {rnd_trade['cum']:.2%} 最大回撤 {rnd_trade['mdd']:.2%}")
        print()

    path = os.path.join(OUT, "backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
