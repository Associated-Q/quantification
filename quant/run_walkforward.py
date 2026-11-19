"""固定敞口 + 样本内外对比（walk-forward）。

不做参数调优：所有策略用同一套默认参数，只比较样本内与样本外的表现差异。
每个切片内部独立计算信号与 ATR，避免用全序列算完再切造成泄漏。
"""
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_source
import engine
import strategies

IS_START, IS_END = "2023-07-12", "2025-06-30"
OOS_START, OOS_END = "2025-07-01", "2026-10-30"
WARMUP = 60
MIN_BARS = 80
MODES = {"risk": {"risk_pct": 0.005, "atr_stop": 2.0},
         "fixed": {"size_pct": 0.5, "atr_stop": 2.0}}
PERIODS = {"IS": (IS_START, IS_END), "OOS": (OOS_START, OOS_END)}


def load_all():
    con = duckdb.connect(data_source.DB, read_only=True)
    df = con.execute("SELECT * FROM kline_daily ORDER BY symbol, trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def main():
    df = load_all()
    rows, skipped = [], []
    for sym, g in df.groupby("symbol", sort=True):
        for pname, (start, end) in PERIODS.items():
            seg = g[(g.trade_date >= start) & (g.trade_date <= end)].reset_index(drop=True)
            tradeable = max(0, len(seg) - WARMUP)
            if len(seg) < MIN_BARS:
                skipped.append({"symbol": sym, "period": pname, "bars": len(seg)})
                continue
            for sname, fn in strategies.STRATEGIES.items():
                entry, exit_ = fn(seg)
                a = strategies.atr(seg)
                for mode, kw in MODES.items():
                    eq, tr = engine.run(seg, entry, exit_, a, **kw)
                    m = engine.metrics(eq, tr)
                    rows.append({"symbol": sym, "strategy": sname, "mode": mode,
                                 "period": pname, "bars": len(seg),
                                 "tradeable_bars": tradeable,
                                 "annual": m["annual"], "max_drawdown": m["max_drawdown"],
                                 "sharpe": m["sharpe"], "trades": m["trades"],
                                 "win_rate": m["win_rate"], "profit_factor": m["profit_factor"]})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "walkforward.csv"),
               index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    cols = ["annual", "max_drawdown", "sharpe", "trades", "win_rate", "profit_factor"]
    print("=== 等权汇总（strategy × mode × period）===")
    agg = res.groupby(["strategy", "mode", "period"])[cols].mean().round(3)
    print(agg.to_string())

    print("\n=== 样本外 vs 样本内（annual 与 profit_factor 并排）===")
    piv = res.pivot_table(index=["strategy", "mode"], columns="period",
                          values=["annual", "profit_factor"], aggfunc="mean").round(3)
    print(piv.to_string())

    print(f"\n标的数 {res['symbol'].nunique()} | 明细行 {len(res)} | 已写 walkforward.csv")
    print("每段 bar 数（按标的）：")
    bars = res.groupby(["period", "symbol"])[["bars", "tradeable_bars"]].first()
    print(bars.to_string())
    if skipped:
        print("\n因数据不足被跳过：", skipped)
    zero = res[res["trades"] == 0][["symbol", "strategy", "mode", "period"]].drop_duplicates()
    if not zero.empty:
        print(f"\n零成交组合共 {len(zero)} 个（多为标的单价过高、一手资金门槛不足）：")
        print(zero.groupby("symbol").size().to_string())

    out = res.sort_values(["symbol", "strategy", "mode", "period"])
    print("\n=== 明细（前 20 行）===")
    print(out.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
