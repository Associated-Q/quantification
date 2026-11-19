"""跑一批 ETF 的基线回测，输出每标的每策略的指标与汇总。

用法：python run_baseline.py [--codes 510300,510500] [--days 800]
"""
import argparse
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_source
import engine
import strategies

DEFAULT_CODES = ["510300", "510500", "510050", "159915", "588000", "512880",
                 "512480", "512170", "515050", "518880", "513050", "159949"]


def load(codes, days):
    db = data_source.DB
    have = set()
    if os.path.exists(db):
        con = duckdb.connect(db)
        have = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM kline_daily").fetchall()}
        con.close()
    for c in codes:
        sym = data_source.to_symbol(c)
        if sym not in have:
            data_source.save(data_source.fetch_daily(c, count=days))
    con = duckdb.connect(db)
    q = f"SELECT * FROM kline_daily WHERE symbol IN ({','.join(['?'] * len(codes))}) ORDER BY symbol, trade_date"
    df = con.execute(q, [data_source.to_symbol(c) for c in codes]).df()
    con.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes")
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--risk", type=float, default=0.005)
    args = ap.parse_args()
    codes = args.codes.split(",") if args.codes else DEFAULT_CODES

    all_df = load(codes, args.days)
    rows = []
    for sym, g in all_df.groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        for name, fn in list(strategies.STRATEGIES.items()) + [("buy_hold", strategies.buy_hold)]:
            entry, exit_ = fn(g)
            kw = {"atr_stop": None, "size_pct": 1.0} if name == "buy_hold" else {}
            eq, tr = engine.run(g, entry, exit_, strategies.atr(g),
                                risk_pct=args.risk, **kw)
            m = engine.metrics(eq, tr)
            rows.append({"symbol": sym, "strategy": name, **m})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 按策略汇总（等权平均） ===")
    agg = res.groupby("strategy")[["total_return", "annual", "max_drawdown", "sharpe",
                                   "trades", "win_rate", "profit_factor"]].mean().round(3)
    print(agg.to_string())
    print(f"\n标的数 {res['symbol'].nunique()} | 区间 {all_df['trade_date'].min()} -> {all_df['trade_date'].max()}")


if __name__ == "__main__":
    main()
