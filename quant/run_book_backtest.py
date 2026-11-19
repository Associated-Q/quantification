"""按书里的规则做样本内外回测，与买入持有对比。

阈值在跑之前就定好（入场 ≥ +2，离场 ≤ -1），不根据结果调整。
"""
import os
import sys

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book_strategies as bs
import data_source
import engine
import strategies

# 固定标的池：结果只跟策略参数有关，不随数据库里多抓了哪只票而漂移。
# 修改此列表会让历史回测数字不可直接比较。
UNIVERSE = [
    "sh510050", "sh510300", "sh510500", "sh512170", "sh512480", "sh512880",
    "sh513050", "sh515050", "sh518880", "sh588000", "sh600519",
    "sz159915", "sz159949",
]

IS = ("2023-07-12", "2025-06-30")
OOS = ("2025-07-01", "2026-10-30")


def load():
    con = duckdb.connect(data_source.DB, read_only=True)
    df = con.execute("SELECT * FROM kline_daily ORDER BY symbol, trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def score_for(g, variant):
    sc = bs.book_score(g)
    if variant in ("wave", "mtf"):
        sc = sc + pd.Series(bs.weekly_wave_daily(g), index=g.index)
    if variant == "mtf":
        sc = sc + pd.Series(bs.tf_trend_daily(g), index=g.index)
    return sc


def main():
    all_df = load()
    # 只保留样本内、样本外都有足够 bar 的标的，否则两个区间不可比
    have = set(all_df.symbol)
    eligible = [s for s in UNIVERSE if s in have]
    missing = [s for s in UNIVERSE if s not in have]
    if missing:
        print("警告：标的不在库中，已跳过:", missing)
    print(f"标的池（固定 {len(UNIVERSE)} 只，本次可用 {len(eligible)} 只）")
    rows = []
    for sym in eligible:
        g = all_df[all_df.symbol == sym].reset_index(drop=True)
        for period, (a, b) in (("IS", IS), ("OOS", OOS)):
            s = g[(g.trade_date >= a) & (g.trade_date <= b)].reset_index(drop=True)
            if len(s) < 90:
                continue
            atr = strategies.atr(s)
            for variant in ("base", "wave", "mtf"):
                sc = score_for(s, variant)
                entry, exit_ = sc >= 2.0, sc <= -1.0
                for mode, kw in (("fixed", {"size_pct": 0.5, "atr_stop": 2.0}),
                                 ("risk", {"risk_pct": 0.005, "atr_stop": 2.0})):
                    eq, tr = engine.run(s, entry, exit_, atr, **kw)
                    m = engine.metrics(eq, tr)
                    rows.append({"symbol": sym, "period": period, "variant": variant,
                                 "mode": mode, **m})

            e2, x2 = strategies.buy_hold(s)
            eq, tr = engine.run(s, e2, x2, atr, atr_stop=None, size_pct=1.0)
            rows.append({"symbol": sym, "period": period, "variant": "buy_hold",
                         "mode": "buy_hold", **engine.metrics(eq, tr)})

    res = pd.DataFrame(rows)
    res.to_csv("book_backtest.csv", index=False)

    print("=== 各变体汇总（固定 50% 仓位，标的等权平均）===")
    fixed = res[res["mode"] == "fixed"]
    agg = fixed.groupby(["variant", "period"])[["annual", "max_drawdown", "sharpe",
                                                "trades", "win_rate", "profit_factor"]].mean().round(3)
    print(agg.to_string())

    print("\n=== 风险仓位模式 ===")
    risk = res[res["mode"] == "risk"]
    print(risk.groupby(["variant", "period"])[["annual", "max_drawdown", "profit_factor"]]
          .mean().round(3).to_string())

    print("\n=== 各标的年化（mtf 变体，固定 50% 仓位）===")
    p = res[(res["mode"] == "fixed") & (res["variant"] == "mtf")].pivot(
        index="symbol", columns="period", values="annual").round(3)
    print(p.to_string())

    n = res[res["mode"] == "fixed"]["symbol"].nunique()
    print(f"\n标的数 {n} | 区间 IS {IS[0]}~{IS[1]} / OOS {OOS[0]}~{OOS[1]}")


if __name__ == "__main__":
    main()
