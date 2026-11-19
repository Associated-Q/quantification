"""多周期分析：日线用完整书内规则，周线/月线用趋势与浪型做方向约束。

组合方式（大周期定方向，小周期定时点）：
- 月线趋势（MA12 与 MA24）权重 1.0
- 周线趋势（MA20 与 MA60）权重 0.75
- 周线浪型（若成立）权重 0.75
日线原有的书内规则保持原权重，最终合并成一个评分。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book_rules as br
import data_source
import realtime
import wave as wv

TABLES = {"day": "kline_daily", "week": "kline_weekly", "month": "kline_monthly"}
MIN_BARS = {"day": 70, "week": 70, "month": 30}
FETCH_COUNTS = {"day": 800, "week": 740, "month": 175}


def ensure(sym, period):
    """确保库里有该周期的数据，不够或日线过期就现拉一次（腾讯前复权）。"""
    table = TABLES[period]
    df = realtime.period_history(sym, table)
    last = None if df.empty else str(df["trade_date"].iloc[-1])[:10]
    stale = realtime.is_stale(sym, last, table)
    if len(df) < MIN_BARS[period] or stale:
        count = FETCH_COUNTS[period]
        data_source.save(data_source.fetch_kline(sym, period, count=count), table=table)
        df = realtime.period_history(sym, table)
    return df


def _trend(df, fast, slow):
    if len(df) < slow + 2:
        return None
    c = df["close"].astype(float)
    mf, ms = float(c.rolling(fast).mean().iloc[-1]), float(c.rolling(slow).mean().iloc[-1])
    if pd.isna(mf) or pd.isna(ms):
        return None
    return {"ma_fast": round(mf, 3), "ma_slow": round(ms, 3),
            "up": mf > ms, "bars": len(df)}


def analyze(symbol, price=None, cite=None):
    day_df = ensure(symbol, "day")
    base = br.analyze(day_df, price, cite)

    out = {"day": base, "week": {}, "month": {}, "tf_votes": [], "notes": []}

    wk = ensure(symbol, "week")
    wt = _trend(wk, 20, 60)
    if wt:
        out["week"]["trend"] = wt
        side = 1 if wt["up"] else -1
        base["votes"].append({
            "key": "tf_week", "rule": f"周线{'多头' if wt['up'] else '空头'}排列",
            "side": side, "weight": 0.75, "book": "b2", "cite": "移动平均线",
            "evidence": f"周线 MA20 {wt['ma_fast']} {'>' if wt['up'] else '<'} MA60 {wt['ma_slow']}"
                        f"（{wt['bars']} 周数据）", "page": None})
    ww = wv.analyze_wave(wk)
    out["week"]["wave"] = ww
    if ww.get("valid") and ww.get("sid"):
        base["votes"].append({
            "key": "tf_week_wave", "rule": f"周线波浪·{ww['position']}",
            "side": ww["sid"], "weight": 0.75, "book": "b5", "cite": "推动浪",
            "evidence": f"周线五浪结构，{ww['bars_since_last_pivot']} 周前完成，"
                        f"转折阈值 {int(ww['threshold_pct'] * 100)}%", "page": None})

    mo = ensure(symbol, "month")
    mt = _trend(mo, 12, 24)
    if mt:
        out["month"]["trend"] = mt
        side = 1 if mt["up"] else -1
        base["votes"].append({
            "key": "tf_month", "rule": f"月线{'多头' if mt['up'] else '空头'}排列",
            "side": side, "weight": 1.0, "book": "b2", "cite": "移动平均线",
            "evidence": f"月线 MA12 {mt['ma_fast']} {'>' if mt['up'] else '<'} MA24 {mt['ma_slow']}"
                        f"（{mt['bars']} 月数据）", "page": None})
    else:
        out["notes"].append("月线数据不足 24 根，未参与判断")

    # 大周期共振提示
    w_up = out["week"].get("trend", {}).get("up")
    m_up = out["month"].get("trend", {}).get("up")
    if w_up is not None and m_up is not None and w_up == m_up:
        out["alignment"] = f"周线与月线同向{'向上' if w_up else '向下'}"
    elif w_up is not None and m_up is not None:
        out["alignment"] = "周线与月线方向分歧，信号减半处理"
    else:
        out["alignment"] = "周期数据不完整，未做共振判断"

    score = round(sum(v["weight"] * v["side"] for v in base["votes"]), 2)
    if score >= 2:
        action, pos = "加仓", 0.6
    elif score >= 0.5:
        action, pos = "持有偏多", 0.45
    elif score > -0.5:
        action, pos = "观望", 0.3
    elif score > -2:
        action, pos = "减仓", 0.15
    else:
        action, pos = "清仓", 0.0
    base.update({"score": score, "action": action, "target_position": pos})

    if cite:
        for vt in base["votes"]:
            if vt.get("page"):
                continue
            try:
                hit = cite(vt["cite"], vt["book"])
                if hit:
                    vt["book"], vt["page"] = hit["book"], hit["page"]
            except Exception:
                pass

    out["action"] = action
    out["score"] = score
    out["target_position"] = pos
    out["votes"] = base["votes"]
    out["skipped"] = base.get("skipped", [])
    out["metrics"] = base.get("metrics", {})
    out["wave"] = base.get("wave")
    return out
