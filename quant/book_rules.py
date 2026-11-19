"""按书里的规则给出加减仓建议，每条结论都对应到原书页码。

规则来源：
  b1 日本蜡烛图技术新解 —— 吞噬、黄昏星、启明星、锤子、流星、十字星
  b2 期货市场技术分析   —— 跳空、相对力度、支撑线、移动平均线、趋势线
  b3 道氏理论           —— 主要趋势结构、成交量确认
  b5 艾略特波浪理论     —— 斐波那契回撤位

不使用书外的通用指标作为主要依据；ATR 只用于给止损留出波动空间，并在结果里标注。
"""
import numpy as np
import pandas as pd

import wave as wv

# 规则名 -> 用于定位原书页码的检索词
CITE = {
    "engulf": "吞噬形态", "evening_star": "黄昏星", "morning_star": "启明星",
    "hammer": "锤子线", "shooting_star": "流星线", "doji": "十字星",
    "gap": "跳空缺口", "rsi": "相对力度", "support": "支撑线",
    "ma": "移动平均线", "volume": "交易量", "dow": "基本趋势",
    "fib": "斐波那契回撤", "wave": "推动浪",
}


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def analyze(df: pd.DataFrame, price: float | None = None, cite=None):
    """返回 vote 列表，每条含规则名、方向、权重、原书页码、证据。"""
    if len(df) < 70:
        return {"action": "数据不足", "score": 0, "votes": [], "skipped": [],
                "metrics": {"bars": len(df)}, "target_position": None}

    o, c, h, l, v = (df[x].astype(float) for x in ("open", "close", "high", "low", "volume"))
    px = float(price) if price else float(c.iloc[-1])
    i = -1
    ma20, ma60 = c.rolling(20).mean().iloc[i], c.rolling(60).mean().iloc[i]
    atr = float(_atr(df).iloc[i])
    body = abs(c.iloc[i] - o.iloc[i])
    rng = max(h.iloc[i] - l.iloc[i], 1e-9)
    prev_body = abs(c.iloc[i - 1] - o.iloc[i - 1])
    vol_ma = float(v.rolling(20).mean().iloc[i])

    votes, skipped = [], []

    def add(key, rule, side, weight, book, evidence):
        votes.append({"key": key, "rule": rule, "side": side, "weight": weight,
                      "book": book, "evidence": evidence, "cite": CITE.get(key, "")})

    # ---- b1 蜡烛图形态 ----
    bull_engulf = (c.iloc[i] > o.iloc[i] and c.iloc[i - 1] < o.iloc[i - 1]
                   and c.iloc[i] >= o.iloc[i - 1] and o.iloc[i] <= c.iloc[i - 1])
    bear_engulf = (c.iloc[i] < o.iloc[i] and c.iloc[i - 1] > o.iloc[i - 1]
                   and c.iloc[i] <= o.iloc[i - 1] and o.iloc[i] >= c.iloc[i - 1])
    downtrend = c.iloc[i] < ma20
    uptrend = c.iloc[i] > ma20
    if bull_engulf:
        add("engulf", "看涨吞噬形态", 1, 2.0, "b1",
            f"今日阳线实体完全包住昨日阴线实体（{o.iloc[i]:.3f}~{c.iloc[i]:.3f}）"
            + ("，且处于均线下方的下跌背景中，符合书中「形态需有趋势背景」的要求" if downtrend
               else "，但不在下跌背景下，书中强调形态需结合趋势判断，故力度有限"))
    elif bear_engulf:
        add("engulf", "看跌吞噬形态", -1, 2.0, "b1",
            f"今日阴线实体完全包住昨日阳线实体（{o.iloc[i]:.3f}~{c.iloc[i]:.3f}）")

    # 黄昏星 / 启明星：三根K线
    if i >= -3:
        b1_, b2_, b3_ = (abs(c.iloc[j] - o.iloc[j]) for j in (-3, -2, -1))
        if (c.iloc[-3] > o.iloc[-3] and b1_ > 0 and b2_ < b1_ * 0.6
                and c.iloc[-1] < o.iloc[-1] and c.iloc[-1] <= (o.iloc[-3] + c.iloc[-3]) / 2):
            add("evening_star", "黄昏星形态", -1, 2.0, "b1",
                "三根K线：长阳 → 小实体 → 长阴，且第三根收在前一根阳线中点下方")
        if (c.iloc[-3] < o.iloc[-3] and b1_ > 0 and b2_ < b1_ * 0.6
                and c.iloc[-1] > o.iloc[-1] and c.iloc[-1] >= (o.iloc[-3] + c.iloc[-3]) / 2):
            add("morning_star", "启明星形态", 1, 2.0, "b1",
                "三根K线：长阴 → 小实体 → 长阳，且第三根收在前一根阴线中点上方")

    lower_shadow = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
    upper_shadow = h.iloc[i] - max(o.iloc[i], c.iloc[i])
    if body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.5 * body:
        add("hammer", "锤子线", 1, 1.5 if downtrend else 1.0, "b1",
            f"下影线为实体的 {lower_shadow / body:.1f} 倍、上影线极短"
            + ("，出现在下跌之后" if downtrend else "，但不在明显下跌之后，书中要求锤子需处于下跌末端"))
    if body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.5 * body:
        add("shooting_star", "流星线", -1, 1.5 if uptrend else 1.0, "b1",
            f"上影线为实体的 {upper_shadow / body:.1f} 倍、下影线极短"
            + ("，出现在上涨之后" if uptrend else "，但不在上涨之后，参考价值有限"))

    if body <= 0.1 * rng:
        add("doji", "十字星", 0, 0.0, "b1",
            "实体不足全幅的 10%，体现多空僵持，本身不构成买卖信号，需等下一根确认")

    # ---- b2 跳空 ----
    if l.iloc[i] > h.iloc[i - 1]:
        add("gap", "向上跳空", 1, 0.5, "b2",
            f"今日最低 {l.iloc[i]:.3f} 高于昨日最高 {h.iloc[i - 1]:.3f}")
    elif h.iloc[i] < l.iloc[i - 1]:
        add("gap", "向下跳空", -1, 0.5, "b2",
            f"今日最高 {h.iloc[i]:.3f} 低于昨日最低 {l.iloc[i - 1]:.3f}")

    # ---- b2 相对力度指数 ----
    r = float(_rsi(c).iloc[i])
    if np.isfinite(r):
        if r >= 70:
            add("rsi", "相对力度超买", -1, 1.0, "b2", f"RSI14 = {r:.0f}，进入超买区")
        elif r <= 30:
            add("rsi", "相对力度超卖", 1, 1.0, "b2", f"RSI14 = {r:.0f}，进入超卖区")

    # ---- b2 移动平均线 ----
    if px > ma20 > ma60:
        add("ma", "均线多头排列", 1, 1.0, "b2",
            f"现价 {px:.3f} > MA20 {ma20:.3f} > MA60 {ma60:.3f}")
    elif px < ma20 < ma60:
        add("ma", "均线空头排列", -1, 1.0, "b2",
            f"现价 {px:.3f} < MA20 {ma20:.3f} < MA60 {ma60:.3f}")

    # ---- b2 支撑线 ----
    support = float(l.iloc[-20:].min())
    if 0 <= px - support <= 1.0 * atr:
        add("support", "接近前期支撑", 1, 1.0, "b2",
            f"现价 {px:.3f} 距 20 日低点支撑 {support:.3f} 仅 {(px - support) / atr:.1f} 倍 ATR")

    # ---- b3 道氏趋势结构 ----
    rh, ph = h.iloc[-10:].max(), h.iloc[-20:-10].max()
    rl, pl = l.iloc[-10:].min(), l.iloc[-20:-10].min()
    if rh > ph and rl > pl:
        add("dow", "道氏上升结构", 1, 1.5, "b3",
            f"近期高点 {rh:.3f}>{ph:.3f}、低点 {rl:.3f}>{pl:.3f}，高点低点同步抬高")
    elif rh < ph and rl < pl:
        add("dow", "道氏下降结构", -1, 1.5, "b3",
            f"近期高点 {rh:.3f}<{ph:.3f}、低点 {rl:.3f}<{pl:.3f}，高点低点同步下移")

    # ---- b3/b1 成交量确认 ----
    if vol_ma > 0 and v.iloc[i] > vol_ma * 1.2:
        if c.iloc[i] > c.iloc[i - 1]:
            add("volume", "放量上涨", 1, 0.5, "b3",
                f"成交量 {v.iloc[i] / vol_ma:.1f} 倍于 20 日均量，价格同步上行，趋势得到确认")
        elif c.iloc[i] < c.iloc[i - 1]:
            add("volume", "放量下跌", -1, 0.5, "b3",
                f"成交量 {v.iloc[i] / vol_ma:.1f} 倍于 20 日均量，价格下跌，抛压得到确认")

    # ---- b5 斐波那契回撤 ----
    win = df.iloc[-60:]
    lo_i, hi_i = win["low"].idxmin(), win["high"].idxmax()
    if lo_i < hi_i:
        lo, hi = float(win["low"].min()), float(win["high"].max())
        if hi > lo:
            retr = (hi - px) / (hi - lo)
            if 0.382 <= retr <= 0.618:
                add("fib", "斐波那契回撤位", 1, 1.0, "b5",
                    f"自 {lo:.3f} 涨至 {hi:.3f} 后回撤 {retr:.1%}，位于 0.382~0.618 区间")

    # ---- b5 波浪结构（ZigZag + 三条规则）----
    w = wv.analyze_wave(df)
    if w.get("ok") and w.get("sid") and w.get("valid"):
        rules_txt = "、".join(f"{k}{'成立' if ok else '不成立'}" for k, ok in w["rules"].items())
        add("wave", f"波浪结构·{w['position']}", w["sid"], w.get("weight", 1.5), "b5",
            f"{w['direction']}五浪结构（{w['bars_since_last_pivot']} 根K线前完成）：{rules_txt}；"
            f"浪1/3/5 终点 {w['wave_ends']['浪1']}/{w['wave_ends']['浪3']}/{w['wave_ends']['浪5']}；"
            f"转折阈值 {int(w['threshold_pct'] * 100)}%，启发式划分")
    else:
        why = w.get("why")
        if not why:
            bad = [k for k, ok in w["rules"].items() if not ok]
            why = "推动浪三条规则未全部成立：" + "、".join(f"{b}不成立" for b in bad)
        skipped.append({"rule": "波浪结构", "book": "b5", "why": why})

    # ---- 书中不适用于单一标的的规则，明确列出 ----
    skipped.append({"rule": "相互验证", "book": "b3",
                    "why": "道氏的相互验证需要成对指数（如工业与运输），单一标的不适用"})

    score = round(sum(x["weight"] * x["side"] for x in votes), 2)
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

    if cite:
        for vt in votes:
            try:
                hits = cite(vt["cite"], vt["book"])
                if hits:
                    vt["book"] = hits["book"]
                    vt["page"] = hits["page"]
            except Exception:
                vt["page"] = None

    return {"action": action, "score": score, "target_position": pos,
            "votes": votes, "skipped": skipped, "wave": w,
            "metrics": {"price": round(px, 3), "ma20": round(float(ma20), 3),
                        "ma60": round(float(ma60), 3), "rsi14": round(r, 1) if np.isfinite(r) else None,
                        "atr": round(atr, 4), "support": round(support, 3),
                        "stop_price": round(support - 0.5 * atr, 3)}}
