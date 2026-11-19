"""条件式交易计划：把浪型、斐波那契、均线、江恩窗口与周月趋势合成建议。

只回答三件事：
1. 长线/短线当前偏多、偏空还是分歧；
2. 压力/支撑各在哪，强度来自哪些依据；
3. 到什么价位、满足什么条件才加减仓，跌破哪里失效。

所有价位都来自真实 K 线（浪型转折点、斐波那契、均线、近 120 日摆动点、
江恩锚点），不预测方向，也不生成不存在的数字。
"""
import math

import wave

CLUSTER_ATR = 0.30      # 同一价位簇的容差（ATR 与现价百分比取大者）
ZONE_ATR = 0.25         # 加/减仓区间的半宽
NEAR_ATR = 2.5          # 距离现价超过这么多 ATR 的支撑/压力不用于“现在加减”
MAX_LEVELS = 4


def _atr(df):
    return wave._atr(df)


def _fib_anchor(df, result):
    """优先当前五浪，其次最近的浪组，最后最近波段。"""
    piv = result.get("pivots") or []
    if result.get("valid") and len(piv) == 6:
        return piv[0], piv[-1], "当前五浪"
    groups = wave.wave_groups(df)
    if groups:
        g = max(groups, key=lambda x: x["pivots"][-1][0])
        return g["pivots"][0], g["pivots"][-1], f"最近一组{g['degree_name']}级别五浪"
    swing = wave.last_swing(df)
    if swing:
        a, b, _th = swing
        return a, b, "最近波段"
    return None


def _swing_points(df, price, atr, lookback=120):
    win = df.tail(lookback).reset_index(drop=True)
    if len(win) < 10:
        return []
    pct = min(0.08, max(0.03, 1.2 * (atr or 0.0) / price)) if price > 0 else 0.05
    piv = wave.clean_zigzag(wave.zigzag(win["high"], win["low"], pct))
    n = len(win)
    return [{"price": float(p), "bars_ago": n - 1 - int(i), "kind": k}
            for i, p, k in piv if n - 1 - int(i) >= 2]


def build_levels(price, atr, day_df, week_df, day_wave, week_wave, gann=None):
    """把各种依据的价位聚类成带来源和强度的支撑/压力。"""
    if not price or price <= 0:
        raise ValueError(f"现价不合法：{price}")
    atr = atr or _atr(day_df)
    tol = max(CLUSTER_ATR * (atr or 0.0), price * 0.004)
    cands = []

    def add(value, source, weight, bars_ago=None):
        try:
            p = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(p) or p <= 0:
            return
        if p < price * 0.55 or p > price * 1.45:
            return
        cands.append({"price": p, "source": source, "weight": float(weight),
                      "bars_ago": bars_ago})

    for df, result, prefix in ((day_df, day_wave, "日线"),
                               (week_df, week_wave, "周线")):
        anchor = _fib_anchor(df, result or {})
        if not anchor:
            continue
        (start, end, label) = anchor
        p0, p1 = float(start[1]), float(end[1])
        for lv in wave.fib_levels(p0, p1):
            strong = lv["ratio"] in (0.382, 0.618)
            add(lv["price"], f"{prefix}{label} {lv['ratio']:.3f} 回撤",
                2.0 if strong else 1.2)

    for n, w in ((20, 1.5), (60, 1.5), (250, 1.5)):
        ma = day_df["close"].rolling(n).mean().iloc[-1]
        if math.isfinite(float(ma)):
            add(ma, f"日线 MA{n}", w)
    for n, w in ((20, 1.5), (60, 1.5)):
        ma = week_df["close"].rolling(n).mean().iloc[-1]
        if math.isfinite(float(ma)):
            add(ma, f"周线 MA{n}", w)

    for p in _swing_points(day_df, price, atr):
        label = "高" if p["kind"] == "H" else "低"
        add(p["price"], f"近 120 日摆动{label}点", 1.0, p["bars_ago"])

    win = day_df.tail(120)
    add(win["low"].min(), "近 120 日最低", 1.0)
    add(win["high"].max(), "近 120 日最高", 1.0)

    for a in (gann or {}).get("anchors") or []:
        add(a.get("price"), f"江恩锚点 {a.get('date', '')}", 0.8)

    clusters = []
    for c in sorted(cands, key=lambda x: x["price"]):
        if clusters and abs(c["price"] - clusters[-1]["price"]) <= tol:
            clusters[-1]["members"].append(c)
        else:
            clusters.append({"price": c["price"], "members": [c]})

    out = []
    for cl in clusters:
        members = cl["members"]
        total_w = sum(m["weight"] for m in members)
        avg = sum(m["price"] * m["weight"] for m in members) / total_w
        sources, seen = [], set()
        for m in members:
            if m["source"] not in seen:
                seen.add(m["source"])
                sources.append(m["source"])
        bars = [m["bars_ago"] for m in members if m["bars_ago"] is not None]
        out.append({
            "price": round(avg, 3),
            "strength": round(total_w, 1),
            "sources": sources,
            "bars_ago": min(bars) if bars else None,
            "distance_pct": round((avg - price) / price * 100, 2),
            "distance_atr": round(abs(avg - price) / atr, 2) if atr else None,
        })

    support = sorted([x for x in out if x["price"] < price],
                     key=lambda x: price - x["price"])[:MAX_LEVELS]
    resistance = sorted([x for x in out if x["price"] > price],
                        key=lambda x: x["price"] - price)[:MAX_LEVELS]
    return {"support": support, "resistance": resistance, "atr": round(float(atr), 3)
            if atr else None, "cluster_tol": round(tol, 3)}


def _zone(level, price, atr):
    half = max(ZONE_ATR * (atr or 0.0), price * 0.003)
    return {"price": level["price"], "low": round(level["price"] - half, 3),
            "high": round(level["price"] + half, 3),
            "sources": level["sources"], "strength": level["strength"],
            "distance_atr": level["distance_atr"]}


def _pick(levels, price, atr, side, min_strength=2.0, max_atr=NEAR_ATR):
    near = [x for x in levels[side]
            if x["distance_atr"] is not None and x["distance_atr"] <= max_atr]
    strong = [x for x in near if x["strength"] >= min_strength]
    return (strong or near or [None])[0]


def _stop_below(levels, zone_price):
    below = [x for x in levels["support"] if x["price"] < zone_price]
    if not below:
        return None
    return max(below, key=lambda x: x["strength"])


def _long_plan(levels, price, atr, analysis, week_wave):
    month = (analysis.get("month") or {}).get("trend") or {}
    week = (analysis.get("week") or {}).get("trend") or {}
    m_up, w_up = month.get("up"), week.get("up")
    reasons = []
    if m_up is True:
        reasons.append("月线 MA12>MA24")
    elif m_up is False:
        reasons.append("月线 MA12<MA24")
    if w_up is True:
        reasons.append("周线 MA20>MA60")
    elif w_up is False:
        reasons.append("周线 MA20<MA60")
    wave_up = None
    if (week_wave or {}).get("valid"):
        wave_up = week_wave.get("direction") == "上升"
        reasons.append(f"周线{week_wave.get('direction')}{week_wave.get('pattern')}")

    if m_up is True and w_up is True:
        bias = "偏多"
    elif m_up is False and w_up is False:
        bias = "偏空"
    else:
        bias = "分歧"
    if bias == "偏多" and wave_up is False:
        bias = "分歧"
        reasons.append("周线浪型向下，与均线方向冲突")
    elif bias == "偏空" and wave_up is True:
        bias = "分歧"
        reasons.append("周线浪型向上，与均线方向冲突")

    add = reduce = stop = None
    if bias == "偏多":
        at = _pick(levels, price, atr, "support", 2.0, 3.0)
        if at is None:
            action = "持有，等新的支撑形成后再加"
        else:
            add = _zone(at, price, atr)
            if at["distance_atr"] is not None and at["distance_atr"] > 2.0:
                action = f"不追高，等回踩 {at['price']} 附近再分批加"
            else:
                action = f"回踩 {at['price']} 附近且收盘不破，可分批加仓"
            low = _stop_below(levels, at["price"])
            stop = {"price": low["price"] if low else add["low"],
                    "why": "收盘跌破则本次加仓失效" if low else "跌破加仓区下沿失效"}
        res = _pick(levels, price, atr, "resistance", 2.0, 3.0)
        if res:
            reduce = _zone(res, price, atr)
            action += f"；反弹到 {res['price']} 附近先减一部分，不追高"
    elif bias == "偏空":
        at = _pick(levels, price, atr, "resistance", 2.0, 3.0)
        if at:
            reduce = _zone(at, price, atr)
            action = f"反弹至 {at['price']} 附近分批减仓；收盘站上则暂停减仓"
            stop = {"price": at["price"], "why": "收盘站上则空头计划失效"}
        else:
            action = "反弹减仓，但不追空；等压力位确认"
    else:
        action = "观望；等周线与月线方向一致后再决定加或减"
        sup = _pick(levels, price, atr, "support", 2.0, 2.0)
        res = _pick(levels, price, atr, "resistance", 2.0, 2.0)
        if sup and res and (res["price"] - sup["price"]) < max(1.0 * (atr or 0.0),
                                                             price * 0.02):
            action += "；支撑与压力过近，区间太窄，等方向选择"
        else:
            if sup:
                add = _zone(sup, price, atr)
            if res:
                reduce = _zone(res, price, atr)
            if sup and res:
                action += f"，区间参考 {sup['price']} ~ {res['price']}"

    return {"bias": bias, "action": action, "why": "；".join(reasons) or "周期数据不足",
            "add_zone": add, "reduce_zone": reduce, "stop": stop,
            "size": "分批执行，单次不超过目标仓位的 1/3" if bias != "偏空" else "以减仓为主"}


def _short_plan(levels, price, atr, day_wave, gann):
    ma20 = day_wave.get("_ma20")
    ma60 = day_wave.get("_ma60")
    score = 0.0
    reasons = []
    if day_wave.get("valid"):
        up = day_wave.get("direction") == "上升"
        score += 1.0 if up else -1.0
        reasons.append(f"日线{day_wave.get('direction')}{day_wave.get('pattern')}")
    else:
        reasons.append(f"日线浪型未通过校验：{day_wave.get('why') or '无'}")

    if ma20 is not None:
        if price > ma20:
            score += 0.5
            reasons.append("价在 MA20 上方")
        else:
            score -= 0.5
            reasons.append("价在 MA20 下方")
    if ma20 is not None and ma60 is not None:
        if ma20 > ma60:
            score += 0.5
            reasons.append("MA20>MA60")
        else:
            score -= 0.5
            reasons.append("MA20<MA60")

    g = (gann or {}).get("current") or {}
    side = g.get("side") or 0
    score += 0.5 * side
    if side:
        reasons.append(f"江恩：{g.get('head', '')}")

    bias = "偏多" if score >= 1.0 else ("偏空" if score <= -1.0 else "分歧")
    add = reduce = stop = None
    if bias == "偏多":
        at = _pick(levels, price, atr, "support", 1.5, 1.8)
        if at:
            add = _zone(at, price, atr)
            action = f"不追高；回踩 {at['price']} 且收盘不破可短加"
            low = _stop_below(levels, at["price"])
            stop = {"price": low["price"] if low else add["low"],
                    "why": "跌破短加支撑失效"}
        else:
            action = "偏多但离支撑较远，等回踩，不追高"
        res = _pick(levels, price, atr, "resistance", 1.5, 1.5)
        if res:
            reduce = _zone(res, price, atr)
            action += f"；接近 {res['price']} 先减"
    elif bias == "偏空":
        at = _pick(levels, price, atr, "resistance", 1.5, 1.8)
        if at:
            reduce = _zone(at, price, atr)
            action = f"反弹至 {at['price']} 附近减；收盘站上则解除"
            stop = {"price": at["price"], "why": "收盘站上则短空失效"}
        else:
            action = "偏空但离压力较远，反弹减，不追空"
    else:
        sup = _pick(levels, price, atr, "support", 1.5, 1.5)
        res = _pick(levels, price, atr, "resistance", 1.5, 1.5)
        if sup and res and (res["price"] - sup["price"]) < max(1.0 * (atr or 0.0),
                                                             price * 0.02):
            action = "支撑与压力过近，区间太窄，观望等方向"
        else:
            if sup:
                add = _zone(sup, price, atr)
            if res:
                reduce = _zone(res, price, atr)
            action = (f"区间 {sup['price']} ~ {res['price']} 内高抛低吸/观望"
                      if sup and res else "方向分歧，观望")

    return {"bias": bias, "action": action, "why": "；".join(reasons),
            "add_zone": add, "reduce_zone": reduce, "stop": stop,
            "gann": g.get("head") or "", "gann_action": g.get("action") or ""}


def build(symbol, name, day_df, week_df, price, analysis, gann, day_wave, week_wave):
    atr = _atr(day_df)
    ma20 = float(day_df["close"].rolling(20).mean().iloc[-1])
    ma60 = float(day_df["close"].rolling(60).mean().iloc[-1])
    day_wave = dict(day_wave or {})
    day_wave["_ma20"] = ma20 if math.isfinite(ma20) else None
    day_wave["_ma60"] = ma60 if math.isfinite(ma60) else None

    levels = build_levels(price, atr, day_df, week_df, day_wave, week_wave, gann)
    long_plan = _long_plan(levels, price, atr, analysis or {}, week_wave)
    short_plan = _short_plan(levels, price, atr, day_wave, gann)

    warnings = []
    rule_action = (analysis or {}).get("action")
    if rule_action and long_plan["bias"] == "偏空" and rule_action in ("加仓", "持有偏多"):
        warnings.append(f"规则评分当前为「{rule_action}」，但周/月结构偏空；以结构失效位为准，不追多。")
    if rule_action and long_plan["bias"] == "偏多" and rule_action in ("减仓", "清仓"):
        warnings.append(f"规则评分当前为「{rule_action}」，但周/月结构偏多；若回踩支撑不破，不必一次减完。")
    if not (week_wave or {}).get("valid"):
        warnings.append("周线浪型未通过校验，长线方向主要看周/月均线。")
    if not (analysis.get("month") or {}).get("trend"):
        warnings.append("月线数据不可用，长线方向仅看周线。")
    if not (day_wave or {}).get("valid"):
        warnings.append("日线浪型未通过校验，短线方向主要看均线与江恩窗口。")
    if long_plan["bias"] == "偏多" and short_plan["bias"] == "偏空":
        warnings.append("长线偏多、短线偏空：长线加仓宜等短线站回 MA20/压力位后再执行，"
                        "避免在回调途中一次加满。")
    elif long_plan["bias"] == "偏空" and short_plan["bias"] == "偏多":
        warnings.append("长线偏空、短线偏多：短线反弹以减仓为主，不宜当作长线反转。")

    conditions = []
    if long_plan.get("add_zone"):
        z = long_plan["add_zone"]
        conditions.append(f"长线加仓：回踩 {z['low']}~{z['high']} 且收盘不破，分批加。")
    if long_plan.get("stop"):
        conditions.append(f"长线失效：收盘跌破 {long_plan['stop']['price']}。")
    if long_plan.get("reduce_zone"):
        z = long_plan["reduce_zone"]
        conditions.append(f"长线减仓：反弹至 {z['low']}~{z['high']} 先减一部分。")
    if short_plan.get("add_zone"):
        z = short_plan["add_zone"]
        conditions.append(f"短线加仓：回踩 {z['low']}~{z['high']} 且收盘不破。")
    if short_plan.get("reduce_zone"):
        z = short_plan["reduce_zone"]
        conditions.append(f"短线减仓：反弹至 {z['low']}~{z['high']} 附近。")
    if short_plan.get("stop"):
        conditions.append(f"短线失效：收盘跌破或站上 {short_plan['stop']['price']}。")

    asof = ""
    if "trade_date" in day_df.columns and len(day_df):
        asof = str(day_df["trade_date"].iloc[-1])[:10]
    return {
        "symbol": symbol, "name": name,
        "asof": asof or str((gann or {}).get("asof") or ""),
        "price": round(float(price), 3),
        "atr": levels["atr"],
        "long": long_plan, "short": short_plan,
        "levels": {"support": levels["support"], "resistance": levels["resistance"]},
        "rule_action": rule_action,
        "rule_target_position": (analysis or {}).get("target_position"),
        "conditions": conditions,
        "warnings": warnings,
        "disclaimer": ("以上为基于浪型、斐波那契、均线与时间窗口的条件式计划，"
                       "不是价格预测，也不构成投资建议；结构失效时应重新评估。"),
    }
