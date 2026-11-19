"""机械化的浪型识别：用 ZigZag 转折点找波段，再按书里的三条规则校验。

必须说明的局限：
- ZigZag 的转折幅度决定「浪的级别」。评分路径按波动率（ATR）选一档基础阈值，
  最多再看相邻一档，不做多档全扫——多档全扫会让随机走势也有很高概率凑出五浪。
- 图表展示另走 wave_groups()：允许小/中/大三档扫描并返回多组历史结构，
  但只用于画图，不参与评分，避免假阳性进入信号。
- 波浪理论本身允许多种数法，本模块只给出「在给定阈值下唯一确定的一种」；
- 推动浪允许两种结构：标准推动浪，以及浪4 与浪1 重叠但三段推动逐段收缩的
  终结楔形（ending diagonal）。三浪/ABC 调整结构不在此模块识别。
- 只有同时通过「新鲜度、级别、比例」三道校验的结构才返回有效方向；否则只报告
  原因，不参与评分。宁缺毋滥是刻意的。
- 因此输出标注为启发式判断，权重低于形态类规则，并在证据里写明启发式来源。
"""
import pandas as pd

# 结构必须贴近当下：超过这么多根之前完成的结构不再据此判断
RECENT_BARS = 15
MAX_DEVIATION = 0.10      # 现价与浪5终点的最大偏离
TURN_PCT = 0.02           # 浪5后反转的确认幅度
MIN_SPAN_PCT = 0.08       # 结构总跨度至少为现价的 8%
MIN_SPAN_ATR = 2.5        # 结构总跨度至少为 2.5 倍 ATR
MIN_SPAN_RANGE = 0.45     # 结构总跨度至少为近 120 日振幅的 45%
R2_MIN, R2_MAX = 0.236, 0.95   # 浪2 回撤区间
R4_MIN, R4_MAX = 0.10, 0.618   # 浪4 回撤区间（仅标准推动浪）
W3_DOMINANCE = 1.2        # 浪3 至少为浪1/浪5 中较长者的 1.2 倍（仅标准推动浪）
AMBIGUITY_RATIO = 0.75    # 第二组合法结构的跨度达到首选组的 75% 即视为数法不唯一
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)   # 回撤位，用于判断支撑/压力
DEGREE_NAMES = ("小", "中", "大")
DEGREE_MULTS = (1.0, 1.8, 3.2)   # 相对基础阈值的倍数，仅用于图表多级别展示


def zigzag(high, low, pct=0.05):
    """返回 [(index, price, 'H'|'L')]，相邻转折点方向交替。"""
    if len(high) < 3:
        return []
    piv = []
    direction = 0
    hi_i, lo_i = 0, 0
    hi, lo = float(high[0]), float(low[0])
    for i in range(1, len(high)):
        if direction >= 0:
            if high[i] > hi:
                hi, hi_i = float(high[i]), i
            # 前复权后早期价格可能为负（高分红股）。百分比阈值必须按绝对值计算，
            # 否则负价格乘 (1-pct) 会把「下跌阈值」抬到当前价上方。
            if hi != 0 and low[i] < hi - pct * abs(hi):
                piv.append((hi_i, hi, "H"))
                direction, lo, lo_i = -1, float(low[i]), i
        if direction <= 0:
            if low[i] < lo:
                lo, lo_i = float(low[i]), i
            if lo != 0 and high[i] > lo + pct * abs(lo):
                piv.append((lo_i, lo, "L"))
                direction, hi, hi_i = 1, float(high[i]), i
    if direction >= 0 and hi_i > (piv[-1][0] if piv else -1):
        piv.append((hi_i, hi, "H"))
    elif direction <= 0 and lo_i > (piv[-1][0] if piv else -1):
        piv.append((lo_i, lo, "L"))

    # 同一根 K 线同时成为高点和低点时只保留后者，避免出现零长度波段
    clean = []
    for p in piv:
        if clean and p[0] == clean[-1][0]:
            clean[-1] = p
        else:
            clean.append(p)
    return clean


def clean_zigzag(piv):
    """合并连续同向转折点，只保留更极端的那个。

    ZigZag 在快速行情里会连续输出两个高点或两个低点；不合并的话，按
    「方向必须交替」筛选五浪窗口时，最近几个窗口会被整段跳过。
    """
    out = []
    for idx, price, kind in piv:
        if out and out[-1][2] == kind:
            if (kind == "H" and price >= out[-1][1]) or \
                    (kind == "L" and price <= out[-1][1]):
                out[-1] = (idx, price, kind)
        else:
            out.append((idx, price, kind))
    return out


def _atr(df, n=14):
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def _scan_pcts(df):
    """按 ATR 选基础阈值，最多看相邻一档，避免多档全扫的假阳性。"""
    cur = float(df["close"].iloc[-1])
    a = _atr(df)
    if cur <= 0 or not pd.notna(a):
        base = 0.05
    else:
        base = min(0.08, max(0.03, MIN_SPAN_ATR * a / cur))
    return sorted({round(base, 3), round(min(0.08, base * 1.5), 3)})


def _legs(p, up):
    pts = [x[1] for x in p]
    p0, p1, p2, p3, p4, p5 = pts
    if up:
        return (p1 - p0, p3 - p2, p5 - p4), pts
    return (p0 - p1, p2 - p3, p4 - p5), pts


def _check(p, up):
    (w1, w3, w5), pts = _legs(p, up)
    p0, p1, p2, p3, p4, p5 = pts
    return {
        "浪2 未跌破浪1 起点": (p2 > p0) if up else (p2 < p0),
        "浪4 未进入浪1 价格区": (p4 > p1) if up else (p4 < p1),
        "浪3 不是最短的推动浪": w3 > min(w1, w5),
    }, pts


def _diagonal(p, up):
    """收缩型终结楔形：浪4 允许重叠浪1，但三段推动必须逐段收缩。

    标准推动浪要求浪4 不进入浪1 价格区；终结楔形是这条规则的合法例外，
    特征是浪1 > 浪3 > 浪5、浪4 回到浪1 区域内但不到浪2 终点、浪5 创新极值。
    不满足收缩条件时返回 None，仍按普通推动浪的规则判为不成立。
    """
    (w1, w3, w5), (p0, p1, p2, p3, p4, p5) = _legs(p, up)
    if not (w1 > w3 > w5 > 0):
        return None
    if up:
        tail = p2 > p0 and p3 > p1 and p5 > p3 and p2 < p4 < p1
    else:
        tail = p2 < p0 and p3 < p1 and p5 < p3 and p1 < p4 < p2
    if not tail:
        return None
    return {
        "浪2 未跌破浪1 起点": True,
        "浪4 与浪1 重叠，但浪1>浪3>浪5 收缩（终结楔形）": True,
        "浪3 不是最短的推动浪": True,
        "浪5 创新极值": True,
    }


def _structure(p, up):
    """返回 (结构名, 规则, 六个端点)；不成立时结构名为 None。"""
    rules, vals = _check(p, up)
    if all(rules.values()):
        return "推动浪", rules, vals
    diag = _diagonal(p, up)
    if diag:
        return "终结楔形", diag, vals
    return None, rules, vals


def _ratios(p, up):
    """返回 (浪2回撤, 浪4回撤, 浪5是否创新极值, 三段推动长度)。"""
    (w1, w3, w5), (p0, p1, p2, p3, p4, p5) = _legs(p, up)
    if up:
        r2 = (p1 - p2) / w1 if w1 else None
        r4 = (p3 - p4) / w3 if w3 else None
        newext = p5 > p3
    else:
        r2 = (p2 - p1) / w1 if w1 else None
        r4 = (p4 - p3) / w3 if w3 else None
        newext = p5 < p3
    return r2, r4, newext, (w1, w3, w5)


def _quality_reason(pattern, r2, r4, legs):
    w1, w3, w5 = legs
    if r2 is None:
        return "浪2 回撤不可计算"
    if not (R2_MIN <= r2 <= R2_MAX):
        return f"浪2 回撤 {r2:.0%} 不在 {R2_MIN:.0%}~{R2_MAX:.0%}"
    if pattern == "推动浪":
        if r4 is None:
            return "浪4 回撤不可计算"
        if not (R4_MIN <= r4 <= R4_MAX):
            return f"浪4 回撤 {r4:.0%} 不在 {R4_MIN:.0%}~{R4_MAX:.0%}"
        if w3 < W3_DOMINANCE * max(w1, w5):
            return f"浪3 仅为最长推动浪的 {w3 / max(w1, w5):.2f} 倍，未达 {W3_DOMINANCE} 倍"
    return None


def _describe(up, pattern, turned, failed):
    if failed:
        if up:
            return "五浪上涨失败（未创新高）后可能转入调整" if turned else "五浪上涨失败（未创新高），结构未完成"
        return "五浪下跌失败（未创新低）后可能转入反弹" if turned else "五浪下跌失败（未创新低），结构未完成"
    if up:
        if pattern == "终结楔形":
            return "五浪上涨（终结楔形）后可能转入调整" if turned else "处于上升五浪（终结楔形）中"
        return "五浪上涨后可能转入调整" if turned else "处于上升五浪中"
    if pattern == "终结楔形":
        return "五浪下跌（终结楔形）后可能转入反弹" if turned else "处于下降五浪（终结楔形）中"
    return "五浪下跌后可能转入反弹" if turned else "处于下降五浪中"


def _result(best, cur, n, alternatives):
    p, rules, vals, up, pattern = (best["pivots"], best["rules"], best["vals"],
                                   best["up"], best["pattern"])
    p0, p1, p2, p3, p4, p5 = vals
    bars_since = best["bars_since"]
    turned = (cur < p5 * (1 - TURN_PCT)) if up else (cur > p5 * (1 + TURN_PCT))
    if up:
        sid = -1 if turned else 1
    else:
        sid = 1 if turned else -1
    weight = 0.75 if best["failed"] else 1.5
    return {"ok": True, "valid": True, "direction": "上升" if up else "下降",
            "pattern": pattern, "failed": best["failed"], "rules": rules,
            "position": _describe(up, pattern, turned, best["failed"]),
            "sid": sid, "weight": weight,
            "bars_since_last_pivot": bars_since, "span_bars": best["span_bars"],
            "alternatives": alternatives,
            "ratios": {"浪2回撤": round(best["r2"], 3), "浪4回撤": round(best["r4"], 3),
                       "浪3/最长推动浪": round(best["legs"][1] / max(best["legs"][0], best["legs"][2]), 2)
                       if max(best["legs"][0], best["legs"][2]) else None},
            "pivots": [(int(i), round(float(v), 3)) for i, v, _ in p],
            "wave_ends": {"起点": round(p0, 3), "浪1": round(p1, 3), "浪2": round(p2, 3),
                          "浪3": round(p3, 3), "浪4": round(p4, 3), "浪5": round(p5, 3)},
            "price": round(cur, 3), "threshold_pct": best["pct"],
            "scanned_pivots": n}


def analyze_wave(df: pd.DataFrame, pct=None, scan=12):
    """在自适应阈值下搜索当前、级别足够、比例合规的五浪结构。

    pct 为 None 时按 ATR 选一档基础阈值并最多看相邻一档；显式传入时只按该
    阈值计算（回测按周线滚动时用）。只有同时满足以下条件才返回有效方向：
    - 结构终点在最近 RECENT_BARS 根内，且现价未偏离浪5终点超过 MAX_DEVIATION；
    - 总跨度达到现价 8%、2.5 ATR、近 120 日振幅 45% 三者中的全部要求；
    - 浪2/浪4 回撤在合理区间，标准推动浪的浪3 明显长于其余两段；
    - 不存在方向相反或跨度接近的其他合法数法。
    任一条件不满足时返回 valid=False 和真实原因，不参与评分。
    """
    if len(df) < 6:
        return {"ok": False, "why": f"K线仅 {len(df)} 根，不足以划分五浪"}
    pcts = (float(pct),) if pct is not None else _scan_pcts(df)
    cur = float(df["close"].iloc[-1])
    n = len(df)
    atr = _atr(df)
    win = df.tail(120)
    rng = float(win["high"].max() - win["low"].min())

    candidates, stale, rejects = [], [], []
    failed = None
    for th in pcts:
        piv = clean_zigzag(zigzag(df["high"], df["low"], th))
        if len(piv) < 6:
            continue
        for end in range(len(piv) - 1, 4, -1):
            p = piv[end - 5:end + 1]
            kinds = "".join(x[2] for x in p)
            if kinds not in ("LHLHLH", "HLHLHL"):
                continue
            up = kinds == "LHLHLH"
            pattern, rules, vals = _structure(p, up)
            bars_since = n - 1 - p[-1][0]
            if pattern is None:
                if failed is None or bars_since < failed["bars_since"]:
                    failed = {"bars_since": bars_since, "rules": rules, "pivots": p}
                continue

            span = max(vals) - min(vals)
            item = {"pivots": p, "rules": rules, "vals": vals, "up": up,
                    "pattern": pattern, "pct": th, "bars_since": bars_since,
                    "span": span, "span_bars": p[-1][0] - p[0][0]}

            if bars_since > RECENT_BARS:
                item["why"] = (f"最近一组五浪成立于 {bars_since} 根前，超过 {RECENT_BARS} 根，"
                               f"结构已陈旧，不据此判断")
                stale.append(item)
                continue
            dev = abs(cur / vals[5] - 1) if vals[5] else 0.0
            if dev > MAX_DEVIATION:
                item["why"] = (f"现价 {cur:.3f} 已偏离浪5终点 {vals[5]:.3f} "
                               f"超过 {int(MAX_DEVIATION * 100)}%，原结构失效")
                stale.append(item)
                continue

            if span / cur < MIN_SPAN_PCT:
                item["why"] = f"结构跨度仅为现价的 {span / cur:.1%}，未达 {MIN_SPAN_PCT:.0%}"
                rejects.append(item)
                continue
            if span < MIN_SPAN_ATR * atr:
                item["why"] = f"结构跨度仅为 {span / atr:.1f} 倍 ATR，未达 {MIN_SPAN_ATR} 倍" if atr else "ATR 不可用"
                rejects.append(item)
                continue
            if rng > 0 and span < MIN_SPAN_RANGE * rng:
                item["why"] = f"结构跨度仅为近 120 日振幅的 {span / rng:.0%}，未达 {MIN_SPAN_RANGE:.0%}"
                rejects.append(item)
                continue

            r2, r4, newext, legs = _ratios(p, up)
            reason = _quality_reason(pattern, r2, r4, legs)
            if reason:
                item["why"] = reason
                rejects.append(item)
                continue

            item.update({"r2": r2, "r4": r4, "legs": legs, "failed": not newext})
            candidates.append(item)

    if not candidates:
        if stale:
            best = min(stale, key=lambda c: c["bars_since"])
            return {"ok": True, "valid": False, "rules": best["rules"],
                    "why": best["why"],
                    "pivots": [(int(i), round(float(v), 3)) for i, v, _ in best["pivots"]]}
        if rejects:
            best = min(rejects, key=lambda c: c["bars_since"])
            return {"ok": True, "valid": False, "rules": best["rules"],
                    "why": "最近候选五浪未通过级别/比例校验：" + best["why"],
                    "pivots": [(int(i), round(float(v), 3)) for i, v, _ in best["pivots"]]}
        if failed is not None:
            bad = [k for k, v in failed["rules"].items() if not v]
            return {"ok": True, "valid": False, "rules": failed["rules"],
                    "why": "最近一组五浪划分不成立：" + "、".join(f"{b}不成立" for b in bad),
                    "pivots": [(int(i), round(float(v), 3)) for i, v, _ in failed["pivots"]]}
        return {"ok": False, "why": "最近转折点方向不构成五段式结构"}

    dirs = {c["up"] for c in candidates}
    if len(dirs) > 1:
        return {"ok": True, "valid": False,
                "why": f"同时存在上升与下降两种方向、共 {len(candidates)} 组合法五浪，"
                       f"方向不一致，不据此判断"}

    candidates.sort(key=lambda c: -c["span"])
    best = candidates[0]
    distinct, seen = [], set()
    for c in candidates:
        key = tuple(round(v, 3) for v in c["vals"])
        if key in seen:
            continue
        seen.add(key)
        distinct.append(c)
    if len(distinct) > 1 and distinct[1]["span"] >= AMBIGUITY_RATIO * best["span"]:
        return {"ok": True, "valid": False,
                "why": f"存在 {len(distinct)} 组跨度接近的合法数法"
                       f"（{best['span']:.3f} vs {distinct[1]['span']:.3f}），不据此判断",
                "alternatives": len(distinct),
                "pivots": [(int(i), round(float(v), 3)) for i, v, _ in best["pivots"]]}

    return _result(best, cur, n, len(distinct))


def fib_levels(start, end, ratios=FIB_RATIOS):
    """以 start→end 为整段，返回回撤位价格。

    上涨段（start < end）的回撤位落在 end 下方，下跌段落在 end 上方，
    两者都是「行情折返时可能遇到支撑/压力」的位置。方向无关，公式相同。
    """
    if start is None or end is None or start == end:
        return []
    span = float(end) - float(start)
    return [{"ratio": r, "price": round(float(end) - span * r, 3)} for r in ratios]


def last_swing(df):
    """最近一段已完成的波段（ZigZag 最后两个转折点）。

    没有有效五浪时用它做斐波那契回撤的锚点，调用方必须注明来源是「最近波段」
    而不是浪型，避免把一段普通波动说成数浪结果。
    """
    for th in _scan_pcts(df):
        piv = clean_zigzag(zigzag(df["high"], df["low"], th))
        if len(piv) >= 2:
            return piv[-2], piv[-1], th
    return None


def wave_groups(df, strict=True):
    """按小/中/大三个级别扫描整段 K 线，返回所有成立的五浪结构。

    与 analyze_wave 的分工：
    - analyze_wave 只看最近结构、参与评分；本函数扫描全历史、只用于画图，
      绝不要拿它的输出去打分，否则多档扫描的假阳性会污染信号。
    - 同一级别内结构不重叠：一组浪5 结束就是下一组的起点（共用端点）；
    - 不同级别允许嵌套，用来体现「大浪里套小浪」；
    - 每组仍然逐条校验五浪规则与比例，不成立的窗口直接跳过。

    strict=False 时只校验结构规则、不校验浪2/浪4 回撤与浪3 比例，用于展示
    更多边缘结构；默认 True，与评分口径保持一致。
    """
    if len(df) < 6:
        return []
    base = _scan_pcts(df)[0]
    best = {}
    for level, mult in enumerate(DEGREE_MULTS):
        th = round(min(0.25, max(0.02, base * mult)), 3)
        piv = clean_zigzag(zigzag(df["high"], df["low"], th))
        i = 0
        while i + 5 < len(piv):
            p = piv[i:i + 6]
            kinds = "".join(x[2] for x in p)
            if kinds not in ("LHLHLH", "HLHLHL"):
                i += 1
                continue
            up = kinds == "LHLHLH"
            pattern = _structure(p, up)[0]
            if pattern is None:
                i += 1
                continue
            r2, r4, newext, legs = _ratios(p, up)
            if strict and _quality_reason(pattern, r2, r4, legs):
                i += 1
                continue
            key = tuple(x[0] for x in p)
            # 同一组转折点在多个级别都可能成立时，保留更高级别
            if key not in best or level > best[key]["degree"]:
                best[key] = {
                    "degree": level, "degree_name": DEGREE_NAMES[level],
                    "threshold_pct": th, "up": up, "pattern": pattern,
                    "failed": not newext, "pivots": p,
                    "span_bars": p[-1][0] - p[0][0],
                }
            i += 5
    groups = sorted(best.values(), key=lambda g: (g["pivots"][0][0], g["degree"]))
    return groups
