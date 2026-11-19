"""固定规则层：江恩时间周期 / 波浪结构 / 斐波那契 / 易经六爻。

所有函数只接受"t 时点已确认"的转折点序列，不使用未来数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ===================================================================== 江恩

CYCLES = (9, 18, 27, 36, 45, 60)


def gann_windows(piv: pd.DataFrame, n_bars: int, designated: dict | None = None) -> pd.DataFrame:
    """从每个已确认转折点投影 9/18/27/36/45/60 日周期窗口。

    designated: {pd.Timestamp: 名称}，用户指定锚点（无对应转折点时用当日收盘价与ATR）。
    返回列 anchor_idx anchor_date anchor_price anchor_atr cycle center_idx lo_idx hi_idx
    """
    rows = []
    for _, r in piv.iterrows():
        a_idx = int(r["idx"])
        a_atr = float(r["atr"]) if "atr" in r and np.isfinite(r["atr"]) else np.nan
        for c in CYCLES:
            ctr = a_idx + c
            rows.append({
                "anchor_idx": a_idx,
                "anchor_date": r["date"],
                "anchor_price": float(r["price"]),
                "anchor_atr": a_atr,
                "cycle": c,
                "center_idx": ctr,
                "lo_idx": ctr - 1,
                "hi_idx": ctr + 1,
            })
    out = pd.DataFrame(rows)
    return out


def gann_resonance(win: pd.DataFrame, t: int) -> pd.DataFrame:
    """t 时点附近 ±1 日内落有 >=2 个周期窗口，标记为时间共振。"""
    near = win[(win["lo_idx"] <= t + 1) & (win["hi_idx"] >= t - 1)]
    if near.empty:
        return near
    return near.sort_values("center_idx")


def gann_angles(anchor_price: float, anchor_atr: float, days: int,
                direction: int = 1) -> dict:
    """ATR 标准化江恩角度线。1单位价格 = 锚点ATR14；1单位时间 = 1交易日。

    direction=+1 向上，-1 向下。返回各斜率线在"锚点后第 days 日"的价格。
    """
    unit = anchor_atr
    return {
        "1x1": anchor_price + direction * 1.0 * unit * days,
        "1x2": anchor_price + direction * 2.0 * unit * days,
        "2x1": anchor_price + direction * 0.5 * unit * days,
    }


def gann_stats(df: pd.DataFrame, piv: pd.DataFrame, last_n: int | None = 10) -> pd.DataFrame:
    """统计已实现窗口的前后3日/5日收益与反转命中率（按周期分组）。

    只统计最近 last_n 个"阶段"转折点（默认10个 × 6周期 = 60 个窗口），
    避免早期已被替换的转折点稀释统计。
    """
    close = df["close"].to_numpy(float)
    n = len(close)
    use = piv.tail(last_n) if last_n else piv
    rows = []
    for _, r in use.iterrows():
        for c in CYCLES:
            ctr = int(r["idx"]) + c
            if ctr - 4 < 0 or ctr + 5 >= n:
                continue
            pre3 = close[ctr - 1] / close[ctr - 4] - 1.0
            post3 = close[ctr + 3] / close[ctr] - 1.0
            post5 = close[ctr + 5] / close[ctr] - 1.0
            rows.append({
                "cycle": c, "center_idx": ctr, "center_date": df["date"].iloc[ctr],
                "pre3": pre3, "post3": post3, "post5": post5,
                "reversal": bool(np.sign(pre3) * np.sign(post3) < 0),
            })
    return pd.DataFrame(rows)


def reversal_base_rate(df: pd.DataFrame) -> float:
    """无条件基率：任意一日的前3日方向与后3日方向相反的比例。"""
    c = df["close"].to_numpy(float)
    n = len(c)
    hit = tot = 0
    for t in range(3, n - 3):
        pre = np.sign(c[t] - c[t - 3])
        post = np.sign(c[t + 3] - c[t])
        if pre == 0 or post == 0:
            continue
        tot += 1
        hit += int(pre * post < 0)
    return hit / tot if tot else float("nan")


# ===================================================================== 斐波那契

RETRACE = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTEND = (1.0, 1.272, 1.618, 2.0)
TIME_RATIOS = (0.382, 0.5, 0.618, 1.0, 1.618)


def _swing_dict(a, b, label: str) -> dict:
    hi, lo = max(a["price"], b["price"]), min(a["price"], b["price"])
    return {
        "label": label,
        "start_date": a["date"], "start_price": float(a["price"]),
        "end_date": b["date"], "end_price": float(b["price"]),
        "high": float(hi), "low": float(lo), "range": float(hi - lo),
        "direction": -1 if a["kind"] == "H" else 1,
        "start_idx": int(a["idx"]), "end_idx": int(b["idx"]),
        "bars": int(b["idx"] - a["idx"]),
        "wave_kind": "high->low" if a["kind"] == "H" else "low->high",
    }


def fib_swing(piv: pd.DataFrame, min_atr_mult: float = 3.0, min_bars: int = 3) -> dict | None:
    """最近一个已确认的**主要**波段：向后取第一个振幅 >= min_atr_mult×ATR 的相邻转折点对。"""
    if len(piv) < 2:
        return None
    for i in range(len(piv) - 2, -1, -1):
        a, b = piv.iloc[i], piv.iloc[i + 1]
        if a["kind"] == b["kind"]:
            continue
        rngp = abs(a["price"] - b["price"])
        if rngp < min_atr_mult * float(a["atr"]) or (b["idx"] - a["idx"]) < min_bars:
            continue
        return _swing_dict(a, b, "当前波段")
    return None


def fib_structural(piv: pd.DataFrame) -> dict | None:
    """全段结构：样本内已确认转折点的最低点与最高点组成的大级别波段。"""
    if len(piv) < 4:
        return None
    lo_row = piv.loc[piv["price"].idxmin()]
    hi_row = piv.loc[piv["price"].idxmax()]
    a, b = (lo_row, hi_row) if lo_row["idx"] < hi_row["idx"] else (hi_row, lo_row)
    return _swing_dict(a, b, "全段结构")


def fib_levels(sw: dict) -> pd.DataFrame:
    """回撤位（相对波段终点）与扩展位。"""
    lo, hi, rng = sw["low"], sw["high"], sw["range"]
    rows = []
    for r in RETRACE:
        rows.append({"kind": "retrace", "ratio": r,
                     "price": lo + r * rng if sw["direction"] < 0 else hi - r * rng})
    for e in EXTEND:
        rows.append({"kind": "extend", "ratio": e,
                     "price": lo - (e - 1.0) * rng if sw["direction"] < 0
                     else hi + (e - 1.0) * rng})
    return pd.DataFrame(rows)


def fib_time_windows(sw: dict) -> pd.DataFrame:
    rows = []
    for r in TIME_RATIOS:
        rows.append({"ratio": r, "offset": r * sw["bars"],
                     "center_idx": int(sw["start_idx"] + round(r * sw["bars"]))})
    return pd.DataFrame(rows)


# ===================================================================== 波浪

def _impulse_ok(p: list[dict], up: bool) -> tuple[bool, list[str]]:
    """五浪推动校验。p 为 6 个交替转折点（P0..P5），up=True 表示向上推动。"""
    notes = []
    if up:
        need = ["L", "H", "L", "H", "L", "H"]
    else:
        need = ["H", "L", "H", "L", "H", "L"]
    if [x["kind"] for x in p] != need:
        return False, ["转折点类型不交替"]

    v = [x["price"] for x in p]
    w1 = abs(v[1] - v[0]); w3 = abs(v[3] - v[2]); w5 = abs(v[5] - v[4])
    if up:
        if v[2] <= v[0]:
            return False, ["第二浪跌破第一浪起点"]
        if v[3] <= v[1]:
            return False, ["第三浪未创新高"]
        if v[4] <= v[1]:
            notes.append("第四浪与第一浪重叠（推动浪存疑）")
    else:
        if v[2] >= v[0]:
            return False, ["第二浪升破第一浪起点"]
        if v[3] >= v[1]:
            return False, ["第三浪未创新低"]
        if v[4] >= v[1]:
            notes.append("第四浪与第一浪重叠（推动浪存疑）")
    if w3 <= min(w1, w5):
        return False, ["第三浪为最短推动浪"]
    return True, notes


def analyze_wave(piv: pd.DataFrame, price_now: float) -> dict:
    """尝试标注当前波浪。失败即返回结构不明，不强行数浪。"""
    if len(piv) < 4:
        return {"state": "结构不明", "detail": "已确认转折点不足4个",
                "invalid": None, "alts": [], "label_code": 0.0,
                "last_pivots": piv.tail(2).to_dict("records")}

    lp = piv.iloc[-1]
    up_next = lp["kind"] == "L"          # 最后是低点 -> 可能向上推动

    # 情形A：完整五浪已经走完（最后6个转折点）
    if len(piv) >= 6:
        p6 = piv.tail(6).to_dict("records")
        ok, notes = _impulse_ok(p6, up_next)
        if ok:
            w = 5
            return {"state": f"推动浪第{w}浪结束（{'上涨' if up_next else '下跌'}）",
                    "detail": "；".join(notes) or "五浪结构成立",
                    "invalid": float(p6[0]["price"]), "alts": [],
                    "label_code": float(w if up_next else -w), "last_pivots": p6[-2:]}

    # 情形B：推动浪进行中（最后4个转折点，处于浪3或浪4）
    p4 = piv.tail(4).to_dict("records")
    kinds = [x["kind"] for x in p4]
    up = kinds[0] == "L"
    v = [x["price"] for x in p4]
    if up and kinds == ["L", "H", "L", "H"] and v[2] > v[0] and v[3] > v[1]:
        return {"state": "推动浪第4浪进行中（上涨）",
                "detail": "第三浪已创新高，等第四浪回撤确认",
                "invalid": float(p4[0]["price"]), "alts": [],
                "label_code": 4.0, "last_pivots": [p4[-1]]}
    if (not up) and kinds == ["H", "L", "H", "L"] and v[2] < v[0] and v[3] < v[1]:
        return {"state": "推动浪第4浪进行中（下跌）",
                "detail": "第三浪已创新低，等第四浪反弹确认",
                "invalid": float(p4[0]["price"]), "alts": [],
                "label_code": -4.0, "last_pivots": [p4[-1]]}

    # 情形C：只有一段推动 + 一段调整
    p3 = piv.tail(3).to_dict("records")
    if [x["kind"] for x in p3] in (["L", "H", "L"], ["H", "L", "H"]):
        return {"state": "结构不明", "detail": "仅一段推动+一段调整，浪级无法确定",
                "invalid": float(lp["price"]), "alts": [],
                "label_code": 0.0, "last_pivots": p3[-2:]}

    return {"state": "结构不明", "detail": "转折点不满足推动浪规则",
            "invalid": float(lp["price"]), "alts": [],
            "label_code": 0.0, "last_pivots": piv.tail(2).to_dict("records")}


# ===================================================================== 易经

TRIGRAM_NAME = {1: "乾", 2: "兑", 3: "离", 4: "震", 5: "巽", 6: "坎", 7: "艮", 8: "坤"}
TRIGRAM_SYM = {1: "☰", 2: "☱", 3: "☲", 4: "☳", 5: "☴", 6: "☵", 7: "☶", 8: "☷"}
# 自下而上的爻：1=阳，0=阴
TRIGRAM_BITS = {1: (1, 1, 1), 2: (1, 1, 0), 3: (1, 0, 1), 4: (1, 0, 0),
                5: (0, 1, 1), 6: (0, 1, 0), 7: (0, 0, 1), 8: (0, 0, 0)}

HEX_NAMES = {
    1: {1: "乾为天", 2: "天泽履", 3: "天火同人", 4: "天雷无妄", 5: "天风姤", 6: "天水讼", 7: "天山遯", 8: "天地否"},
    2: {1: "泽天夬", 2: "兑为泽", 3: "泽火革", 4: "泽雷随", 5: "泽风大过", 6: "泽水困", 7: "泽山咸", 8: "泽地萃"},
    3: {1: "火天大有", 2: "火泽睽", 3: "离为火", 4: "火雷噬嗑", 5: "火风鼎", 6: "火水未济", 7: "火山旅", 8: "火地晋"},
    4: {1: "雷天大壮", 2: "雷泽归妹", 3: "雷火丰", 4: "震为雷", 5: "雷风恒", 6: "雷水解", 7: "雷山小过", 8: "雷地豫"},
    5: {1: "风天小畜", 2: "风泽中孚", 3: "风火家人", 4: "风雷益", 5: "巽为风", 6: "风水涣", 7: "风山渐", 8: "风地观"},
    6: {1: "水天需", 2: "水泽节", 3: "水火既济", 4: "水雷屯", 5: "水风井", 6: "坎为水", 7: "水山蹇", 8: "水地比"},
    7: {1: "山天大畜", 2: "山泽损", 3: "山火贲", 4: "山雷颐", 5: "山风蛊", 6: "山水蒙", 7: "艮为山", 8: "山地剥"},
    8: {1: "地天泰", 2: "地泽临", 3: "地火明夷", 4: "地雷复", 5: "地风升", 6: "地水师", 7: "地山谦", 8: "坤为地"},
}


def hexagram(ts: pd.Timestamp) -> dict:
    """固定公历年月日时起卦（梅花易数先天数）。

    上卦 = (年支数 + 公历月 + 公历日) % 8，0 作 8
    下卦 = (年支数 + 公历月 + 公历日 + 时支数) % 8，0 作 8
    动爻 = 上式 % 6，0 作 6
    时支固定取 15:00 = 申时 = 9
    """
    year = ts.year
    branch = (year - 4) % 12 + 1          # 子=1 ... 亥=12
    month, day = ts.month, ts.day
    hour_branch = 9                        # 申时（15:00-16:59）
    s1 = branch + month + day
    s2 = s1 + hour_branch
    upper = s1 % 8 or 8
    lower = s2 % 8 or 8
    moving = s2 % 6 or 6

    bits = list(TRIGRAM_BITS[lower])
    changed = bits.copy()
    changed[moving - 1] = 1 - changed[moving - 1]
    val = changed[2] * 4 + changed[1] * 2 + changed[0]
    lower2 = 8 - val

    return {
        "date": ts.strftime("%Y-%m-%d"),
        "upper": TRIGRAM_NAME[upper],
        "lower": TRIGRAM_NAME[lower],
        "upper_sym": TRIGRAM_SYM[upper],
        "lower_sym": TRIGRAM_SYM[lower],
        "upper_num": upper, "lower_num": lower,
        "moving": moving,
        "name": HEX_NAMES[upper][lower],
        "changed_name": HEX_NAMES[upper][lower2],
        "seed": f"年支{branch}+月{month}+日{day}+时9={s2}",
    }
