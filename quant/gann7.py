"""江恩“7”数循环：从阶段高低点投影 7 的倍数交易日窗口。

规则来源：
  b6《江恩实战精华》第一章（五）：7 天及其倍数、7 周及其倍数都可以是市场重要转折点；
  b9《江恩趋势预测法》：1 周有 7 天，7×7 = 49 天，7 周经常预示重要转折点。

口径：
  * 锚点取最近 5 个 ZigZag 阶段转折点（5% 阈值，与 quant/wave.py 一致）；
  * 窗口中心 = 锚点后第 7k 个交易日，前后各 1 个交易日为窗口；
  * 只用截至当日已完成的 K 线，最后一个转折点标注“未确认”，不当作既成事实；
  * 历史检验用“窗口前后 3 日方向相反”作为命中，并与该标的的无条件基率对照，
    明确报告样本数与 p 值——不把时间窗口包装成有方向优势的指标。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import wave as wv

CYCLES = (7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77)
WINDOW = 1                 # 中心前后各 1 个交易日
PIVOT_PCT = 0.05
ANCHORS_N = 5
MIN_BARS = 70


def _date_at(dates, i):
    return str(pd.Timestamp(dates[i]))[:10]


def _binom_two_sided(k: int, n: int, p: float) -> float:
    """精确二项检验（双侧），口径接近 scipy.stats.binomtest。

    大样本时 math.comb(n, i) 会超出 float 范围（大整数乘 float 直接 Overflow），
    因此在 log 空间计算组合数与概率，最后做相对归一；比较与求和不受缩放影响。
    """
    if n <= 0 or not np.isfinite(p) or p <= 0 or p >= 1:
        return float("nan")
    log_p, log_q = math.log(p), math.log1p(-p)
    lg_n = math.lgamma(n + 1)
    logs = [lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * log_p + (n - i) * log_q for i in range(n + 1)]
    top = max(logs)
    probs = [math.exp(x - top) for x in logs]
    total = sum(probs)
    observed = probs[k]
    return float(min(1.0, sum(x for x in probs if x <= observed + 1e-15) / total))


def _base_rate(close: np.ndarray) -> tuple[int, int]:
    """任意一日前 3 日与后 3 日方向相反的比例（无条件基率）。"""
    hit = total = 0
    for t in range(3, len(close) - 3):
        pre = np.sign(close[t] - close[t - 3])
        post = np.sign(close[t + 3] - close[t])
        if pre == 0 or post == 0:
            continue
        total += 1
        hit += int(pre * post < 0)
    return hit, total


def _stats(df: pd.DataFrame, piv: list, cycles) -> dict:
    close = df["close"].to_numpy(float)
    n = len(close)
    by_cycle = []
    hits = samples = 0
    for c in cycles:
        ch = cn = 0
        for idx, _px, _kind in piv[:-1]:          # 最后一个转折点尚未确认，剔除
            ctr = idx + c
            if ctr - 4 < 0 or ctr + 5 >= n:
                continue
            pre = np.sign(close[ctr - 1] - close[ctr - 4])
            post = np.sign(close[ctr + 3] - close[ctr])
            if pre == 0 or post == 0:
                continue
            cn += 1
            ch += int(pre * post < 0)
        if cn:
            by_cycle.append({"cycle": c, "samples": cn, "hits": ch,
                             "hit_rate": round(ch / cn, 4)})
        hits += ch
        samples += cn

    base_hits, base_total = _base_rate(close)
    base = base_hits / base_total if base_total else float("nan")
    rate = hits / samples if samples else float("nan")
    p_value = _binom_two_sided(hits, samples, base) if samples else float("nan")

    if not samples:
        verdict = "样本不足，未做统计检验"
    elif not np.isfinite(p_value) or p_value > 0.05:
        verdict = "与随机基率无显著差异，不能据此认为窗口有方向优势"
    elif rate > base:
        verdict = "命中率高于随机基率（仍需样本外复核）"
    else:
        verdict = "命中率低于随机基率（不能作为反向信号使用）"

    return {
        "hits": hits, "samples": samples,
        "hit_rate": round(rate, 4) if np.isfinite(rate) else None,
        "base_rate": round(base, 4) if np.isfinite(base) else None,
        "p_value": round(p_value, 4) if np.isfinite(p_value) else None,
        "by_cycle": by_cycle,
        "verdict": verdict,
        "definition": "命中＝窗口中心前后 3 日方向相反；样本为锚点×周期，窗口高度重叠、不独立",
    }


def analyze(df: pd.DataFrame, pivot_pct: float = PIVOT_PCT) -> dict:
    """返回小周期（江恩 7 循环）分析。数据不合法时抛 ValueError。"""
    need = ("trade_date", "open", "high", "low", "close")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"日线缺少字段: {', '.join(missing)}")
    if len(df) < MIN_BARS:
        raise ValueError(f"日线仅 {len(df)} 根，至少需要 {MIN_BARS} 根才能做 7 循环分析")

    df = df.reset_index(drop=True)
    dates = df["trade_date"].to_numpy()
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    now = len(df) - 1

    piv = wv.zigzag(high, low, pivot_pct)
    if len(piv) < 3:
        raise ValueError(f"5% ZigZag 仅找到 {len(piv)} 个转折点，无法投影 7 循环")

    # ---------------- 锚点 ----------------
    anchors = []
    for j, (idx, px, kind) in enumerate(piv):
        idx = int(idx)
        confirmed = j < len(piv) - 1
        if kind == "H":
            invalidated = bool(high[idx + 1:].max() > px) if idx + 1 < now + 1 else False
            broke_at = idx + 1 + int(np.argmax(high[idx + 1:])) if invalidated else None
        else:
            invalidated = bool(low[idx + 1:].min() < px) if idx + 1 < now + 1 else False
            broke_at = idx + 1 + int(np.argmin(low[idx + 1:])) if invalidated else None
        anchors.append({
            "idx": idx, "date": _date_at(dates, idx), "kind": kind,
            "label": "阶段高点" if kind == "H" else "阶段低点",
            "price": round(float(px), 4), "confirmed": confirmed,
            "bars_ago": now - idx, "invalidated": invalidated,
            "invalidated_at": _date_at(dates, broke_at) if broke_at is not None else None,
        })

    # ---------------- 窗口投影 ----------------
    windows = []
    for a in anchors[-ANCHORS_N:]:
        for c in CYCLES:
            ctr = a["idx"] + c
            d = ctr - now
            if d > 45:
                continue
            if d < -10:
                continue
            if ctr > now:
                center_date = None
                lo_date = hi_date = None
            else:
                center_date = _date_at(dates, ctr)
                lo_date = _date_at(dates, max(0, ctr - WINDOW))
                hi_date = _date_at(dates, min(now, ctr + WINDOW))
            if a["invalidated"]:
                status = "已失效"
            elif d > WINDOW:
                status = "待验证"
            elif d >= -WINDOW:
                status = "窗口内"
            else:
                status = "已实现"
            windows.append({
                "anchor_date": a["date"], "anchor_kind": a["kind"],
                "anchor_label": a["label"], "anchor_price": a["price"],
                "anchor_confirmed": a["confirmed"], "anchor_invalidated": a["invalidated"],
                "cycle": c, "center_idx": ctr, "center_date": center_date,
                "lo_date": lo_date, "hi_date": hi_date,
                "days_to_center": d, "status": status,
            })
    windows.sort(key=lambda w: (w["center_idx"], w["anchor_date"]))

    valid = [w for w in windows if not w["anchor_invalidated"]]
    active = [w for w in valid if w["status"] == "窗口内"]
    future = [w for w in valid if w["days_to_center"] > WINDOW]
    active_anchors = {w["anchor_date"] for w in active}

    # ---------------- 当前判断 ----------------
    last = anchors[-1]
    bars_since = now - last["idx"]
    cycle_pos = bars_since % 7
    next_seven = (7 - cycle_pos) % 7

    if active:
        sides = {1 if w["anchor_kind"] == "L" else -1 for w in active}
        side = sides.pop() if len(sides) == 1 else 0
        action = "留意反弹" if side > 0 else "留意回调" if side < 0 else "观察"
        resonance = len(active_anchors) >= 2 or len(active) >= 2
        nearest = min(active, key=lambda w: abs(w["days_to_center"]))
        head = (f"处于江恩 7 循环窗口：{nearest['anchor_date']} "
                f"{nearest['anchor_label']} {nearest['anchor_price']} +{nearest['cycle']} 日"
                + ("（多窗口时间共振）" if resonance else ""))
        if side > 0:
            detail = (f"低点锚点后的时间窗口，价格未跌破 {nearest['anchor_price']}，"
                      f"留意阶段低点/反弹；若收盘跌破该价位则窗口失效。")
        elif side < 0:
            detail = (f"高点锚点后的时间窗口，价格未突破 {nearest['anchor_price']}，"
                      f"留意阶段高点/回调；若收盘突破该价位则窗口失效。")
        else:
            detail = "同时存在高点与低点锚点窗口，方向分歧，按观望处理。"
    elif future:
        side = 0
        action = "观察"
        nearest = min(future, key=lambda w: w["days_to_center"])
        d = nearest["days_to_center"]
        head = (f"当前不在窗口内；最近窗口为 {nearest['anchor_date']} "
                f"{nearest['anchor_label']} {nearest['anchor_price']} +{nearest['cycle']} 日，"
                f"中心在 {d} 个交易日后")
        detail = "窗口只是时间参考，需等价格在窗口附近给出确认。"
    else:
        side = 0
        action = "观察"
        nearest = None
        head = "未来 45 个交易日内没有有效的 7 循环窗口"
        detail = "等新的阶段转折点确认后重新投影。"

    return {
        "asof": _date_at(dates, now),
        "price": round(float(close[now]), 4),
        "bars": len(df),
        "pivot_pct": pivot_pct,
        "anchors": anchors[-ANCHORS_N:],
        "cycle": {
            "anchor_date": last["date"], "anchor_kind": last["kind"],
            "anchor_label": last["label"], "anchor_price": last["price"],
            "anchor_confirmed": last["confirmed"],
            "bars_since_anchor": bars_since,
            "position": cycle_pos, "next_node_in": next_seven,
        },
        "current": {
            "in_window": bool(active),
            "windows": [{k: w[k] for k in ("anchor_date", "anchor_kind", "anchor_label",
                                           "anchor_price", "cycle", "center_date",
                                           "days_to_center", "status")}
                        for w in active],
            "nearest": ({k: nearest[k] for k in ("anchor_date", "anchor_kind", "anchor_label",
                                                 "anchor_price", "cycle", "center_date",
                                                 "days_to_center", "status")}
                        if nearest else None),
            "side": side, "action": action, "head": head, "detail": detail,
        },
        "windows": windows,
        "stats": _stats(df, piv, CYCLES),
        "disclaimer": ("江恩 7 循环只给出时间窗口，不预测方向；上方统计已按本标的实测，"
                       "命中率与随机基率无显著差异时不得当作优势使用。"),
    }
