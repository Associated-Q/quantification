"""AI 读图研判：数字全部由代码计算，模型只做描述与研判。

分工：
- collect_facts() 只用本地 K 线与现有规则模块算数，不让模型碰数字；
- describe() 让模型用书里的形态名描述 facts，并校验它没有编造数字/形态；
- retrieve() 用描述里的形态词去 book 来源段落里做 BM25 检索；
- advise() 让模型结合描述与书内段落给出动作，引用必须落到 [n]。

所有结论按 (trade_date, symbol) 缓存进 DuckDB 的 ai_advice 表；同一天重复
请求直接返回缓存。模型返回非法 JSON 时抛 AIAnalysisError，不做静默降级。
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import date, datetime

import duckdb
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAG = os.path.join(ROOT, "_distill", "rag")
sys.path.insert(0, HERE)
sys.path.insert(0, RAG)

import data_source  # noqa: E402
import multi_tf as mtf  # noqa: E402
import realtime  # noqa: E402
import wave as wv  # noqa: E402
import book_rules as br  # noqa: E402
import search as S  # noqa: E402

DB = data_source.DB
TABLE = "ai_advice"

ACTIONS = ("加仓", "持有偏多", "观望", "减仓", "清仓")
PATTERN_KEYS = {"engulf", "evening_star", "morning_star", "hammer",
                "shooting_star", "doji", "gap"}
PASSAGE_MAX_CHARS = 1200
PASSAGE_TOP_K = 6

_INDEX = None
_INDEX_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()
_LAST_MODEL = None


class AIAnalysisError(Exception):
    """AI 链路中的显式错误；stage 标识失败环节。"""

    def __init__(self, message, stage="ai"):
        super().__init__(message)
        self.stage = stage


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _native(obj):
    """把 numpy / date 类型递归转成 JSON 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return str(obj)
    return obj


def _r(value, digits=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _round_or_none(value, digits=3):
    return _r(value, digits)


def _con():
    return duckdb.connect(DB)


def _ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            trade_date DATE,
            symbol VARCHAR,
            facts VARCHAR,
            description VARCHAR,
            advice VARCHAR,
            passages VARCHAR,
            model VARCHAR,
            created_at TIMESTAMP,
            PRIMARY KEY (trade_date, symbol))
    """)


def _book_title(book_stem):
    """与 web/app.py 的 _title 保持一致，但不反向依赖 web 层。"""
    t = book_stem.split("_", 1)[1] if "_" in book_stem else book_stem
    t = re.sub(r"^\d+_", "", t)
    t = t.replace("(Z-Library)", "").replace("（典藏版）", "")
    return t.strip().rstrip("(（ ").strip()


def get_index():
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = S.Index()
        return _INDEX


def get_last_model():
    return _LAST_MODEL


# ---------------------------------------------------------------------------
# 1) 代码计算的事实
# ---------------------------------------------------------------------------

def _pct_change(close, days):
    if len(close) <= days:
        return None
    base = float(close.iloc[-1 - days])
    if base == 0:
        return None
    return round((float(close.iloc[-1]) / base - 1) * 100, 2)


def _swing_levels(df, px, atr):
    """近 60 日摆动高低点聚类，给出最近支撑/阻力。

    摆动点来自 wave.zigzag；阈值随波动率自适应（1.0 ATR / 现价，限定 3%~6%），
    避免低波动 ETF 没有摆动点、高波动个股摆动点过多。收盘前 2 根内的未确认
    转折点不参与聚类，防止把当日最低价当成“支撑位”。
    """
    win = df.tail(60).reset_index(drop=True)
    pct = 0.05
    if px > 0:
        pct = min(0.06, max(0.03, 1.0 * (atr or 0.0) / px))
    piv = wv.clean_zigzag(wv.zigzag(win["high"], win["low"], pct=pct))
    points = [{"price": float(price), "bars_ago": len(win) - 1 - int(idx),
               "kind": kind}
              for idx, price, kind in piv if len(win) - 1 - int(idx) >= 2]

    notes = []
    if len(points) < 2:
        # 摆动点太少时退化为局部极值，保证支撑/阻力仍由真实价格产生
        for i in range(2, len(win) - 2):
            lo, hi = float(win["low"].iloc[i]), float(win["high"].iloc[i])
            if lo <= min(float(win["low"].iloc[i - 2:i + 3])):
                points.append({"price": lo, "bars_ago": len(win) - 1 - i, "kind": "L"})
            if hi >= max(float(win["high"].iloc[i - 2:i + 3])):
                points.append({"price": hi, "bars_ago": len(win) - 1 - i, "kind": "H"})
        notes.append("近60日摆动点不足，支撑/阻力改用局部极值")

    tol = max((atr or 0.0) * 0.2, px * 0.005)

    def cluster(ps):
        out = []
        for p in sorted(ps, key=lambda x: x["price"]):
            if out and abs(p["price"] - out[-1]["last"]) <= tol:
                cl = out[-1]
                cl["prices"].append(p["price"])
                cl["last"] = p["price"]
                cl["bars_ago"] = min(cl["bars_ago"], p["bars_ago"])
                cl["touches"] += 1
            else:
                out.append({"prices": [p["price"]], "last": p["price"],
                            "bars_ago": p["bars_ago"], "touches": 1})
        return [{"price": round(sum(c["prices"]) / len(c["prices"]), 3),
                 "bars_ago": int(c["bars_ago"]), "touches": int(c["touches"])}
                for c in out]

    lows = cluster([p for p in points if p["kind"] == "L"])
    highs = cluster([p for p in points if p["kind"] == "H"])

    # 支撑优先取现价下方的摆动低点；没有则取已跌破的摆动高点（转支撑）
    sup_c = [c for c in lows if c["price"] < px] or [c for c in highs if c["price"] < px]
    res_c = [c for c in highs if c["price"] > px] or [c for c in lows if c["price"] > px]

    if sup_c:
        support = dict(max(sup_c, key=lambda x: x["price"]), method="近60日摆动点聚类")
    else:
        low_i = int(win["low"].idxmin())
        support = {"price": round(float(win["low"].iloc[low_i]), 3),
                   "bars_ago": len(win) - 1 - low_i, "touches": 1,
                   "method": "近60日最低（现价下方无摆动点）"}
        notes.append("现价下方没有聚类支撑，改用近60日最低价")
    if res_c:
        resistance = dict(min(res_c, key=lambda x: x["price"]), method="近60日摆动点聚类")
    else:
        high_i = int(win["high"].idxmax())
        resistance = {"price": round(float(win["high"].iloc[high_i]), 3),
                      "bars_ago": len(win) - 1 - high_i, "touches": 1,
                      "method": "近60日最高（现价上方无摆动点）"}
        notes.append("现价上方没有聚类阻力，改用近60日最高价")

    def dist(price):
        if not atr or atr <= 0:
            return None
        return round(abs(px - price) / atr, 2)

    support["distance_atr"] = dist(support["price"])
    resistance["distance_atr"] = dist(resistance["price"])
    return {"support": support, "resistance": resistance, "notes": notes,
            "swing_pct": round(pct, 3)}


def _recent_gap(df, window=10):
    """最近 window 根 K 线内是否出现跳空；返回最近的一次。"""
    high, low = df["high"].astype(float), df["low"].astype(float)
    n = len(df)
    for i in range(n - 1, max(n - 1 - window, 0), -1):
        if low.iloc[i] > high.iloc[i - 1]:
            lower, upper = float(high.iloc[i - 1]), float(low.iloc[i])
            return {"has": True, "direction": "向上", "date": str(df["trade_date"].iloc[i])[:10],
                    "bars_ago": n - 1 - i, "lower": round(lower, 3),
                    "upper": round(upper, 3), "price": round((lower + upper) / 2, 3),
                    "size": round(upper - lower, 3), "window": window}
        if high.iloc[i] < low.iloc[i - 1]:
            lower, upper = float(high.iloc[i]), float(low.iloc[i - 1])
            return {"has": True, "direction": "向下", "date": str(df["trade_date"].iloc[i])[:10],
                    "bars_ago": n - 1 - i, "lower": round(lower, 3),
                    "upper": round(upper, 3), "price": round((lower + upper) / 2, 3),
                    "size": round(upper - lower, 3), "window": window}
    return {"has": False, "window": window}


def _double_pattern(df, atr):
    """用 ZigZag 最近三个已确认转折点识别双重顶/双重底。

    只使用收盘前 2 根以外的转折点：最后一根 K 线上的转折还未确认，
    拿它判断双顶/双底容易把当日冲高当成形态的第二个高点。
    """
    piv = [p for p in wv.clean_zigzag(wv.zigzag(df["high"], df["low"], pct=0.05))
           if len(df) - 1 - p[0] >= 2]
    if len(piv) < 3:
        return None
    p = piv[-3:]
    kinds = "".join(x[2] for x in p)
    close = float(df["close"].iloc[-1])
    tol = max((atr or 0.0) * 0.6, close * 0.015)
    bars_ago = len(df) - 1 - p[-1][0]
    if bars_ago > 60:
        return None
    if kinds == "HLH" and abs(p[0][1] - p[2][1]) <= tol:
        neckline = p[1][1]
        return {"name": "双重顶",
                "evidence": f"两个摆动高点 {p[0][1]:.3f} 与 {p[2][1]:.3f} "
                            f"相差 {abs(p[0][1] - p[2][1]):.3f}，中间低点 {neckline:.3f}；"
                            f"现价 {close:.3f}{'已跌破' if close < neckline else '尚未跌破'}颈线"}
    if kinds == "LHL" and abs(p[0][1] - p[2][1]) <= tol:
        neckline = p[1][1]
        return {"name": "双重底",
                "evidence": f"两个摆动低点 {p[0][1]:.3f} 与 {p[2][1]:.3f} "
                            f"相差 {abs(p[0][1] - p[2][1]):.3f}，中间高点 {neckline:.3f}；"
                            f"现价 {close:.3f}{'已突破' if close > neckline else '尚未突破'}颈线"}
    return None


def _trend_of(symbol, period, fast, slow, notes):
    try:
        tf = mtf.ensure(symbol, period)
    except Exception as exc:
        tf = realtime.period_history(symbol, mtf.TABLES[period])
        if tf is None or tf.empty:
            notes.append(f"{period} 数据获取失败：{type(exc).__name__}: {exc}")
            return None
    trend = mtf._trend(tf, fast, slow)
    if not trend:
        notes.append(f"{period} 数据不足 {slow + 2} 根，未做趋势判断")
        return None
    trend = dict(trend)
    trend["last_date"] = str(tf["trade_date"].iloc[-1])[:10]
    return trend


def collect_facts(symbol, price=None, prev_close=None):
    """返回全部由代码计算的事实；模型只能引用这里的数字。"""
    sym = data_source.to_symbol(symbol)
    df = realtime.history(sym, 400)
    if df is None or df.empty:
        raise AIAnalysisError(f"{sym} 无日线数据", "facts")
    df = df.reset_index(drop=True)
    if len(df) < 60:
        raise AIAnalysisError(f"{sym} 日线仅 {len(df)} 根，不足以计算事实", "facts")

    o, c, h, l, v = (df[x].astype(float) for x in
                     ("open", "close", "high", "low", "volume"))
    last_close = float(c.iloc[-1])
    px = float(price) if price is not None else last_close
    # px 来自实时报价时，日线最后一根就是昨收；取不到实时价（收盘后/周末）时
    # px 等于最后收盘，此时昨收应取倒数第二根，才能显示上一交易日的涨跌。
    if prev_close is not None:                 # 行情自带的昨收最准（前复权与实时价口径一致）
        prev_close = float(prev_close)
    else:
        prev_close = (float(c.iloc[-2]) if abs(px - last_close) < 1e-9 and len(c) >= 2
                      else last_close)
    atr = float(br._atr(df).iloc[-1])
    rsi = float(br._rsi(c).iloc[-1])
    vol_ma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else None

    notes = []
    mas = {}
    for n in (5, 20, 60, 250):
        if len(c) >= n:
            mas[f"ma{n}"] = _r(c.rolling(n).mean().iloc[-1], 3)
        else:
            mas[f"ma{n}"] = None
            notes.append(f"日线不足 {n} 根，MA{n} 为空")

    levels = _swing_levels(df, px, atr)
    notes.extend(levels.pop("notes"))
    swing_pct = levels.pop("swing_pct", None)

    zig = []
    for idx, pval, kind in wv.clean_zigzag(wv.zigzag(df["high"], df["low"], pct=0.05))[-6:]:
        zig.append({"date": str(df["trade_date"].iloc[int(idx)])[:10],
                    "price": round(float(pval), 3),
                    "type": "高" if kind == "H" else "低",
                    "bars_ago": len(df) - 1 - int(idx)})

    w = wv.analyze_wave(df)
    wave = {
        "ok": bool(w.get("ok", False)),
        "valid": bool(w.get("valid", False)),
        "position": w.get("position"),
        "direction": w.get("direction"),
        "pattern": w.get("pattern"),
        "failed": w.get("failed"),
        "alternatives": w.get("alternatives"),
        "ratios": w.get("ratios"),
        "rules": w.get("rules"),
        "why": w.get("why"),
        "bars_since_last_pivot": w.get("bars_since_last_pivot"),
        "wave_ends": w.get("wave_ends"),
        "threshold_pct": w.get("threshold_pct"),
        "pivots": w.get("pivots"),
    }

    chart_patterns = []
    for vote in br.analyze(df).get("votes", []):
        if vote.get("key") in PATTERN_KEYS:
            chart_patterns.append({"name": vote["rule"], "evidence": vote["evidence"],
                                   "key": vote["key"]})
    dp = _double_pattern(df, atr)
    if dp:
        chart_patterns.append(dp)

    week = _trend_of(sym, "week", 20, 60, notes)
    month = _trend_of(sym, "month", 12, 24, notes)

    recent_bars = []
    for row in df.tail(5).itertuples():
        recent_bars.append({"date": str(row.trade_date)[:10], "open": round(float(row.open), 3),
                            "high": round(float(row.high), 3), "low": round(float(row.low), 3),
                            "close": round(float(row.close), 3),
                            "volume": round(float(row.volume), 2)})

    facts = {
        "symbol": sym,
        "trade_date": str(df["trade_date"].iloc[-1])[:10],
        "current_price": round(px, 3),
        "last_close": round(last_close, 3),
        "prev_close": round(prev_close, 3),
        "change_pct": round((px / prev_close - 1) * 100, 2) if prev_close else None,
        "change_pct_5d": _pct_change(c, 5),
        "change_pct_20d": _pct_change(c, 20),
        "change_pct_60d": _pct_change(c, 60),
        **mas,
        "atr14": _r(atr, 4),
        "rsi14": _r(rsi, 1),
        "volume_ratio": _r(float(v.iloc[-1]) / vol_ma20, 2) if vol_ma20 else None,
        "high_20d": round(float(h.iloc[-20:].max()), 3),
        "low_20d": round(float(l.iloc[-20:].min()), 3),
        "support": levels["support"],
        "resistance": levels["resistance"],
        "swing_pct": swing_pct,
        "gap": _recent_gap(df),
        "zigzag_last6": zig,
        "wave": wave,
        "week_trend": week,
        "month_trend": month,
        "chart_patterns": chart_patterns,
        "recent_bars": recent_bars,
        "notes": notes,
    }
    return _native(facts)


# ---------------------------------------------------------------------------
# 2) 描述：一次 LLM 调用 + 数字/形态落地校验
# ---------------------------------------------------------------------------

DESCRIBE_PROMPT = """你是 A 股技术分析描述助手。下面是程序计算好的 facts（JSON）。
硬性要求：
1. 只能使用 facts 中已经出现的数字，不得自创、推算、改写或补充任何新数值。
   引用数字时保持原值（允许四舍五入到与原值相同的精度）。
2. summary、wave、support_state、momentum 都用中文。
3. patterns 只能从 facts.chart_patterns[].name 中选取，不得写 facts 里没有的形态；
   如果没有形态，输出空数组 []。
4. wave 只描述 facts.wave 的判断结果；support_state 只描述支撑/阻力及距离；
   momentum 只描述涨跌幅、RSI、量比等 facts 中已有的动量信息。
5. 只输出严格 JSON，不要 markdown 代码块，不要任何解释。结构必须是：
{"summary": "...", "patterns": ["..."], "wave": "...", "support_state": "...", "momentum": "..."}

facts：
{facts}"""

DESCRIBE_KEYS = ("summary", "patterns", "wave", "support_state", "momentum")


def _chat_json(prompt, stage):
    """调用模型并严格解析 JSON；任何失败都抛错，不返回空结果。"""
    global _LAST_MODEL
    S.load_env()
    base = os.environ.get("OCR_API_BASE")
    key = os.environ.get("OCR_API_KEY")
    model = os.environ.get("LLM_MODEL") or os.environ.get("OCR_MODEL")
    if not (base and key and model):
        raise AIAnalysisError("未配置模型接口（OCR_API_BASE/OCR_API_KEY/LLM_MODEL）", stage)
    try:
        resp = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "x-opencode-session": f"ai-{stage}"},
            json={"model": model, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise AIAnalysisError(f"模型请求失败：{type(exc).__name__}: {exc}", stage) from exc
    if resp.status_code != 200:
        raise AIAnalysisError(f"模型接口返回 {resp.status_code}：{resp.text[:300]}", stage)
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AIAnalysisError(f"模型响应结构异常：{resp.text[:300]}", stage) from exc

    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIAnalysisError(
            f"{stage} 返回非法 JSON：{exc}；原文前 300 字：{(content or '')[:300]}",
            stage) from exc
    if not isinstance(data, dict):
        raise AIAnalysisError(f"{stage} 返回的不是 JSON 对象：{(content or '')[:300]}", stage)
    _LAST_MODEL = model
    return data


def chat_json(prompt, stage="ai"):
    """公开的严格 JSON 调用，供其他模块复用；失败时抛 AIAnalysisError。"""
    return _chat_json(prompt, stage)


_ALIASES = {"吞没": "吞噬"}


def _norm(text):
    text = str(text)
    for src, dst in _ALIASES.items():
        text = text.replace(src, dst)
    return re.sub(r"[\s，。、；：（）()\[\]【】《》\"'“”‘’]", "", text)


def _numeric_tokens(text):
    return [(float(tok), len(tok.split(".")[1]) if "." in tok else 0)
            for tok in re.findall(r"\d+(?:\.\d+)?", str(text))]


def _ungrounded_numbers(description, facts):
    fact_text = json.dumps(facts, ensure_ascii=False, default=str)
    fact_nums = [x for x, _ in _numeric_tokens(fact_text)]
    text = " ".join([str(description.get(k, "")) for k in DESCRIBE_KEYS[:1] + DESCRIBE_KEYS[2:]]
                    + [str(x) for x in description.get("patterns") or []])
    missing = []
    for token, digits in _numeric_tokens(text):
        if any(abs(token - f) < 1e-9 for f in fact_nums):
            continue
        if any(round(f, digits) == round(token, digits) for f in fact_nums):
            continue
        missing.append(token)
    return sorted(set(missing))


def _allowed_patterns(facts):
    return [p.get("name", "") for p in facts.get("chart_patterns") or []]


def _pattern_grounded(pattern, allowed):
    p = _norm(pattern)
    if not p:
        return False
    for name in allowed:
        a = _norm(name)
        if a and (a in p or p in a):
            return True
    return False


def _validate_description(data, facts):
    missing = [k for k in DESCRIBE_KEYS if k not in data]
    if missing:
        raise AIAnalysisError(f"description 缺少字段：{', '.join(missing)}", "describe")
    if not isinstance(data["summary"], str) or not isinstance(data["wave"], str) \
            or not isinstance(data["support_state"], str) or not isinstance(data["momentum"], str):
        raise AIAnalysisError("description 的文本字段必须是字符串", "describe")
    if not isinstance(data["patterns"], list) or \
            not all(isinstance(x, str) for x in data["patterns"]):
        raise AIAnalysisError("description.patterns 必须是字符串数组", "describe")
    bad_numbers = _ungrounded_numbers(data, facts)
    allowed = _allowed_patterns(facts)
    bad_patterns = [p for p in data["patterns"] if not _pattern_grounded(p, allowed)]
    return bad_numbers, bad_patterns


def describe(facts):
    """生成中文描述；数字或形态无法在 facts 中落地时重试一次，再失败就抛错。"""
    prompt = DESCRIBE_PROMPT.replace("{facts}", json.dumps(facts, ensure_ascii=False, indent=1))
    data = _chat_json(prompt, "describe")
    bad_numbers, bad_patterns = _validate_description(data, facts)
    if bad_numbers or bad_patterns:
        fixes = []
        if bad_numbers:
            fixes.append("以下数字在 facts 中不存在，必须删除或改用 facts 中的原值："
                         + "、".join(str(x) for x in bad_numbers))
        if bad_patterns:
            fixes.append("以下形态在 facts.chart_patterns 中没有依据，必须删除或改用其中的名称："
                         + "、".join(bad_patterns))
        retry = prompt + "\n\n上一次输出不合格，请重写：\n" + "\n".join(fixes)
        data = _chat_json(retry, "describe_retry")
        bad_numbers, bad_patterns = _validate_description(data, facts)
        if bad_numbers or bad_patterns:
            raise AIAnalysisError(
                "description 未能落地到 facts："
                + (f"数字 {bad_numbers} " if bad_numbers else "")
                + (f"形态 {bad_patterns}" if bad_patterns else ""),
                "describe_grounding")
    return data


# ---------------------------------------------------------------------------
# 3) 检索 book 来源父块
# ---------------------------------------------------------------------------

def _format_passages(hits):
    out = []
    for h in hits:
        p = h["parent"]
        text = (p.get("text") or "").strip()
        if len(text) > PASSAGE_MAX_CHARS:
            text = text[:PASSAGE_MAX_CHARS].rstrip() + "…"
        out.append({"n": len(out) + 1,
                    "book_title": _book_title(p.get("book", "")) or p.get("title", ""),
                    "page_start": p.get("page_start"),
                    "page_end": p.get("page_end"),
                    "text": text})
        if len(out) >= PASSAGE_TOP_K:
            break
    return out


BOOK_FAMILY = {"b1": "蜡烛图", "b2": "期货", "b3": "道氏", "b4": "波浪", "b5": "波浪",
               "b6": "江恩", "b7": "凯利", "b8": "江恩", "b9": "江恩", "b10": "江恩"}
MAX_PER_FAMILY = 2      # 江恩系列共4本、占语料41%，不限配额会挤掉其它流派


def _diverse(hits, top_k=PASSAGE_TOP_K, max_per_book=2):
    """同一本书最多 max_per_book 段，同一流派最多 MAX_PER_FAMILY 段。

    形态为空时检索词退化为行情叙述，江恩类书籍因语料占比高会挤掉其它书；
    按流派限额后，蜡烛图/期货/道氏/波浪 才有机会进入引用。
    """
    picked, used, fam = [], {}, {}
    for h in hits:
        book = h["parent"].get("book", "")
        family = BOOK_FAMILY.get(book.split("_")[0], book)
        if used.get(book, 0) >= max_per_book or fam.get(family, 0) >= MAX_PER_FAMILY:
            continue
        used[book] = used.get(book, 0) + 1
        fam[family] = fam.get(family, 0) + 1
        picked.append(h)
        if len(picked) >= top_k:
            break
    return picked


def retrieve(description, query=None):
    """用 summary + patterns + wave 拼检索词，只在 book 来源里取 top 6。"""
    if query is None:
        parts = []
        for key in ("summary", "wave"):
            value = description.get(key)
            if value:
                parts.append(str(value))
        # 形态词是检索的核心，重复几次提高 BM25 词频，避免被 summary 里的
        # 数字和波浪术语稀释。
        patterns = [str(x) for x in description.get("patterns") or []]
        parts.extend(patterns * 3)
        query = " ".join(parts).strip()
    if not query:
        return []
    idx = get_index()
    # 先多召回再按来源过滤，避免 top_k 里全是 qa/comment
    hits = idx.search(query, PASSAGE_TOP_K * 8)
    books = [h for h in hits
             if h["parent"].get("source", "book") == "book"
             and h["parent"].get("page_start") is not None]
    return _format_passages(_diverse(books))


# ---------------------------------------------------------------------------
# 4) 研判：第二次 LLM 调用
# ---------------------------------------------------------------------------

ADVISE_PROMPT = """你是 A 股交易研判助手。请只基于下面的 facts、description 和书内段落给出动作。

硬性要求：
1. action 只能是：加仓 / 持有偏多 / 观望 / 减仓 / 清仓。
2. target_position 是 0 到 1 之间的数字。
3. rationale 必须用中文，并用 [1][2] 这样的编号引用书内段落；不得引用不存在的编号。
4. 如果书内段落与当前形态无关，必须返回 "观望"，并在 rationale 中写明
   "书中无对应依据"；不要为了给出动作而强行引用无关段落。
5. 不得自创 facts 之外的数字。
6. 只输出严格 JSON，不要 markdown 代码块，不要解释。结构必须是：
{"action": "...", "target_position": 0.0, "rationale": "...", "risks": "..."}

facts：
{facts}

description：
{description}

书内段落：
{passages}"""


def _pattern_cores(pattern):
    cores = ["吞噬", "黄昏星", "启明星", "锤子", "流星", "十字", "跳空",
             "双重顶", "双重底", "头肩"]
    found = [c for c in cores if c in _norm(pattern)]
    return found or [_norm(pattern)]


def _passages_relevant(description, passages):
    """形态词必须出现在前两段，或至少两段里，才算检索到相关依据。

    只在一段靠后的段落里偶然出现一次，不足以支撑一个方向性动作。
    """
    patterns = description.get("patterns") or []
    if not patterns:
        return True
    bodies = [_norm(p["text"]) for p in passages]
    for pattern in patterns:
        cores = _pattern_cores(pattern)
        hit_idx = [i for i, body in enumerate(bodies)
                   if any(core and core in body for core in cores)]
        if not hit_idx:
            continue
        if any(i < 2 for i in hit_idx) or len(hit_idx) >= 2:
            return True
    return False


def _no_basis_advice(reason):
    return {"action": "观望", "target_position": 0.3,
            "rationale": f"书中无对应依据：{reason}",
            "risks": "本判断仅表示未找到可引用的书中依据，不构成方向性建议；"
                     "后续若出现新的形态或放量突破关键价位，应重新分析。"}


def _validate_advice(data, passages):
    missing = [k for k in ("action", "target_position", "rationale", "risks") if k not in data]
    if missing:
        raise AIAnalysisError(f"advice 缺少字段：{', '.join(missing)}", "advise")
    action = data["action"]
    if action not in ACTIONS:
        raise AIAnalysisError(f"advice.action 非法：{action!r}", "advise")
    tp = data["target_position"]
    if isinstance(tp, bool) or not isinstance(tp, (int, float)) or not (0 <= float(tp) <= 1):
        raise AIAnalysisError(f"advice.target_position 非法：{tp!r}", "advise")
    if not isinstance(data["rationale"], str) or not isinstance(data["risks"], str):
        raise AIAnalysisError("advice.rationale/risks 必须是字符串", "advise")

    refs = [int(x) for x in re.findall(r"\[(\d+)\]", data["rationale"])]
    bad_refs = [r for r in refs if r < 1 or r > len(passages)]
    if bad_refs:
        raise AIAnalysisError(f"advice.rationale 引用了不存在的段落编号：{bad_refs}", "advise")
    if action != "观望" and not refs:
        raise AIAnalysisError("advice.action 不是观望，但 rationale 没有引用任何 [n] 段落", "advise")
    if action == "观望" and not refs and "书中无对应依据" not in data["rationale"]:
        raise AIAnalysisError("advice 为观望，但既未引用段落也未说明“书中无对应依据”", "advise")
    data["target_position"] = float(tp)
    return data


def advise(facts, description, passages, samples=3):
    """结合事实、描述、书内段落给出动作；无相关依据时强制观望。

    模型即使 temperature=0 也不是严格确定（实测同一输入会给出不同动作），
    因此并行采样多次、按多数票取值；没有任何动作过半时判为分歧并降级观望。
    """
    if not passages:
        return _no_basis_advice("未检索到与当前形态相关的书中段落。")
    if not _passages_relevant(description, passages):
        patterns = "、".join(description.get("patterns") or [])
        return _no_basis_advice(
            f"检索到的段落未提及当前形态（{patterns}），不据此给出方向性动作。")

    passage_text = "\n\n".join(
        f"[{p['n']}] 《{p['book_title']}》第{p.get('page_start')}-{p.get('page_end')}页\n{p['text']}"
        for p in passages)
    prompt = (ADVISE_PROMPT
              .replace("{facts}", json.dumps(facts, ensure_ascii=False, indent=1))
              .replace("{description}", json.dumps(description, ensure_ascii=False, indent=1))
              .replace("{passages}", passage_text))

    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, samples)) as ex:
        futures = [ex.submit(_chat_json, prompt, "advise") for _ in range(samples)]
        for fut in futures:
            try:
                results.append(_validate_advice(fut.result(), passages))
            except Exception as e:
                errors.append(e)
    if not results:
        raise AIAnalysisError("advise", f"{samples} 次研判全部失败：{errors[0]}")

    counts = Counter(r["action"] for r in results)
    top, n = counts.most_common(1)[0]
    votes = dict(counts)
    if n * 2 <= len(results):                    # 没有任何动作过半
        out = _no_basis_advice(
            f"{len(results)} 次独立研判结论不一致（{votes}），按观望处理，请人工判断。")
        out.update({"votes": votes, "samples": len(results)})
        return out

    chosen = next(r for r in results if r["action"] == top)
    chosen.update({"votes": votes, "samples": len(results)})
    return chosen


# ---------------------------------------------------------------------------
# 缓存 / 对外入口
# ---------------------------------------------------------------------------

def _load_cache(trade_date, symbol):
    con = _con()
    try:
        with _DB_LOCK:
            _ensure_table(con)
            row = con.execute(
                f"SELECT facts, description, advice, passages, model, created_at "
                f"FROM {TABLE} WHERE trade_date = ? AND symbol = ?",
                [trade_date, symbol]).fetchone()
    finally:
        con.close()
    if not row:
        return None
    try:
        return {"facts": json.loads(row[0]), "description": json.loads(row[1]),
                "advice": json.loads(row[2]), "passages": json.loads(row[3]),
                "model": row[4], "created_at": str(row[5])}
    except (TypeError, ValueError) as exc:
        raise AIAnalysisError(f"缓存记录损坏 {trade_date}/{symbol}: {exc}", "cache") from exc


def _save_cache(trade_date, symbol, facts, description, advice, passages, model):
    con = _con()
    try:
        with _DB_LOCK:
            _ensure_table(con)
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE}"
                "(trade_date, symbol, facts, description, advice, passages, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [trade_date, symbol,
                 json.dumps(facts, ensure_ascii=False),
                 json.dumps(description, ensure_ascii=False),
                 json.dumps(advice, ensure_ascii=False),
                 json.dumps(passages, ensure_ascii=False),
                 model, datetime.now()])
    finally:
        con.close()


def is_cached(trade_date, symbol):
    return _load_cache(trade_date, symbol) is not None


PRICE_DRIFT_THRESHOLD = 0.015     # 现价与研判时所用价格偏离超过 1.5% 即视为过期


def _load_latest_cache(symbol):
    """取该标的最近一条研判（不限日期），返回 (trade_date, 缓存内容) 或 (None, None)。

    缓存按「日线最后一根的日期」存放，与自然日不一定相同，因此不能用 today 直接查。
    """
    con = _con()
    try:
        with _DB_LOCK:
            _ensure_table(con)
            row = con.execute(
                f"SELECT trade_date, facts, description, advice, passages, model, created_at "
                f"FROM {TABLE} WHERE symbol = ? ORDER BY trade_date DESC LIMIT 1",
                [symbol]).fetchone()
    finally:
        con.close()
    if not row:
        return None, None
    try:
        return row[0], {"facts": json.loads(row[1]), "description": json.loads(row[2]),
                        "advice": json.loads(row[3]), "passages": json.loads(row[4]),
                        "model": row[5], "created_at": str(row[6])}
    except (TypeError, ValueError) as exc:
        raise AIAnalysisError(f"缓存记录损坏 {symbol}: {exc}", "cache") from exc


def latest(symbol, trade_date=None):
    """读最近一次研判，并核对现价是否已大幅偏离；偏离则标 stale，由调用方决定是否重算。"""
    sym = data_source.to_symbol(symbol)
    stored_date, cached = _load_latest_cache(sym)
    if cached is None:
        return None
    out = {"symbol": sym, "date": str(stored_date)[:10], "cached": True,
           "elapsed_ms": 0, "stale": False, **cached}
    try:
        quote = realtime.fetch_quotes([sym]).get(sym) or {}
        now = quote.get("price")
        used = (cached.get("facts") or {}).get("current_price")
        if now and used and float(used) > 0:
            drift = float(now) / float(used) - 1
            out.update({"current_price": now, "price_used": used,
                        "price_drift_pct": round(drift * 100, 2),
                        "stale": abs(drift) >= PRICE_DRIFT_THRESHOLD})
    except Exception as e:                       # 行情拿不到不影响返回缓存内容
        logger.warning("核对 %s 现价失败: %s", sym, e)
    return out


def analyze(symbol, price=None, force=False, search_query=None):
    """完整链路；同一天同标的有缓存则直接返回。"""
    t0 = time.perf_counter()
    sym = data_source.to_symbol(symbol)
    df = realtime.history(sym, 400)
    if df is None or df.empty:
        raise AIAnalysisError(f"{sym} 无日线数据", "facts")
    trade_date = str(df["trade_date"].iloc[-1])[:10]
    ref = realtime.market_last_date()
    if ref and trade_date < ref:
        # 刷新后仍落后于大盘，说明本地数据过期（停牌或拉取失败）。
        # 用旧数据研判会给出与盘面无关的结论，这里必须硬失败。
        raise AIAnalysisError(
            f"{sym} 日线只更新到 {trade_date}，大盘已到 {ref}；"
            f"本地数据过期，拒绝基于旧数据研判。", "facts")

    if not force:
        cached = _load_cache(trade_date, sym)
        if cached:
            return {"symbol": sym, "date": trade_date, "facts": cached["facts"],
                    "description": cached["description"], "passages": cached["passages"],
                    "advice": cached["advice"], "cached": True,
                    "created_at": cached.get("created_at"),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # 不传价格时用实时报价，而不是日线最后一根的收盘价：
    # 盘中（尤其涨停/大跌）用昨收会让研判完全脱离当前盘面。
    quote = {}
    if price is None:
        try:
            quote = realtime.fetch_quotes([sym]).get(sym) or {}
            price = quote.get("price")
        except Exception as e:
            logger.warning("取 %s 实时价失败，改用日线收盘价: %s", sym, e)

    facts = collect_facts(sym, price, quote.get("prev_close"))
    trade_date = facts["trade_date"]
    description = describe(facts)
    passages = retrieve(description, query=search_query)
    advice = advise(facts, description, passages)
    model = get_last_model() or ""

    now = datetime.now()
    try:
        _save_cache(trade_date, sym, facts, description, advice, passages, model)
    except Exception as exc:
        raise AIAnalysisError(f"写入 ai_advice 缓存失败：{type(exc).__name__}: {exc}",
                              "cache") from exc

    return {"symbol": sym, "date": trade_date, "facts": facts,
            "description": description, "passages": passages, "advice": advice,
            "cached": False, "created_at": now.isoformat(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


def list_advice(symbol=None, limit=200):
    con = _con()
    try:
        with _DB_LOCK:
            _ensure_table(con)
            if symbol:
                rows = con.execute(
                    f"SELECT trade_date, description, advice FROM {TABLE} "
                    "WHERE symbol = ? ORDER BY trade_date DESC LIMIT ?",
                    [symbol, limit]).fetchall()
            else:
                rows = con.execute(
                    f"SELECT trade_date, description, advice FROM {TABLE} "
                    "ORDER BY trade_date DESC, symbol LIMIT ?", [limit]).fetchall()
    finally:
        con.close()
    items = []
    for trade_date, desc_raw, advice_raw in rows:
        try:
            desc = json.loads(desc_raw)
            adv = json.loads(advice_raw)
        except (TypeError, ValueError) as exc:
            raise AIAnalysisError(f"缓存记录损坏 {trade_date}: {exc}", "cache") from exc
        items.append({"date": str(trade_date), "action": adv.get("action"),
                      "target_position": adv.get("target_position"),
                      "summary": desc.get("summary", "")})
    return items


def _refresh(symbol):
    df = data_source.fetch_daily(symbol, count=800)
    if df is None or df.empty:
        raise ValueError(f"{symbol} 拉取日线返回空")
    data_source.save(df)


def run_symbols(symbols, refresh=False):
    """对给定标的执行；已缓存则跳过。返回 {date, count, errors}。"""
    seen, errors, dates = set(), [], []
    count = 0
    for raw in symbols:
        try:
            sym = data_source.to_symbol(raw)
        except Exception as exc:
            errors.append({"symbol": str(raw), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if sym in seen:
            continue
        seen.add(sym)
        try:
            if refresh:
                _refresh(sym)
            result = analyze(sym)
            dates.append(result["date"])
            if not result["cached"]:
                count += 1
        except Exception as exc:
            errors.append({"symbol": sym, "error": f"{type(exc).__name__}: {exc}"})
    out_date = max(dates) if dates else datetime.now().strftime("%Y-%m-%d")
    return {"date": out_date, "count": count, "errors": errors}


def run_daily(refresh=False):
    """对全部分组自选去重执行。"""
    return run_symbols(realtime.items(), refresh=refresh)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 读图研判单标的测试")
    parser.add_argument("symbol")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = analyze(args.symbol, force=args.force)
    out = os.path.join(HERE, f"ai_probe_{result['date'].replace('-', '')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"symbol={result['symbol']} date={result['date']} cached={result['cached']} "
          f"elapsed_ms={result['elapsed_ms']} out={out}")
