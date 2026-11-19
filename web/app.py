"""技术分析知识库服务：本地 uvicorn 直接跑，也能原样部署到腾讯云。

本地启动：
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
RAG = ROOT / "_distill" / "rag"
sys.path.insert(0, str(RAG))
sys.path.insert(0, str(ROOT / "quant"))
import search as S  # noqa: E402
import realtime as rt  # noqa: E402
import book_rules as br  # noqa: E402
import gann7  # noqa: E402
import multi_tf as mtf  # noqa: E402
import wave as wv  # noqa: E402
import wave_ai as wai  # noqa: E402
import plan as plan_mod  # noqa: E402
import paper as pp  # noqa: E402
import position as pos  # noqa: E402
import snapshot as snap  # noqa: E402
import ai_analyst as ai  # noqa: E402

logger = logging.getLogger("technical-analysis")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def verify_api_key(request: Request,
                   x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """未配置 WEB_API_KEY 时保持原前端完全无鉴权；配置了则使用常数时间比较。"""
    expected = os.environ.get("WEB_API_KEY", "")
    if not expected or request.url.path in ("/", "/api/health"):
        return
    import hmac
    if x_api_key is None:
        raise HTTPException(status_code=401,
                            detail={"error": "missing_api_key", "message": "请求头缺少 X-API-Key"})
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401,
                            detail={"error": "invalid_api_key", "message": "X-API-Key 不匹配"})


app = FastAPI(title="技术分析知识库", version="0.1.0",
              dependencies=[Depends(verify_api_key)])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_state = {}
_lock = threading.Lock()
_started_at = time.time()
_metrics = {}
_metrics_lock = threading.Lock()
_cache = {}
_cache_lock = threading.Lock()
_cache_accesses = 0
_cache_hits = 0
_cache_misses = 0
_cache_evictions = 0
_CACHE_MAX_ITEMS = 256


def _cache_get(key):
    global _cache_accesses, _cache_hits, _cache_misses
    now = time.monotonic()
    with _cache_lock:
        _cache_accesses += 1
        entry = _cache.get(key)
        if entry is not None and entry[0] > now:
            _cache_hits += 1
            return entry[1]
        _cache_misses += 1
        if entry is not None:
            _cache.pop(key, None)
        # 惰性清理过期键（每 50 次访问或到达容量上限时）；容量上限由 _cache_set 维护。
        if _cache_accesses % 50 == 0 or len(_cache) >= _CACHE_MAX_ITEMS:
            for old_key in [k for k, v in _cache.items() if v[0] <= now]:
                _cache.pop(old_key, None)
        return None


def _cache_set(key, value, ttl):
    global _cache_evictions
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ITEMS and key not in _cache:
            _cache.pop(next(iter(_cache)))  # 按插入顺序淘汰最旧一项
            _cache_evictions += 1
        _cache[key] = (time.monotonic() + ttl, value)
    return value


def cached_quotes(symbols):
    syms = list(symbols)
    if not syms:
        return {}
    key = ("quotes", tuple(syms))
    cached = _cache_get(key)
    return cached if cached is not None else _cache_set(key, rt.fetch_quotes(syms), 3)


@app.middleware("http")
async def request_log(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        path = request.url.path
        route = request.scope.get("route")
        endpoint = getattr(route, "path", path)
        with _metrics_lock:
            if route is not None or endpoint in _metrics:
                stat = _metrics.setdefault(endpoint, {"count": 0, "total_ms": 0.0})
                stat["count"] += 1
                stat["total_ms"] += elapsed_ms
        logger.info("%s %s status=%s elapsed_ms=%.1f%s", request.method, path, status,
                    elapsed_ms, " SLOW" if elapsed_ms > 2000 else "")


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("未捕获异常 %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={
        "error": type(exc).__name__, "message": str(exc)[:500] or "服务端内部错误"})


@app.get("/api/metrics", dependencies=[Depends(verify_api_key)])
def metrics():
    with _metrics_lock:
        endpoints = {name: {"count": stat["count"],
                            "avg_ms": round(stat["total_ms"] / stat["count"], 2)}
                     for name, stat in _metrics.items() if stat["count"]}
    with _cache_lock:
        cache_stats = {"items": len(_cache), "max_items": _CACHE_MAX_ITEMS,
                       "hits": _cache_hits, "misses": _cache_misses,
                       "evictions": _cache_evictions}
    return {"started_at": datetime.fromtimestamp(_started_at, timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - _started_at, 2),
            "endpoints": endpoints, "cache_keys": cache_stats["items"],
            "cache": cache_stats}


def cite_index():
    with _lock:
        if "idx" not in _state:
            _state["idx"] = S.Index()
        return _state["idx"]


def cite(term, book_prefix=None):
    """把规则定位到具体书页；限定在规则所属的书里找，找不到就返回 None，不编造。"""
    hits = cite_index().search(term, 20)
    if book_prefix:
        hits = [h for h in hits if h["parent"]["book"].startswith(book_prefix)]
    if not hits:
        return None
    p = hits[0]["parent"]
    return {"book": _title(p["book"]), "page": f"{p['page_start']}-{p['page_end']}"}


def engine():
    with _lock:
        if "hybrid" not in _state:
            _state["hybrid"] = S.Hybrid(str(RAG / "embeddings.npy"))
            _state["books"] = _load_books()
        return _state["hybrid"]


def _title(book_stem):
    import re
    t = book_stem.split("_", 1)[1] if "_" in book_stem else book_stem
    t = re.sub(r"^\d+_", "", t)
    t = t.replace("(Z-Library)", "").replace("（典藏版）", "")
    return t.strip().rstrip("(（ ").strip()


def _load_books():
    out = []
    for p in sorted((ROOT / "_distill" / "out").glob("b*.md")):
        out.append({"id": p.name.split("_")[0], "title": _title(p.stem)})
    return out


class SearchReq(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    books: Optional[List[str]] = None
    sources: Optional[List[str]] = None


class AskReq(SearchReq):
    top_k: int = Field(5, ge=1, le=10)


ANSWER_PROMPT = """你是技术分析资料助手。请严格依据给出的事实与资料作答：
1. 行情数字只能来自【标的实时数据】，不得编造。可以基于给定的支撑、阻力、ATR 做加减推算，
   但必须标注「按 ATR 推算」并写出算式；
2. 回答必须按以下四段组织，不得省略任何一段：
   一、当前状态：均线排列、近期走势、量能、摆动指标；
   二、形态与浪型：识别到的形态、浪型是否成立、当前处于第几浪；
   三、书里怎么说：引用资料编号，逐条对应上面的形态与指标，说明该形态的标准做法；
   四、操作建议：明确给出「加仓 / 持有 / 观望 / 减仓 / 清仓」之一，并给出
      目标仓位百分比、参考价位（取自支撑或阻力，或按 ATR 推算）、失效条件
      （跌破哪个价位则本条建议不再成立）；
3. 每个要点后用 [编号] 标注出处，并保留资料上的来源标注；
4. 资料分三类：书（规则）、UP主问答与评论（个人经验）、社区讨论（情绪）。
   引用时要分清，不要把个人经验或社区情绪说成书本规则；
5. 数据或资料不足时，在对应段落直接说明，但第四段的操作建议仍必须给出，
   并注明依据强度（例如「仅据实时数据与 ATR 推断，无书内段落支持」）。

问题：{q}

【标的实时数据】
{data}

资料：
{docs}"""

METHOD_PROMPT = """你是技术分析资料助手。用户问的是方法或概念问题，请依据资料回答：
1. 先用一两句话直接回答核心问题；
2. 再分点说明具体做法、适用条件、失效情形；
3. 每个要点后用 [编号] 标注出处，保留书名与页码；
4. 资料分三类：书（规则）、UP主问答与评论（个人经验）、社区讨论（情绪），引用时不要混淆；
5. 资料里没有的不要编造；如果该方法在书中主要靠图示表达、文字不足以说明，直接指出这一限制。

问题：{q}

资料：
{docs}"""


def _build_prompt(req, data_text, docs):
    """有标的数据时用四段式个股分析，否则用方法讲解式，避免答非所问。"""
    if data_text:
        return (ANSWER_PROMPT.replace("{q}", req.query)
                              .replace("{data}", data_text)
                              .replace("{docs}", docs))
    return METHOD_PROMPT.replace("{q}", req.query).replace("{docs}", docs)


_DATA_CACHE = {}          # (symbol, 日期) -> (data, data_text, keywords)
_DATA_CACHE_MAX = 2000


def _query_data(query: str):
    """从问题里解析标的并取当日行情与指标。

    返回 (data, data_text, keywords)；keywords 是本次分析触发的规则名与浪型，
    用于拿「当前形态」去书里检索，而不是拿股票名字去搜。

    行情每次都取最新（内部有 3 秒缓存）；指标与形态按「标的+日期」缓存，
    因为日线数据一天只更新一次，重复计算没有意义。
    """
    info = rt.resolve_symbol(query)
    if not info:
        return None, "", []
    sym = info["symbol"]
    try:
        q = rt.fetch_quotes([sym]).get(sym, {})
        day = datetime.now().strftime("%Y-%m-%d")
        key = (sym, day)
        cached = _DATA_CACHE.get(key)
        if cached is None:
            facts = ai.collect_facts(sym, q.get("price"), q.get("prev_close"))
            rule = mtf.analyze(sym, q.get("price"), cite)
            keywords = [v["rule"] for v in (rule.get("votes") or []) if v.get("side")]
            wave = (rule.get("wave") or {})
            if wave.get("position"):
                keywords.append(wave["position"])
            data = {
                "symbol": sym, "code": info["code"], "name": q.get("name") or info["name"],
                "price": q.get("price"), "change_pct": q.get("change_pct"),
                "ma5": facts.get("ma5"), "ma20": facts.get("ma20"),
                "ma60": facts.get("ma60"), "ma250": facts.get("ma250"),
                "rsi14": facts.get("rsi14"), "atr14": facts.get("atr14"),
                "volume_ratio": facts.get("volume_ratio"),
                "support": facts.get("support"), "resistance": facts.get("resistance"),
                "action": rule.get("action"), "score": rule.get("score"),
            }
            text = _facts_text(data, facts, rule, keywords)
            if len(_DATA_CACHE) >= _DATA_CACHE_MAX:
                _DATA_CACHE.clear()
            _DATA_CACHE[key] = (data, text, keywords, facts)
            cached = _DATA_CACHE[key]
        data, text, keywords, _facts = cached
        data = dict(data)                       # 报价每次都刷新，指标沿用当天缓存
        data["price"] = q.get("price") or data["price"]
        data["change_pct"] = q.get("change_pct")
        return data, text, keywords
    except Exception as e:
        logger.warning("查询 %s 的行情失败: %s", sym, e)
        return None, "", []


def _facts_text(data, facts, rule, keywords):
    """把 collect_facts 的完整结果整理成模型可读的事实清单（数字全部来自这里）。"""
    def fmt(v, d=3):
        return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))

    def lvl(item, px):
        if not item or item.get("price") is None:
            return "—"
        return (f"{item['price']}（距现价 {item.get('distance_atr')} 倍 ATR，"
                f"{item.get('bars_ago')} 根K线前，{item.get('touches')} 次触碰）")

    px = data["price"]
    ma5, ma20, ma60, ma250 = (facts.get(k) for k in ("ma5", "ma20", "ma60", "ma250"))
    align = []
    if ma5 and ma20 and ma60:
        if px < ma5 < ma20 < ma60:
            align.append("价格低于 MA5/MA20/MA60，且 MA5<MA20<MA60（空头排列）")
        elif px > ma5 > ma20 > ma60:
            align.append("价格高于 MA5/MA20/MA60，且 MA5>MA20>MA60（多头排列）")
        else:
            align.append("均线呈纠缠状态，未形成明确排列")
    if ma250:
        align.append("价格在年线上方" if px > ma250 else "价格在年线下方")

    wave = facts.get("wave") or {}
    if wave.get("valid") and wave.get("position"):
        rules_txt = "；".join(f"{k}{'成立' if v else '不成立'}" for k, v in (wave.get("rules") or {}).items())
        wave_txt = (f"{wave.get('direction')}五浪结构，当前判断：{wave.get('position')}；"
                    f"{rules_txt}；浪1/3/5 终点 "
                    f"{(wave.get('wave_ends') or {}).get('浪1')}/"
                    f"{(wave.get('wave_ends') or {}).get('浪3')}/"
                    f"{(wave.get('wave_ends') or {}).get('浪5')}")
    else:
        rules_txt = "；".join(f"{k}{'成立' if v else '不成立'}"
                              for k, v in (wave.get("rules") or {}).items())
        wave_txt = f"不成立。{wave.get('why') or rules_txt or '无法划分浪型'}"

    pivots = facts.get("zigzag_last6") or []
    piv_txt = "、".join(f"{p.get('price')}({p.get('type')}，{p.get('bars_ago')}根前)"
                        for p in pivots if isinstance(p, dict)) if pivots else "—"
    wk, mo = facts.get("week_trend") or {}, facts.get("month_trend") or {}
    patterns = facts.get("chart_patterns") or []
    pat_txt = "；".join(f"{p.get('name')}（{p.get('evidence')}）"
                        for p in patterns if isinstance(p, dict)) if patterns else "无"
    gap = facts.get("gap") or {}
    gap_txt = ("无可识别的跳空" if not (isinstance(gap, dict) and gap.get("has"))
               else f"{gap.get('direction', '')}跳空：{gap.get('from')}~{gap.get('to')}")

    lines = [
        "【标的】%s（%s）" % (data["name"], data["symbol"]),
        "【行情】现价 %s　今日 %s%%　近5日 %s%%　近20日 %s%%　近60日 %s%%" % (
            px, fmt(data.get("change_pct"), 2), fmt(facts.get("change_pct_5d"), 2),
            fmt(facts.get("change_pct_20d"), 2), fmt(facts.get("change_pct_60d"), 2)),
        "【均线】MA5 %s　MA20 %s　MA60 %s　MA250 %s" % (ma5, ma20, ma60, ma250),
        "　　　　" + "；".join(align),
        "【摆动指标】RSI14 %s　量比 %s　ATR14 %s" % (
            fmt(facts.get("rsi14"), 1), fmt(facts.get("volume_ratio"), 2), fmt(facts.get("atr14"), 3)),
        "【区间】近20日高 %s　低 %s　区间幅度 %s%%" % (
            fmt(facts.get("high_20d")), fmt(facts.get("low_20d")),
            fmt((facts["high_20d"] - facts["low_20d"]) / facts["low_20d"] * 100, 1)
            if facts.get("high_20d") and facts.get("low_20d") else "—"),
        "【支撑】" + lvl(facts.get("support"), px),
        "【阻力】" + lvl(facts.get("resistance"), px),
        "【缺口】" + gap_txt,
        "【浪型】" + wave_txt,
        "【摆动点】最近6个转折：" + piv_txt,
        "【周线趋势】" + (f"{'向上' if wk.get('up') else '向下'} MA20 {wk.get('ma_fast')} / MA60 {wk.get('ma_slow')}"
                        if wk else "数据不足"),
        "【月线趋势】" + (f"{'向上' if mo.get('up') else '向下'} MA12 {mo.get('ma_fast')} / MA24 {mo.get('ma_slow')}"
                        if mo else "数据不足"),
        "【识别到的形态】" + pat_txt,
        "【规则评分】%s → %s（规则给出的目标仓位 %s%%）" % (
            data.get("score"), data.get("action"),
            round((rule.get("target_position") or 0) * 100)),
        "【触发规则明细】",
    ]
    for v in (rule.get("votes") or []):
        if not v.get("side"):
            continue
        side = "偏多" if v["side"] > 0 else "偏空"
        src = f"{v.get('book', '')} P{v.get('page', '')}" if v.get("page") else ""
        lines.append(f"　- {side} {v['rule']} {src}：{v.get('evidence', '')}")
    if keywords:
        lines.append("【检索关键词】" + "、".join(keywords))
    return "\n".join(lines)


def _effective_sources(req):
    """默认只查书。UP 主问答/评论与社区讨论是蒸馏素材，需显式勾选才参与回答。"""
    return req.sources if req.sources else ["book"]


def _select_hits(req, keywords):
    """有标的时：书籍按「触发的形态」逐条检索。

    拼成一句话检索会让「均线」这类高频词压过「双重顶」这类低频词，
    所以按形态逐条检索再合并。观点类材料默认不参与，除非用户显式勾选。
    """
    sources = _effective_sources(req)
    book_on = "book" in sources
    others = [s for s in sources if s != "book"]
    total = req.top_k + 2
    book_slots = total if book_on else 0

    def dedupe(seq, limit):
        seen, out = set(), []
        for h in seq:
            pid = h["parent"]["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append(h)
            if len(out) >= limit:
                break
        return out

    book_hits = []
    if book_slots:
        if keywords:
            for kw in keywords[:5]:
                book_hits += engine().search(kw, 2, req.books, ["book"])
        else:
            book_hits += engine().search(req.query, book_slots, req.books, ["book"])
    books = dedupe(book_hits, book_slots)

    opinion = []
    if others:
        opinion = dedupe(engine().search(req.query, 3, req.books, others), 3)
    return books + opinion


def _label(h):
    p = h["parent"]
    src = p.get("source", "book")
    date = p.get("date", "")
    if src == "book":
        return f"《{_title(p.get('book', ''))}》第{p.get('page_start')}-{p.get('page_end')}页"
    if src == "qa":
        return f"UP主充电问答（{date}）"
    if src == "comment":
        return f"UP主评论（{date}）"
    return f"社区讨论（{date}，非权威）"


@app.get("/api/health", dependencies=[])
def health():
    return {"ok": True, "loaded": "hybrid" in _state}


@app.get("/api/books")
def books():
    with _lock:
        if "books" not in _state:
            _state["books"] = _load_books()
        return _state["books"]


@app.on_event("startup")
def _warm_index():
    """后台预热检索模型，避免用户第一次搜索等十几秒。"""
    def _load():
        try:
            cite_index()                 # BM25 索引，构建约 7 秒，自选页首次打开要用
            engine()                     # 向量模型
            logger.info("检索模型预热完成")
        except Exception as e:                      # 预热失败不影响服务启动
            logger.warning("检索模型预热失败: %s", e)

    threading.Thread(target=_load, daemon=True).start()


@app.post("/api/search")
def search(req: SearchReq):
    t0 = time.time()
    hits = engine().search(req.query, req.top_k, req.books, req.sources)
    return {"query": req.query, "latency_ms": round((time.time() - t0) * 1000),
            "results": [{"book": h["parent"].get("book", ""),
                         "book_title": _title(h["parent"].get("book", "")) or h["parent"].get("title", ""),
                         "book_id": h["parent"].get("book", "").split("_")[0],
                         "source": h["parent"].get("source", "book"),
                         "author": h["parent"].get("author", ""),
                         "date": h["parent"].get("date", ""),
                         "page_start": h["parent"].get("page_start"),
                         "page_end": h["parent"].get("page_end"),
                         "score": round(h["score"], 5),
                         "text": h["parent"]["text"],
                         "matched": h["child"]["text"][:120]} for h in hits]}


@app.post("/api/ask")
def ask(req: AskReq):
    t0 = time.time()
    data, data_text, keywords = _query_data(req.query)
    hits = (_select_hits(req, keywords) if data
            else engine().search(req.query, req.top_k, req.books,
                                 _effective_sources(req)))
    if not hits and not data:
        return {"answer": "问题中没有识别到具体标的，资料里也没有相关内容。",
                "sources": [], "data": None,
                "latency_ms": round((time.time() - t0) * 1000)}
    docs = ("\n\n".join(f"[{i}] {_label(h)}\n{h['parent']['text']}"
                        for i, h in enumerate(hits, 1))
            if hits else "（本次未检索到相关资料，只能依据标的实时数据作答）")
    S.load_env()
    base, key = os.environ.get("OCR_API_BASE"), os.environ.get("OCR_API_KEY")
    model = os.environ.get("LLM_MODEL") or os.environ.get("OCR_MODEL")
    if not (base and key and model):
        return {"answer": "服务端未配置模型接口（OCR_API_BASE/KEY/MODEL），仅返回检索结果。",
                "sources": _sources(hits), "data": data,
                "latency_ms": round((time.time() - t0) * 1000)}
    prompt = _build_prompt(req, data_text, docs)
    try:
        r = requests.post(f"{base.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {key}",
                                   "x-opencode-session": "rag-ask"},
                          json={"model": model, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=180)
    except requests.Timeout:
        return {"answer": "模型响应超时（180 秒），请重试或改用流式接口 /api/ask/stream。",
                "sources": _sources(hits), "data": data,
                "latency_ms": round((time.time() - t0) * 1000)}
    except requests.RequestException as e:
        return {"answer": f"模型接口请求失败：{type(e).__name__}",
                "sources": _sources(hits), "data": data,
                "latency_ms": round((time.time() - t0) * 1000)}
    if r.status_code != 200:
        return {"answer": f"模型接口返回 {r.status_code}：{r.text[:200]}",
                "sources": _sources(hits), "data": data,
                "latency_ms": round((time.time() - t0) * 1000)}
    answer = r.json()["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": _sources(hits), "data": data,
            "latency_ms": round((time.time() - t0) * 1000)}


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/api/ask/stream")
def ask_stream(req: AskReq):
    """边生成边推送：meta（标的实时数据与出处）→ delta（答案片段）→ done。"""
    t0 = time.time()
    data, data_text, keywords = _query_data(req.query)
    hits = (_select_hits(req, keywords) if data
            else engine().search(req.query, req.top_k, req.books,
                                 _effective_sources(req)))
    docs = ("\n\n".join(f"[{i}] {_label(h)}\n{h['parent']['text']}"
                        for i, h in enumerate(hits, 1))
            if hits else "（本次未检索到相关资料，只能依据标的实时数据作答）")
    sources = _sources(hits)

    def gen():
        yield _sse({"type": "meta", "data": data, "sources": sources,
                    "latency_ms": round((time.time() - t0) * 1000)})
        S.load_env()
        base, key = os.environ.get("OCR_API_BASE"), os.environ.get("OCR_API_KEY")
        model = os.environ.get("LLM_MODEL") or os.environ.get("OCR_MODEL")
        if not (base and key and model):
            yield _sse({"type": "error", "message": "服务端未配置模型接口"})
            return
        prompt = _build_prompt(req, data_text, docs)
        try:
            with requests.post(
                    f"{base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}",
                             "x-opencode-session": "rag-ask",
                             "Content-Type": "application/json"},
                    json={"model": model, "temperature": 0, "stream": True,
                          "messages": [{"role": "user", "content": prompt}]},
                    stream=True, timeout=(10, 180)) as r:
                if r.status_code != 200:
                    yield _sse({"type": "error",
                                "message": f"模型接口返回 {r.status_code}"})
                    return
                r.encoding = "utf-8"
                got = False
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    msg = choices[0].get("message") or {}
                    delta = (choices[0].get("delta") or {}).get("content") or msg.get("content")
                    if delta:
                        got = True
                        yield _sse({"type": "delta", "t": delta})
                if not got:
                    yield _sse({"type": "error", "message": "模型没有返回任何内容"})
                    return
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"[:200]})
            return
        yield _sse({"type": "done", "latency_ms": round((time.time() - t0) * 1000)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sources(hits):
    return [{"n": i, "book": h["parent"].get("book", ""),
             "book_title": _title(h["parent"].get("book", "")) or h["parent"].get("title", ""),
             "source": h["parent"].get("source", "book"),
             "author": h["parent"].get("author", ""),
             "date": h["parent"].get("date", ""),
             "page_start": h["parent"].get("page_start"),
             "page_end": h["parent"].get("page_end"), "text": h["parent"]["text"]}
            for i, h in enumerate(hits, 1)]


class WatchReq(BaseModel):
    code: str = Field(..., min_length=6, max_length=8, pattern=r"^(?:\d{6}|(?:sh|sz|bj)\d{6})$")
    group_id: int = 1


BACKTEST_NOTE = ("书内规则已做样本外回测（2025-07~2026-10，13 只 ETF，固定 50% 仓位）："
                 "年化 +3.7%、最大回撤 -12.6%；同期买入持有 +7.3%、回撤 -34.2%。"
                 "收益跑输躺平，但回撤约为其三成——它是防守型规则，不是收益引擎。")


def _parse_symbol(symbol):
    """校验并规范化证券代码；非法时抛 ValueError，由调用方决定 HTTP 语义。"""
    import data_source as ds

    value = symbol.strip()
    if not value:
        raise ValueError("证券代码不能为空")
    if value[:2] in ("sh", "sz", "bj"):
        if len(value) != 8 or not value[2:].isdigit():
            raise ValueError(f"无效证券代码: {symbol}")
        return value
    if not value.isdigit() or len(value) != 6:
        raise ValueError(f"无效证券代码: {symbol}")
    return ds.to_symbol(value)


def _normalize_symbol(symbol):
    """路径参数用：非法代码返回 422 而不是 500。"""
    try:
        return _parse_symbol(symbol)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _empty_analysis():
    return {"action": "数据不足", "score": 0, "target_position": None,
            "votes": [], "skipped": [], "metrics": {}, "tf": {},
            "week": {}, "month": {}, "notes": []}


def _watch_items(group_id=None):
    syms = rt.items(group_id)
    if not syms:
        return [], True
    try:
        quotes = cached_quotes(syms)
        quote_ok = all(s in quotes for s in syms)
    except Exception as e:
        quotes, quote_ok = {}, False
        logger.warning("quote fetch failed: %s", e)
    out = []
    for s in syms:
        q = quotes.get(s) or {}
        item = {"symbol": s, "code": q.get("code", s[2:]), "name": q.get("name", s),
                "price": q.get("price"), "change_pct": q.get("change_pct"),
                "change": q.get("change"), "high": q.get("high"), "low": q.get("low"),
                "volume": q.get("volume"), "error": None if s in quotes else "暂无行情"}
        try:
            item.update(mtf.analyze(s, q.get("price"), cite))
        except Exception as e:
            item.update(_empty_analysis())
            item["action"] = "数据异常"
            item["error"] = f"{type(e).__name__}: {e}"
        out.append(item)
    return out, quote_ok


@app.get("/api/watchlist")
def watch_list(group_id: int | None = None):
    if not rt.items(group_id):
        return {"items": [], "note": BACKTEST_NOTE, "group_id": group_id}
    items, quote_ok = _watch_items(group_id)
    return {"items": items, "note": BACKTEST_NOTE, "quote_ok": quote_ok,
            "group_id": group_id}


@app.post("/api/watchlist")
def watch_add(req: WatchReq):
    try:
        sym = rt.add(_parse_symbol(req.code), req.group_id)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "symbol": sym, "group_id": req.group_id}


@app.delete("/api/watchlist/{code}")
def watch_del(code: str, group_id: int | None = None):
    try:
        sym = rt.remove(_normalize_symbol(code), group_id)
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "symbol": sym}


class GroupReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)


@app.get("/api/groups")
def group_list():
    return {"items": rt.groups()}


@app.post("/api/groups")
def group_add(req: GroupReq):
    try:
        return {"ok": True, **rt.add_group(req.name)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/groups/{gid}")
def group_rename(gid: int, req: GroupReq):
    try:
        return {"ok": True, **rt.rename_group(gid, req.name)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/groups/{gid}")
def group_del(gid: int):
    try:
        rt.remove_group(gid)
        return {"ok": True, "id": gid}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------
# 持仓 / 账户资金 / 每日建议快照
# ---------------------------------------------------------------------------

class PositionReq(BaseModel):
    code: str = Field(..., min_length=6, max_length=8)
    shares: int
    cost: float


class AccountReq(BaseModel):
    cash: float


def _position_suggestion(action, delta_lots):
    """可执行动作只由「目标股数 − 实际持股」决定。

    action 描述的是规则对走势的判断，与仓位差额是两个不同的问题：
    例如规则转弱（减仓）但你实际仓位低于目标仓位时，仍应加仓补足。
    两者方向相反时一并标记出来，交给界面提示，不替用户裁决。
    """
    if delta_lots is None:
        return "暂无建议", False
    conflict = ((action in ("加仓", "持有偏多") and delta_lots < 0) or
                (action in ("减仓", "清仓") and delta_lots > 0))
    if action == "清仓" and delta_lots < 0:
        return f"清仓，减 {-delta_lots} 手", conflict
    if delta_lots > 0:
        return f"加 {delta_lots} 手", conflict
    if delta_lots < 0:
        return f"减 {-delta_lots} 手", conflict
    return "不动", conflict


@app.get("/api/positions")
def positions_list():
    cash = pos.get_cash()
    items = pos.list_positions()
    equity = cash + sum(i["market_value"] for i in items
                        if i.get("market_value") is not None)
    out = []
    for it in items:
        price = it.get("price")
        action = target_position = None
        if price:
            try:
                analysis = mtf.analyze(it["symbol"], price, cite)
                action = analysis.get("action")
                target_position = analysis.get("target_position")
            except Exception as e:
                logger.warning("持仓分析失败 %s: %s", it["symbol"], e)
        target_shares = delta_lots = None
        if price and target_position is not None:
            target_shares = int(round(target_position * equity / price / 100)) * 100
            delta_lots = int(round((target_shares - it["shares"]) / 100))
        sug, conflict = _position_suggestion(action, delta_lots)
        out.append({
            "symbol": it["symbol"], "code": it["code"], "name": it["name"],
            "shares": it["shares"], "cost": it["cost"], "price": price,
            "market_value": it.get("market_value"), "pnl": it.get("pnl"),
            "pnl_pct": it.get("pnl_pct"), "target_position": target_position,
            "target_shares": target_shares, "delta_lots": delta_lots,
            "rule_action": action, "suggestion": sug, "action_conflict": conflict,
        })
    return {"cash": round(cash, 2), "equity": round(equity, 2), "items": out}


@app.post("/api/positions")
def positions_add(req: PositionReq):
    try:
        sym = _parse_symbol(req.code)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    try:
        quotes = cached_quotes([sym])
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"行情接口不可用：{type(e).__name__}: {e}") from e
    if sym not in quotes:
        raise HTTPException(status_code=422,
                            detail=f"无法识别证券代码 {req.code}（无实时行情）")
    try:
        sym = pos.upsert_position(sym, req.shares, req.cost)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("保存持仓失败 %s", sym)
        raise HTTPException(status_code=500,
                            detail=f"{type(e).__name__}: {e}") from e
    return {"ok": True, "symbol": sym}


@app.delete("/api/positions/{symbol}")
def positions_del(symbol: str):
    sym = _normalize_symbol(symbol)
    try:
        sym = pos.remove_position(sym)
    except Exception as e:
        logger.exception("删除持仓失败 %s", sym)
        raise HTTPException(status_code=500,
                            detail=f"{type(e).__name__}: {e}") from e
    return {"ok": True, "symbol": sym}


@app.post("/api/account")
def account_set(req: AccountReq):
    try:
        cash = pos.set_cash(req.cash)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"ok": True, "cash": cash}


@app.get("/api/snapshots")
def snapshots_list(symbol: Optional[str] = None):
    sym = _normalize_symbol(symbol) if symbol else None
    return {"items": snap.list_snapshots(sym)}


@app.post("/api/snapshot/run")
def snapshot_run():
    result = snap.run_snapshot()
    out = {"date": result["date"], "count": result["count"]}
    if result.get("errors"):
        out["errors"] = result["errors"]
    return out


@app.get("/api/snapshot/eval")
def snapshot_eval():
    return {"items": snap.eval_snapshot()}


# ---------------------------------------------------------------------------
# 模拟盘
# ---------------------------------------------------------------------------

class PaperResetReq(BaseModel):
    initial: float = 1_000_000


@app.get("/api/paper")
def paper_state():
    return pp.get_state()


@app.post("/api/paper/reset")
def paper_reset(req: PaperResetReq):
    try:
        return pp.reset(req.initial)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/paper/rebalance")
def paper_rebalance():
    return pp.rebalance()


@app.get("/api/paper/trades")
def paper_trades():
    items = pp.list_trades()
    syms = sorted({it["symbol"] for it in items})
    if syms:
        try:
            quotes = cached_quotes(syms)
        except Exception as e:                 # noqa: BLE001
            logger.warning("成交记录补名称失败: %s", e)
            quotes = {}
        for it in items:
            q = quotes.get(it["symbol"]) or {}
            raw = it["symbol"]
            code = raw[2:] if raw[:2] in ("sh", "sz", "bj") else raw
            it["code"] = q.get("code") or code
            it["name"] = q.get("name") or ""
    return {"items": items}



def _validated_bars(df, symbol, period, tcol):
    """空数据和缺字段显式报错，避免后续缺列异常或生成伪 K 线。"""
    if df is None or df.empty:
        raise HTTPException(status_code=404,
                            detail=f"{symbol} 无 {period} K 线数据")
    columns = (tcol, "open", "high", "low", "close", "volume")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{symbol} {period} K 线缺少字段: {', '.join(missing)}")
    return df


def _wave_overlay(df, result, tcol):
    """把浪型结果转成图表坐标、多级别浪组与斐波那契位。

    - groups 是小/中/大三档扫描出的全部有效五浪，用于在整段 K 线上画多组浪；
      它只服务画图，不参与评分（评分仍走 analyze_wave 的单结构口径）。
    - 当前结构未通过校验时，candidate 给出候选数法，前端用灰色虚线标出，
      不冒充有效结构。
    - 斐波那契位优先取当前五浪全程，其次取最近一组有效浪，最后退回最近波段，
      并用 source/label 标明来源。
    """
    times = [str(x) for x in df[tcol]]

    def point(p):
        i = int(p[0])
        if i < 0 or i >= len(times):
            raise ValueError(f"浪型转折点索引 {i} 超出 K 线范围 {len(times)}")
        return {"i": i, "price": round(float(p[1]), 3), "t": times[i]}

    def series(pivots):
        return [point(p) for p in pivots]

    groups = []
    for g in wv.wave_groups(df):
        groups.append({
            "degree": g["degree"], "degree_name": g["degree_name"],
            "threshold_pct": g["threshold_pct"], "up": g["up"],
            "pattern": g["pattern"], "failed": g["failed"],
            "span_bars": g["span_bars"], "pivots": series(g["pivots"]),
        })

    current = series(result.get("pivots") or [])
    valid = bool(result.get("valid")) and len(current) == 6
    if valid:
        # 当前结构是评分用的口径，必须出现在图上；若贪心分段没选中它，补一组。
        key = [p["i"] for p in current]
        if not any([p["i"] for p in g["pivots"]] == key for g in groups):
            ref = min((g["threshold_pct"] for g in groups),
                      default=result.get("threshold_pct") or 0.05)
            ratio = (result.get("threshold_pct") or ref) / ref if ref else 1.0
            level = 0 if ratio < 1.3 else (1 if ratio < 2.0 else 2)
            groups.append({
                "degree": level, "degree_name": wv.DEGREE_NAMES[level],
                "threshold_pct": result.get("threshold_pct") or ref,
                "up": result.get("direction") == "上升",
                "pattern": result.get("pattern") or "",
                "failed": bool(result.get("failed")), "span_bars": None,
                "pivots": current, "current": True,
            })
    for g in groups:
        if valid and [p["i"] for p in g["pivots"]] == [p["i"] for p in current]:
            g["current"] = True

    degree_map = {}
    for g in groups:
        item = degree_map.setdefault(g["degree"], {
            "level": g["degree"], "name": g["degree_name"],
            "threshold_pct": g["threshold_pct"], "count": 0})
        item["count"] += 1

    overlay = {
        "valid": bool(result.get("valid")),
        "why": result.get("why") or "",
        "direction": result.get("direction"),
        "pattern": result.get("pattern"),
        "position": result.get("position"),
        "groups": groups,
        "degrees": [degree_map[k] for k in sorted(degree_map)],
        "candidate": [] if valid else current,
        "fib": None,
    }

    if valid:
        degree = next((g["degree_name"] for g in groups
                       if [p["i"] for p in g["pivots"]] == [p["i"] for p in current]), None)
        overlay["fib"] = {
            "source": "wave", "degree": degree,
            "label": f"五浪全程（{degree}级别）" if degree else "五浪全程",
            "start": current[0], "end": current[-1],
            "levels": wv.fib_levels(current[0]["price"], current[-1]["price"]),
        }
        return overlay

    latest = max(groups, key=lambda g: g["pivots"][-1]["i"], default=None)
    if latest:
        overlay["fib"] = {
            "source": "group", "degree": latest["degree_name"],
            "label": f"最近一组{latest['degree_name']}级别五浪",
            "start": latest["pivots"][0], "end": latest["pivots"][-1],
            "levels": wv.fib_levels(latest["pivots"][0]["price"],
                                    latest["pivots"][-1]["price"]),
        }
        return overlay

    swing = wv.last_swing(df)
    if swing:
        a, b, th = swing
        overlay["fib"] = {"source": "swing", "degree": None, "label": "最近波段",
                          "start": point(a), "end": point(b),
                          "threshold_pct": th,
                          "levels": wv.fib_levels(a[1], b[1])}
    return overlay


@app.get("/api/detail/{symbol}")
def detail(symbol: str):
    import pandas as pd

    sym = _normalize_symbol(symbol)
    q = {}
    quote_error = None
    try:
        q = cached_quotes([sym]).get(sym) or {}
        if not q:
            quote_error = "暂无行情"
    except Exception as e:
        quote_error = f"{type(e).__name__}: {e}"
        logger.warning("quote failed for %s: %s", sym, e)
    df = _validated_bars(rt.history(sym, 400), sym, "day", "trade_date")
    try:
        analysis = mtf.analyze(sym, q.get("price"), cite)
        analysis_error = None
    except Exception as e:
        analysis, analysis_error = _empty_analysis(), f"{type(e).__name__}: {e}"
        logger.exception("analysis failed for %s", sym)

    n = 120
    sub = df.tail(n).reset_index(drop=True)
    ma20 = df["close"].rolling(20).mean().tail(n).reset_index(drop=True)
    ma60 = df["close"].rolling(60).mean().tail(n).reset_index(drop=True)
    return {
        "symbol": sym, "name": q.get("name", sym), "quote": q, "analysis": analysis,
        "quote_error": quote_error, "analysis_error": analysis_error,
        "kline": [{"d": str(r.trade_date)[:10], "o": float(r.open), "h": float(r.high),
                   "l": float(r.low), "c": float(r.close), "v": float(r.volume)}
                  for r in sub.itertuples()],
        "ma20": [None if pd.isna(x) else round(float(x), 3) for x in ma20],
        "ma60": [None if pd.isna(x) else round(float(x), 3) for x in ma60],
    }


@app.get("/api/detail/{symbol}/gann7")
def detail_gann7(symbol: str):
    """小周期：江恩 7 数循环（仅用日线已完成 K 线）。"""
    sym = _normalize_symbol(symbol)
    df = _validated_bars(rt.history(sym, 400), sym, "day", "trade_date")
    try:
        result = gann7.analyze(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"{sym}: {e}") from e
    q = {}
    try:
        q = cached_quotes([sym]).get(sym) or {}
    except Exception as e:                    # noqa: BLE001
        logger.warning("quote failed for %s: %s", sym, e)
    result.update({"symbol": sym, "name": q.get("name", sym)})
    return result


@app.get("/api/kline/{symbol}")
def kline(symbol: str, period: str = "day", limit: int = 240):
    import pandas as pd

    if period not in ("day", "week", "month", "m5", "m15", "m30", "m60"):
        raise HTTPException(status_code=422, detail=f"不支持的周期: {period}")
    # 先规范化再查缓存：600000 与 sh600000 共用同一份缓存，非法代码直接 422。
    sym = _normalize_symbol(symbol)
    limit = max(60, min(limit, 1023))
    cache_ttl = 30 if period.startswith("m") else 60
    key = ("kline", sym, period, limit)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if period in ("day", "week", "month"):
        df = mtf.ensure(sym, period)
        tcol = "trade_date"
    else:
        import data_source as ds
        df = ds.fetch_minute(sym, int(period[1:]), 1023)
        tcol = "trade_time"
    df = _validated_bars(df, sym, period, tcol).tail(limit).reset_index(drop=True)

    close = df["close"].astype(float)
    mas = {f"ma{n}": [None if pd.isna(x) else round(float(x), 3)
                      for x in close.rolling(n).mean()]
           for n in (5, 10, 20, 60, 250)}
    result = {
        "symbol": sym, "period": period,
        "bars": [{"t": str(getattr(r, tcol)), "o": float(r.open), "h": float(r.high),
                  "l": float(r.low), "c": float(r.close), "v": float(r.volume)}
                 for r in df.itertuples()],
        "mas": mas,
    }
    try:
        result["wave"] = _wave_overlay(df, wv.analyze_wave(df), tcol)
    except Exception as e:                       # noqa: BLE001
        logger.exception("wave overlay failed for %s %s", sym, period)
        result["wave"] = {"valid": False, "why": f"浪型计算失败：{type(e).__name__}: {e}",
                          "pivots": [], "fib": None}
    return _cache_set(key, result, cache_ttl)


@app.post("/api/wave/ai/{symbol}")
def wave_ai_choice(symbol: str, period: str = "day", force: bool = False):
    """让模型在规则枚举出的浪组里选主数法；编号必须落在候选集合内。"""
    if period not in ("day", "week", "month", "m5", "m15", "m30", "m60"):
        raise HTTPException(status_code=422, detail=f"不支持的周期: {period}")
    sym = _normalize_symbol(symbol)
    if period in ("day", "week", "month"):
        df = mtf.ensure(sym, period)
        tcol = "trade_date"
    else:
        import data_source as ds
        df = ds.fetch_minute(sym, int(period[1:]), 1023)
        tcol = "trade_time"
    df = _validated_bars(df, sym, period, tcol).tail(800).reset_index(drop=True)

    key = ("wave_ai", sym, period, str(df[tcol].iloc[-1]), len(df))
    if not force:
        cached = _cache_get(key)
        if cached is not None:
            return {**cached, "cached": True}

    result = wv.analyze_wave(df)
    payload = wai.build_payload(df, result, tcol, sym, period)
    if not payload["candidates"]:
        out = {"symbol": sym, "period": period, "model": None, "cached": False,
               "chosen": None, "alternatives": [], "confidence": None,
               "reason": "规则没有枚举出任何通过校验的五浪结构，AI 无需选择",
               "candidates": []}
        return _cache_set(key, out, 600)

    try:
        choice = wai.choose(payload)
    except ai.AIAnalysisError as exc:
        raise HTTPException(status_code=502,
                            detail={"error": exc.stage,
                                    "message": str(exc)[:800]}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502,
                            detail={"error": "wave_choice",
                                    "message": str(exc)[:800]}) from exc

    try:
        out = wai.assemble(payload, choice, ai.get_last_model())
    except ValueError as exc:
        raise HTTPException(status_code=502,
                            detail={"error": "wave_choice",
                                    "message": str(exc)[:800]}) from exc
    return _cache_set(key, out, 1800)


@app.get("/api/plan/{symbol}")
def plan_detail(symbol: str):
    """长线/短线条件式计划：支撑压力、加减仓区间与失效位。"""
    sym = _normalize_symbol(symbol)
    day = _validated_bars(mtf.ensure(sym, "day"), sym, "day", "trade_date")
    week = _validated_bars(mtf.ensure(sym, "week"), sym, "week", "trade_date")
    q = {}
    try:
        q = cached_quotes([sym]).get(sym) or {}
    except Exception as e:                       # noqa: BLE001
        logger.warning("quote failed for %s: %s", sym, e)
    price = float(q.get("price") or day["close"].iloc[-1])
    key = ("plan", sym, str(day["trade_date"].iloc[-1]), round(price, 3))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    analysis = mtf.analyze(sym, price)
    try:
        gann = gann7.analyze(day)
    except ValueError as e:
        logger.warning("gann7 failed for %s: %s", sym, e)
        gann = None
    day_wave = wv.analyze_wave(day)
    week_wave = (analysis.get("week") or {}).get("wave") or wv.analyze_wave(week)
    out = plan_mod.build(sym, q.get("name", sym), day, week, price,
                         analysis, gann, day_wave, week_wave)
    return _cache_set(key, out, 60)


REVIEW_PROMPT = """你在复核一只 A 股标的的规则化结论。请只基于下列信息作答。

标的：{name}（{code}），现价 {price}，涨跌 {pct}%
周线趋势：{week}
月线趋势：{month}
综合评分：{score}，系统建议：{action}

触发规则：
{votes}

请用 2–3 句话回答：
1. 当前最关键的矛盾是什么（例如日线与周月线方向冲突、超买但趋势仍向上等）；
2. 这个建议最可能错在哪里；
3. 不要给出具体点位、目标价或收益预测，不要编造未提供的数据。"""


@app.post("/api/review")
def review():
    from concurrent.futures import ThreadPoolExecutor

    items, _quote_ok = _watch_items()
    if not items:
        return {"items": [], "note": "自选为空"}

    S.load_env()
    base, key = os.environ.get("OCR_API_BASE"), os.environ.get("OCR_API_KEY")
    model = os.environ.get("LLM_MODEL") or os.environ.get("OCR_MODEL")
    if not (base and key and model):
        return {"items": [], "error": "服务端未配置模型接口"}

    def ask(it):
        wk = (it.get("week") or {}).get("trend")
        mo = (it.get("month") or {}).get("trend")
        votes = "\n".join(
            f"- {'偏多' if (v.get('side') or 0) > 0 else '偏空' if (v.get('side') or 0) < 0 else '中性'} "
            f"{v.get('rule', '未命名规则')}：《{v.get('book', '')}》P{v.get('page', '?')}｜"
            f"{v.get('evidence', '无证据')}"
            for v in it.get("votes") or [])
        prompt = REVIEW_PROMPT.format(
            name=it.get("name"), code=it.get("code"), price=it.get("price"),
            pct=it.get("change_pct"),
            week=(f"向上 MA20 {wk['ma_fast']} / MA60 {wk['ma_slow']}" if wk else "数据不足"),
            month=(f"向上 MA12 {mo['ma_fast']} / MA24 {mo['ma_slow']}" if mo else "数据不足"),
            score=it.get("score"), action=it.get("action"), votes=votes or "（无）")
        try:
            r = requests.post(f"{base.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {key}",
                                       "x-opencode-session": "review"},
                              json={"model": model, "temperature": 0.3,
                                    "messages": [{"role": "user", "content": prompt}]},
                              timeout=60)
            if r.status_code != 200:
                return {"symbol": it["symbol"], "comment": f"接口 {r.status_code}"}
            return {"symbol": it["symbol"],
                    "comment": r.json()["choices"][0]["message"]["content"].strip()}
        except Exception as e:
            return {"symbol": it["symbol"], "comment": f"失败：{type(e).__name__}"}

    with ThreadPoolExecutor(max_workers=max(len(items), 1)) as ex:
        out = list(ex.map(ask, items))
    return {"items": out, "model": model}


# ---------------------------------------------------------------------------
# AI 读图研判（数字由 quant/ai_analyst.py 计算，模型只描述与研判）
# ---------------------------------------------------------------------------

@app.post("/api/ai/analyze/{symbol}")
def ai_analyze(symbol: str, force: bool = False):
    sym = _normalize_symbol(symbol)
    try:
        return ai.analyze(sym, force=force)
    except ai.AIAnalysisError as exc:
        raise HTTPException(status_code=502,
                            detail={"error": exc.stage, "message": str(exc)[:800]}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/ai/latest/{symbol}")
def ai_latest(symbol: str):
    """只读当天已有结果，不触发模型调用。"""
    sym = _normalize_symbol(symbol)
    data = ai.latest(sym)
    if data is None:
        raise HTTPException(status_code=404, detail="该标的今天还没有 AI 研判")
    return data


@app.get("/api/ai/advice")
def ai_advice(symbol: Optional[str] = None):
    try:
        sym = _normalize_symbol(symbol) if symbol else None
        return {"items": ai.list_advice(sym)}
    except ai.AIAnalysisError as exc:
        raise HTTPException(status_code=502,
                            detail={"error": exc.stage, "message": str(exc)[:800]}) from exc


@app.post("/api/ai/run_daily")
def ai_run_daily():
    return ai.run_daily(refresh=False)


@app.get("/", dependencies=[])
def index():
    # 前端单文件迭代频繁，禁用缓存，避免浏览器拿旧页面看不到新功能
    return FileResponse(Path(__file__).parent / "static" / "index.html",
                        headers={"Cache-Control": "no-store, max-age=0"})
