"""实时行情与自选列表。

行情用腾讯 qt 接口（GBK 编码，零鉴权）；自选列表存在本地 DuckDB。
"""
import logging
import os
import threading
import time

import duckdb
import pandas as pd
import requests

import data_source

logger = logging.getLogger("technical-analysis.realtime")

QT_URL = "https://qt.gtimg.cn/q="
SMARTBOX = "https://smartbox.gtimg.cn/s3/?v=2&q={}&t=all"
HEADERS = {"User-Agent": "Mozilla/5.0"}

_MARKET_LAST = {"date": None, "ts": 0.0}
_MARKET_LOCK = threading.Lock()


_GENERIC = {"大盘", "指数", "市场", "板块", "个股", "股票", "基金", "行情", "走势",
            "仓位", "怎么", "如何", "现在", "今天", "明天", "后市", "操作", "建议"}
# 出现这些词时，说明问的是概念而不是某个标的，不再接受 2 字的模糊匹配
_CONCEPT = ("公式", "理论", "方法", "策略", "指标", "形态", "原理", "技巧",
            "是什么", "怎么用", "含义", "区别", "意义")
_RESOLVE_CACHE = {}


def _smartbox(term):
    import json as _json
    import urllib.parse

    try:
        r = requests.get(SMARTBOX.format(urllib.parse.quote(term)), headers=HEADERS, timeout=8)
        r.encoding = "gbk"
        text = r.text
        if 'v_hint="' not in text:
            return None
        inner = text.split('v_hint="', 1)[1].split('"', 1)[0]
        if not inner or inner == "N":
            return None
        raw = _json.loads('"' + inner + '"')      # 还原 \uXXXX
    except Exception:
        return None
    for item in raw.split("^"):
        parts = item.split("~")
        if len(parts) < 5:
            continue
        market, code, name, _, typ = parts[:5]
        if typ in ("GP-A", "ETF", "LOF") and market in ("sh", "sz"):
            return {"symbol": market + code, "code": code, "name": name, "type": typ}
    return None


def resolve_symbol(query: str):
    """从一句话里解析出可交易标的；解析不到返回 None。

    只接受 A 股个股、ETF、LOF。整句会按片段从长到短试前缀，因此
    「药明康德现在怎么样」也能命中「药明康德」。
    """
    import re as _re

    q = (query or "").strip()
    if not q:
        return None
    if q in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[q]

    result = None
    m = _re.fullmatch(r"(sh|sz|bj)?(\d{6})", q.lower())
    if m:
        result = {"symbol": to_symbol(q), "code": m.group(2), "name": "", "type": ""}
    else:
        cands = []
        if 2 <= len(q) <= 8 and q not in _GENERIC:
            cands.append(q)
        for seg in _re.split(r"[，。？！、；,:.\s]+", q):
            if len(seg) < 2 or seg in _GENERIC:
                continue
            for L in range(min(8, len(seg)), 1, -1):
                cand = seg[:L]
                if cand not in _GENERIC and cand not in cands:
                    cands.append(cand)
        is_concept = any(w in q for w in _CONCEPT)
        for cand in cands[:8]:
            if is_concept and len(cand) < 3:      # 「凯利」这类 2 字别匹到「凯利泰」
                continue
            info = _smartbox(cand)
            if not info:
                continue
            name = info["name"] or ""
            # 名称要与问句有实质重合，避免「银行板块」这类泛词误配到某只个股
            if name and (name in q or cand in name):
                result = info
                break
    _RESOLVE_CACHE[q] = result
    return result


def fetch_quotes(codes):
    """codes 可传 510300 / sh510300；返回 {symbol: {...}}。"""
    syms = [data_source.to_symbol(c) for c in codes]
    if not syms:
        return {}
    r = requests.get(QT_URL + ",".join(syms), headers=HEADERS, timeout=15)
    r.raise_for_status()
    r.encoding = "gbk"
    out = {}
    for line in r.text.split(";"):
        if "=" not in line or "~" not in line:
            continue
        key, val = line.split("=", 1)
        sym = key.strip().replace("v_", "")
        p = val.strip().strip('"').split("~")
        if len(p) < 40 or not p[3]:
            continue
        try:
            out[sym] = {
                "symbol": sym, "code": p[2], "name": p[1],
                "price": float(p[3]), "prev_close": float(p[4]),
                "open": float(p[5]), "high": float(p[33]), "low": float(p[34]),
                "change": float(p[31]) if p[31] else 0.0,
                "change_pct": float(p[32]) if p[32] else 0.0,
                "volume": float(p[6]) if p[6] else 0.0,
            }
        except (ValueError, IndexError):
            continue
    return out


def _con():
    return duckdb.connect(data_source.DB)


DEFAULT_GROUP_NAME = "默认"


def _ensure_watch(con):
    con.execute("CREATE TABLE IF NOT EXISTS watch_groups (id INTEGER PRIMARY KEY, "
                "name VARCHAR UNIQUE, sort_order INTEGER DEFAULT 0)")
    con.execute("CREATE TABLE IF NOT EXISTS watch_group_map (group_id INTEGER, "
                "symbol VARCHAR, added_at TIMESTAMP DEFAULT current_timestamp, "
                "PRIMARY KEY (group_id, symbol))")
    con.execute("CREATE TABLE IF NOT EXISTS watchlist (symbol VARCHAR PRIMARY KEY, "
                "added_at TIMESTAMP DEFAULT current_timestamp)")
    if con.execute("SELECT count(*) FROM watch_groups").fetchone()[0] == 0:
        con.execute("INSERT INTO watch_groups(id, name, sort_order) VALUES (1, ?, 0)",
                    [DEFAULT_GROUP_NAME])
        # 旧版单列表的数据并进默认分组，可重复执行
        con.execute("INSERT OR IGNORE INTO watch_group_map(group_id, symbol) "
                    "SELECT 1, symbol FROM watchlist")


def groups():
    con = _con()
    _ensure_watch(con)
    rows = con.execute(
        "SELECT g.id, g.name, count(m.symbol) FROM watch_groups g "
        "LEFT JOIN watch_group_map m ON m.group_id = g.id "
        "GROUP BY g.id, g.name, g.sort_order ORDER BY g.sort_order, g.id").fetchall()
    con.close()
    return [{"id": r[0], "name": r[1], "count": r[2]} for r in rows]


def add_group(name):
    name = (name or "").strip()
    if not name:
        raise ValueError("分组名不能为空")
    if len(name) > 20:
        raise ValueError("分组名不能超过 20 字")
    con = _con()
    _ensure_watch(con)
    if con.execute("SELECT 1 FROM watch_groups WHERE name = ?", [name]).fetchone():
        con.close()
        raise ValueError(f"分组「{name}」已存在")
    nid = con.execute("SELECT coalesce(max(id), 0) + 1 FROM watch_groups").fetchone()[0]
    con.execute("INSERT INTO watch_groups(id, name, sort_order) VALUES (?, ?, ?)",
                [nid, name, nid])
    con.close()
    return {"id": nid, "name": name}


def rename_group(gid, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("分组名不能为空")
    con = _con()
    _ensure_watch(con)
    if not con.execute("SELECT 1 FROM watch_groups WHERE id = ?", [gid]).fetchone():
        con.close()
        raise ValueError(f"分组 {gid} 不存在")
    con.execute("UPDATE watch_groups SET name = ? WHERE id = ?", [name, gid])
    con.close()
    return {"id": gid, "name": name}


def remove_group(gid):
    con = _con()
    _ensure_watch(con)
    total = con.execute("SELECT count(*) FROM watch_groups").fetchone()[0]
    if total <= 1:
        con.close()
        raise ValueError("至少要保留一个分组")
    con.execute("DELETE FROM watch_group_map WHERE group_id = ?", [gid])
    con.execute("DELETE FROM watch_groups WHERE id = ?", [gid])
    con.close()
    return gid


def add(code, group_id=1):
    sym = data_source.to_symbol(code)
    con = _con()
    _ensure_watch(con)
    if not con.execute("SELECT 1 FROM watch_groups WHERE id = ?", [group_id]).fetchone():
        con.close()
        raise ValueError(f"分组 {group_id} 不存在")
    con.execute("INSERT OR IGNORE INTO watch_group_map(group_id, symbol) VALUES (?, ?)",
                [group_id, sym])
    con.close()
    return sym


def remove(code, group_id=None):
    """group_id 为空时从所有分组移除。"""
    sym = data_source.to_symbol(code)
    con = _con()
    _ensure_watch(con)
    if group_id is None:
        con.execute("DELETE FROM watch_group_map WHERE symbol = ?", [sym])
    else:
        con.execute("DELETE FROM watch_group_map WHERE symbol = ? AND group_id = ?",
                    [sym, group_id])
    con.close()
    return sym


def items(group_id=None):
    """返回指定分组的标的；不传则返回所有分组去重后的集合。"""
    con = _con()
    _ensure_watch(con)
    if group_id is None:
        rows = con.execute(
            "SELECT symbol, min(added_at) AS t FROM watch_group_map "
            "GROUP BY symbol ORDER BY t").fetchall()
    else:
        rows = con.execute(
            "SELECT symbol FROM watch_group_map WHERE group_id = ? ORDER BY added_at",
            [group_id]).fetchall()
    con.close()
    return [r[0] for r in rows]


def period_history(sym, table="kline_daily"):
    """按周期表读取历史（表名来自固定映射，不接受外部输入）。"""
    if table not in ("kline_daily", "kline_weekly", "kline_monthly"):
        raise ValueError(f"未知周期表: {table}")
    con = _con()
    try:
        df = con.execute(
            f"SELECT trade_date, open, high, low, close, volume FROM {table} "
            "WHERE symbol = ? ORDER BY trade_date", [sym]).df()
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df.reset_index(drop=True)


def market_last_date(ttl: float = 600):
    """基准指数（上证）最近一个有K线的交易日；缓存 ttl 秒，避免每只票都联网。

    取不到时返回 None，调用方按"未知"处理，不据此判定过期。
    """
    now = time.time()
    with _MARKET_LOCK:
        cached = _MARKET_LAST["date"]
        if cached and now - _MARKET_LAST["ts"] < ttl:
            return cached
    try:
        ref = data_source.fetch_daily("sh000001", count=10)
        d = str(ref["trade_date"].iloc[-1])[:10]
    except Exception as e:                     # noqa: BLE001
        logger.warning("取大盘最新交易日失败: %s", e)
        return cached
    with _MARKET_LOCK:
        _MARKET_LAST.update({"date": d, "ts": now})
    return d


def is_stale(sym, last_date, table="kline_daily"):
    """本地最后一根K线早于大盘最新交易日即为过期。"""
    ref = market_last_date()
    if ref is None or last_date is None:
        return False
    return str(last_date)[:10] < ref


def history(sym, count=260):
    """取本地日线；缺失或过期则现拉一次并入库。"""
    df = period_history(sym, "kline_daily")
    last = None if df.empty else str(df["trade_date"].iloc[-1])[:10]
    if len(df) < 60 or is_stale(sym, last):
        try:
            data_source.save(data_source.fetch_daily(sym, count=count))
            df = period_history(sym, "kline_daily")
        except Exception as e:                # noqa: BLE001
            if len(df) < 60:
                raise
            logger.warning("%s 刷新日线失败，沿用本地数据（最后 %s）: %s", sym, last, e)
    return df
