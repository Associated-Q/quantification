"""每日建议快照与事后验证。

表（只写 advice_snapshot，首次调用时自动建表）：
    advice_snapshot(trade_date, symbol, action, score, price,
                    target_position, rules, PRIMARY KEY(trade_date, symbol))

run_snapshot()   对当前自选逐只调用 multi_tf.analyze，写入当天快照（同一天重跑覆盖）。
eval_snapshot()  用 kline_daily 计算快照之后第 5/10/20 个交易日的收益率与命中情况。
"""
import json
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_tf
import position as pos
import realtime
from position import connect

logger = logging.getLogger("snapshot")

_UP_ACTIONS = ("加仓", "持有偏多")
_DOWN_ACTIONS = ("减仓", "清仓")


def ensure(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS advice_snapshot (
            trade_date DATE,
            symbol VARCHAR,
            action VARCHAR,
            score DOUBLE,
            price DOUBLE,
            target_position DOUBLE,
            rules VARCHAR,
            PRIMARY KEY (trade_date, symbol)
        )
    """)


def _last_close(sym):
    try:
        df = realtime.period_history(sym, "kline_daily")
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return float(df["close"].astype(float).iloc[-1])


def _rules_json(votes):
    return json.dumps([{
        "rule": v.get("rule"), "side": v.get("side"),
        "weight": v.get("weight"), "evidence": v.get("evidence"),
        "book": v.get("book"), "page": v.get("page"),
    } for v in (votes or [])], ensure_ascii=False)


def run_snapshot(trade_date=None):
    """对当前自选与持仓生成当天快照；同一天重跑会先删后写，结果可重复。"""
    trade_date = trade_date or date.today()
    symbols = list(realtime.items())
    try:                                   # 持仓也要有历史，否则拿不到持有期的建议记录
        for row in pos.list_positions(with_quotes=False):
            sym = row["symbol"]
            if sym not in symbols:
                symbols.append(sym)
    except Exception as e:
        logger.warning("读取持仓失败，仅对自选生成快照: %s", e)
    result = {"date": trade_date.isoformat(), "count": 0, "errors": []}
    if not symbols:
        logger.warning("自选为空，未生成快照")
        return result

    try:
        quotes = realtime.fetch_quotes(symbols)
    except Exception as e:
        quotes = {}
        logger.warning("实时行情获取失败，改用最后收盘价: %s", e)

    rows = []
    for sym in symbols:
        q = quotes.get(sym) or {}
        price = q.get("price")
        try:
            analysis = multi_tf.analyze(sym, price)
            action = analysis.get("action")
            score = analysis.get("score")
            target_position = analysis.get("target_position")
            rules = _rules_json(analysis.get("votes"))
        except Exception as e:
            action, score, target_position = "数据异常", 0.0, None
            rules = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            msg = f"{sym}: {type(e).__name__}: {e}"
            result["errors"].append(msg)
            logger.warning("快照分析失败 %s", msg)
        if price is None:
            price = _last_close(sym)
        rows.append([trade_date, sym, action, score, price, target_position, rules])

    con = connect()
    try:
        ensure(con)
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM advice_snapshot WHERE trade_date = ?", [trade_date])
        con.executemany(
            "INSERT INTO advice_snapshot "
            "(trade_date, symbol, action, score, price, target_position, rules) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()

    result["count"] = len(rows)
    logger.info("快照完成 %s：%d 只", result["date"], result["count"])
    return result


def _hit(action, ret):
    """1=命中，0=未命中，None=不判定（观望、数据异常或收益不可算）。"""
    if ret is None:
        return None
    if action in _UP_ACTIONS:
        return 1 if ret > 0 else 0
    if action in _DOWN_ACTIONS:
        return 1 if ret < 0 else 0
    return None


def eval_snapshot():
    """对每条历史快照计算之后第 5/10/20 个交易日的收益率。

    hit 是对外契约里的标量字段，取 5 日命中结果（1=命中，0=未命中，None=待定），
    与前端 hitState() 的单一命中列一致；三个周期的明细另见 hit5/hit10/hit20
    与 hit_by_horizon。
    """
    con = connect()
    try:
        ensure(con)
        snaps = con.execute(
            "SELECT trade_date, symbol, action, price FROM advice_snapshot "
            "ORDER BY trade_date, symbol").fetchall()
        bars = con.execute(
            "SELECT symbol, trade_date, close FROM kline_daily "
            "ORDER BY symbol, trade_date").fetchall()
    finally:
        con.close()

    history = {}
    for sym, d, close in bars:
        history.setdefault(sym, []).append((d, float(close)))

    items = []
    for trade_date, sym, action, snap_price in snaps:
        series = history.get(sym, [])
        base = float(snap_price) if snap_price else None
        if base is None or base <= 0:
            prior = [c for d, c in series if d <= trade_date]
            base = prior[-1] if prior else None
        future = [c for d, c in series if d > trade_date]

        rets = {}
        for n in (5, 10, 20):
            rets[n] = (round((future[n - 1] / base - 1) * 100, 2)
                       if base and len(future) >= n else None)
        hit = {"5": _hit(action, rets[5]), "10": _hit(action, rets[10]),
               "20": _hit(action, rets[20])}
        items.append({
            "date": trade_date.isoformat(), "symbol": sym, "action": action,
            "price": round(base, 3) if base else None,
            "ret5": rets[5], "ret10": rets[10], "ret20": rets[20],
            "hit": hit["5"], "hit5": hit["5"], "hit10": hit["10"],
            "hit20": hit["20"], "hit_by_horizon": hit,
        })
    return items


def list_snapshots(symbol=None):
    con = connect()
    try:
        ensure(con)
        if symbol:
            rows = con.execute(
                "SELECT trade_date, symbol, action, score, price, target_position "
                "FROM advice_snapshot WHERE symbol = ? "
                "ORDER BY trade_date DESC, symbol", [symbol]).fetchall()
        else:
            rows = con.execute(
                "SELECT trade_date, symbol, action, score, price, target_position "
                "FROM advice_snapshot ORDER BY trade_date DESC, symbol").fetchall()
    finally:
        con.close()
    return [{
        "date": d.isoformat(), "symbol": sym, "action": action,
        "score": score, "price": price, "target_position": target_position,
    } for d, sym, action, score, price, target_position in rows]
