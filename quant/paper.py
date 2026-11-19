"""模拟盘：按多周期规则执行调仓并记账。

表（与 data_source.DB 同库，首次调用时自动建表）：
    paper_account(id INTEGER PK, initial DOUBLE, cash DOUBLE,
                  started_at TIMESTAMP, last_run TIMESTAMP)
    paper_positions(symbol VARCHAR PK, shares INTEGER, cost DOUBLE, last_buy_date DATE)
    paper_trades(id INTEGER PK, trade_date DATE, symbol VARCHAR, side VARCHAR,
                 shares INTEGER, price DOUBLE, amount DOUBLE, fee DOUBLE, reason VARCHAR)
    paper_equity(trade_date DATE PK, equity DOUBLE)
    paper_runs(trade_date DATE PK, summary VARCHAR)   # 持久化最近一次复盘摘要

规则是固定的：target_position 只来自 multi_tf.analyze，本模块不根据历史盈亏
调整目标仓位或任何阈值，只负责执行与记账。
"""
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import data_source  # noqa: E402
import multi_tf  # noqa: E402
import realtime  # noqa: E402
from position import connect  # noqa: E402

DEFAULT_INITIAL = 1_000_000.0
REBALANCE_THRESHOLD = 0.05     # 差额小于权益 5% 不交易
MAX_POSITIONS = 3              # 模拟盘最多同时持有几只 ETF
COSTS = engine.Costs()

# 模拟盘标的池：只做 ETF，不碰个股。兼顾宽基与行业，避免集中在单一风格。
ETF_POOL = [
    "510300",  # 沪深300
    "510500",  # 中证500
    "510050",  # 上证50
    "159915",  # 创业板
    "588000",  # 科创50
    "159949",  # 创业板50
    "512480",  # 半导体
    "512880",  # 证券
    "512170",  # 医疗
    "515050",  # 5G通信
    "518880",  # 黄金
    "513050",  # 中概互联
    "512690",  # 酒
    "516160",  # 新能源
]


def _is_etf(sym):
    """ETF/LOF 最小变动价位为 0.001，个股为 0.01；指数没有价位返回 None。"""
    return engine.tick_size(sym) == 0.001


def ensure(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_account (
            id INTEGER PRIMARY KEY,
            initial DOUBLE,
            cash DOUBLE,
            started_at TIMESTAMP,
            last_run TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol VARCHAR PRIMARY KEY,
            shares INTEGER,
            cost DOUBLE,
            last_buy_date DATE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY,
            trade_date DATE,
            symbol VARCHAR,
            side VARCHAR,
            shares INTEGER,
            price DOUBLE,
            amount DOUBLE,
            fee DOUBLE,
            reason VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_equity (
            trade_date DATE PRIMARY KEY,
            equity DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_runs (
            trade_date DATE PRIMARY KEY,
            summary VARCHAR
        )
    """)


def _iso(ts):
    return ts.isoformat() if ts is not None else None


def _validate_initial(value):
    try:
        initial = float(value)
    except (TypeError, ValueError):
        raise ValueError("初始资金必须为正数") from None
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("初始资金必须为正数")
    return initial


def _load_account(con, initial=DEFAULT_INITIAL):
    row = con.execute(
        "SELECT id, initial, cash, started_at, last_run FROM paper_account "
        "WHERE id = 1").fetchone()
    if row is None:
        con.execute(
            "INSERT INTO paper_account (id, initial, cash, started_at, last_run) "
            "VALUES (1, ?, ?, current_timestamp, NULL)", [initial, initial])
        row = con.execute(
            "SELECT id, initial, cash, started_at, last_run FROM paper_account "
            "WHERE id = 1").fetchone()
    return {"id": int(row[0]), "initial": float(row[1]), "cash": float(row[2]),
            "started_at": row[3], "last_run": row[4]}


def reset(initial=DEFAULT_INITIAL):
    """清空模拟盘并恢复初始资金；返回 {ok, equity}。"""
    initial = _validate_initial(initial)
    con = connect()
    try:
        ensure(con)
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_trades")
        con.execute("DELETE FROM paper_equity")
        con.execute("DELETE FROM paper_runs")
        con.execute("DELETE FROM paper_account")
        con.execute(
            "INSERT INTO paper_account (id, initial, cash, started_at, last_run) "
            "VALUES (1, ?, ?, current_timestamp, NULL)", [initial, initial])
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()
    return {"ok": True, "equity": round(initial, 2)}


def _last_close(sym):
    """行情缺失时的真实兜底：库里最后一根日线收盘价；没有就返回 None。"""
    con = connect()
    try:
        ensure(con)
        row = con.execute(
            "SELECT close FROM kline_daily WHERE symbol = ? "
            "ORDER BY trade_date DESC LIMIT 1", [sym]).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    value = float(row[0])
    return value if math.isfinite(value) and value > 0 else None


def _quote_map(symbols):
    if not symbols:
        return {}
    try:
        return realtime.fetch_quotes(symbols)
    except Exception:
        return {}


def _prices(symbols, quotes):
    """每只标的的估值价：实时价 → 最后一根日线收盘价 → None（不编造）。"""
    out = {}
    for sym in symbols:
        q = quotes.get(sym) or {}
        price = q.get("price")
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            price = _last_close(sym)
        if price is None:
            continue
        out[sym] = float(price)
    return out


def _load_positions(con):
    rows = con.execute(
        "SELECT symbol, shares, cost, last_buy_date FROM paper_positions").fetchall()
    return {r[0]: {"symbol": r[0], "shares": int(r[1]), "cost": float(r[2]),
                   "last_buy_date": r[3]} for r in rows}


def _position_value(positions, prices):
    total = 0.0
    for sym, p in positions.items():
        price = prices.get(sym, p["cost"])
        total += p["shares"] * price
    return total


def _write_trade(con, day, sym, side, shares, price, amount, fee, reason):
    nid = con.execute(
        "SELECT coalesce(max(id), 0) + 1 FROM paper_trades").fetchone()[0]
    con.execute(
        "INSERT INTO paper_trades (id, trade_date, symbol, side, shares, price, "
        "amount, fee, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [nid, day, sym, side, shares, price, amount, fee, reason])


def _save_position(con, sym, shares, cost, last_buy_date):
    con.execute("""
        INSERT INTO paper_positions (symbol, shares, cost, last_buy_date)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
            shares = excluded.shares,
            cost = excluded.cost,
            last_buy_date = excluded.last_buy_date
    """, [sym, shares, cost, last_buy_date])


def rebalance(trade_date=None):
    """按 multi_tf 的 target_position 调仓；返回 orders/skipped/摘要。"""
    day = trade_date or date.today()
    con = connect()
    try:
        ensure(con)
        acc = _load_account(con)
        cash = acc["cash"]
        positions = _load_positions(con)
    finally:
        con.close()

    # 标的池 = 内置 ETF 池 + 已有持仓（持仓里若有历史遗留的个股，保留以便卖出）
    symbols = []
    for code in ETF_POOL:
        sym = data_source.to_symbol(code)
        if sym not in symbols:
            symbols.append(sym)
    for sym in positions:
        if sym not in symbols:
            symbols.append(sym)

    skipped = []
    orders = []
    equity = round(cash + _position_value(positions, {}), 2)

    if not symbols:
        summary = {"buy_count": 0, "sell_count": 0, "skipped_count": 0,
                   "position_count": len(positions), "market_value": 0.0,
                   "equity": equity}
        _finalize(day, cash, equity, [], summary)
        return {"date": day.isoformat(), "orders": [], "skipped": [],
                "equity_after": equity, "summary": summary}

    quotes = _quote_map(symbols)
    prices = _prices(symbols, quotes)
    for sym in symbols:
        if sym not in prices:
            skipped.append({"symbol": sym, "why": "无实时行情且无日线收盘价，无法估值"})

    # 估值：持仓价缺失时按成本价计入，避免权益失真
    equity = round(cash + _position_value(positions, prices), 2)
    threshold = equity * REBALANCE_THRESHOLD

    analyses = {}
    for sym in symbols:
        if sym not in prices:
            continue
        try:
            analyses[sym] = multi_tf.analyze(sym, prices[sym])
        except Exception as e:
            skipped.append({"symbol": sym, "why": f"分析失败: {type(e).__name__}: {e}"})

    # 最多持有 MAX_POSITIONS 只：按规则评分取前 N（同分看目标仓位）。
    # 落选的标的即便有目标仓位也不买入；已持有但落选的会在下面被卖出。
    candidates = [s for s in symbols
                  if s in analyses and not engine.is_index(s) and _is_etf(s)
                  and (analyses[s].get("target_position") or 0) > 0]
    candidates.sort(key=lambda s: (-(analyses[s].get("score") or 0),
                                   -(analyses[s].get("target_position") or 0)))
    selected = set(candidates[:MAX_POSITIONS])
    for s in candidates[MAX_POSITIONS:]:
        skipped.append({"symbol": s,
                        "why": f"未入选前 {MAX_POSITIONS} 名，不建仓"})

    # 入选标的的目标仓位按比例缩放到 100% 以内，避免总仓位超限
    total_target = sum(max(0.0, min(1.0, float(analyses[s]["target_position"])))
                       for s in selected)
    scale = 1.0 / total_target if total_target > 1.0 else 1.0

    sells, buys = [], []
    for sym in symbols:
        if sym not in analyses:
            continue
        if engine.is_index(sym):
            skipped.append({"symbol": sym, "why": "指数不可交易"})
            continue
        if not _is_etf(sym) and sym not in positions:
            skipped.append({"symbol": sym, "why": "模拟盘只买 ETF"})
            continue
        raw_target = analyses[sym].get("target_position")
        if raw_target is None:
            skipped.append({"symbol": sym, "why": "规则未给出目标仓位"})
            continue
        # 落选标的：已持有的清掉（target=0），未持有的自然不动
        target = (max(0.0, min(1.0, float(raw_target))) * scale
                  if sym in selected else 0.0)
        price = prices[sym]
        current_shares = positions.get(sym, {}).get("shares", 0)
        current_value = current_shares * price
        diff = target * equity - current_value
        if abs(diff) < threshold:
            skipped.append({
                "symbol": sym,
                "why": f"差额 {diff:,.0f} 元低于阈值 {threshold:,.0f} 元（5%）"})
            continue
        if diff > 0:
            # T+1 期间已持有的卖不掉，若不加这道闸，同一天重复复盘会不断买进新标的，
            # 实际持仓数会突破 MAX_POSITIONS。
            pending_new = len(positions) + sum(1 for b in buys if b["symbol"] not in positions)
            if sym not in positions and pending_new >= MAX_POSITIONS:
                skipped.append({"symbol": sym,
                                "why": f"已达持仓上限 {MAX_POSITIONS} 只"})
                continue
            lots = int(diff / price // COSTS.lot) * COSTS.lot
            if lots <= 0:
                skipped.append({"symbol": sym, "why": "差额不足一手，跳过"})
                continue
            buys.append({"symbol": sym, "shares": lots, "price": price,
                         "target": target,
                         "reason": f"目标仓位 {target:.0%}，缺口 {diff:,.0f} 元"})
        else:
            lots = min(int((-diff) / price // COSTS.lot) * COSTS.lot, current_shares)
            if lots <= 0:
                skipped.append({"symbol": sym, "why": "差额不足一手，跳过"})
                continue
            sells.append({"symbol": sym, "shares": lots, "price": price,
                          "target": target,
                          "reason": f"目标仓位 {target:.0%}，超出 {(-diff):,.0f} 元"})

    # 先卖后买：卖出释放现金
    for intent in sells:
        sym, shares = intent["symbol"], intent["shares"]
        pos = positions.get(sym)
        if pos is None or shares <= 0:
            continue
        if pos["last_buy_date"] == day:
            skipped.append({"symbol": sym, "why": "T+1：今日买入的持仓不可卖出"})
            continue
        shares = min(shares, pos["shares"])
        tick = engine.tick_size(sym) or 0.01        # 滑点按标的最小变动价位，股票 0.01、ETF 0.001
        px = round(intent["price"] - tick, 4)
        if px <= 0:
            skipped.append({"symbol": sym, "why": "滑点后价格非正，跳过"})
            continue
        amount = round(shares * px, 2)
        fee = engine._fees(amount, COSTS, sell=True)
        cash += amount - fee
        pos["shares"] -= shares
        if pos["shares"] <= 0:
            del positions[sym]
        orders.append({"symbol": sym, "side": "sell", "shares": shares, "price": px,
                       "amount": round(amount, 2), "fee": round(fee, 2),
                       "reason": intent["reason"]})

    for intent in buys:
        sym = intent["symbol"]
        tick = engine.tick_size(sym) or 0.01
        px = round(intent["price"] + tick, 4)
        if px <= 0:
            skipped.append({"symbol": sym, "why": "滑点后价格非正，跳过"})
            continue
        size = intent["shares"]
        # 现金不足时按可用现金缩减手数
        while size > 0:
            amount = size * px
            fee = engine._fees(amount, COSTS, sell=False)
            if amount + fee <= cash + 1e-9:
                break
            size -= COSTS.lot
        if size <= 0:
            skipped.append({"symbol": sym, "why": "现金不足，缩减到 0 手，跳过"})
            continue
        amount = round(size * px, 2)
        fee = engine._fees(amount, COSTS, sell=False)
        cash -= amount + fee
        old = positions.get(sym)
        old_shares = old["shares"] if old else 0
        old_cost = old["cost"] if old else 0.0
        new_shares = old_shares + size
        new_cost = (old_shares * old_cost + amount + fee) / new_shares
        positions[sym] = {"symbol": sym, "shares": new_shares,
                          "cost": round(new_cost, 4), "last_buy_date": day}
        orders.append({"symbol": sym, "side": "buy", "shares": size, "price": px,
                       "amount": round(amount, 2), "fee": round(fee, 2),
                       "reason": intent["reason"]})

    market_value = round(sum(p["shares"] * prices.get(sym, p["cost"])
                             for sym, p in positions.items()), 2)
    equity_after = round(cash + market_value, 2)
    summary = {
        "buy_count": sum(1 for o in orders if o["side"] == "buy"),
        "sell_count": sum(1 for o in orders if o["side"] == "sell"),
        "skipped_count": len(skipped),
        "position_count": len(positions),
        "market_value": market_value,
        "equity": equity_after,
        "total_target": round(total_target, 3),
        "target_scale": round(scale, 3),
    }
    _finalize(day, cash, equity_after, orders, summary, positions)
    return {"date": day.isoformat(), "orders": orders, "skipped": skipped,
            "equity_after": equity_after, "summary": summary}


def _finalize(day, cash, equity, orders, summary, positions=None):
    """把本次复盘的成交、持仓、权益与摘要落库（同一天重跑覆盖权益与摘要）。"""
    con = connect()
    try:
        ensure(con)
        con.execute("BEGIN TRANSACTION")
        for o in orders:
            _write_trade(con, day, o["symbol"], o["side"], o["shares"], o["price"],
                         o["amount"], o["fee"], o["reason"])
        if positions is not None:
            con.execute("DELETE FROM paper_positions")
            for sym, p in positions.items():
                _save_position(con, sym, p["shares"], p["cost"], p["last_buy_date"])
        con.execute(
            "INSERT INTO paper_equity (trade_date, equity) VALUES (?, ?) "
            "ON CONFLICT (trade_date) DO UPDATE SET equity = excluded.equity",
            [day, equity])
        con.execute(
            "INSERT INTO paper_runs (trade_date, summary) VALUES (?, ?) "
            "ON CONFLICT (trade_date) DO UPDATE SET summary = excluded.summary",
            [day, json.dumps(summary, ensure_ascii=False)])
        con.execute("UPDATE paper_account SET cash = ?, last_run = current_timestamp "
                    "WHERE id = 1", [cash])
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def _last_review(con):
    row = con.execute(
        "SELECT summary FROM paper_runs ORDER BY trade_date DESC LIMIT 1").fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def get_state():
    """GET /api/paper 的完整数据；持仓市值与现金相加严格等于 equity。"""
    con = connect()
    try:
        ensure(con)
        acc = _load_account(con)
        positions = _load_positions(con)
        curve = con.execute(
            "SELECT trade_date, equity FROM paper_equity ORDER BY trade_date").fetchall()
        last_review = _last_review(con)
    finally:
        con.close()

    symbols = list(positions)
    quotes = _quote_map(symbols)
    prices = _prices(symbols, quotes)

    items = []
    market_value_total = 0.0
    for sym, p in positions.items():
        q = quotes.get(sym) or {}
        price = prices.get(sym, p["cost"])
        market_value = round(price * p["shares"], 2)
        market_value_total += market_value
        pnl = round((price - p["cost"]) * p["shares"], 2)
        items.append({
            "symbol": sym,
            "code": q.get("code") or (sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym),
            "name": q.get("name") or sym,
            "shares": p["shares"],
            "cost": round(p["cost"], 4),
            "price": round(price, 3),
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": round((price / p["cost"] - 1) * 100, 2) if p["cost"] else None,
        })

    equity = round(acc["cash"] + market_value_total, 2)
    for it in items:
        it["weight"] = round(it["market_value"] / equity, 4) if equity else 0.0

    pnl = round(equity - acc["initial"], 2)
    return {
        "equity": equity,
        "cash": round(acc["cash"], 2),
        "initial": round(acc["initial"], 2),
        "pnl": pnl,
        "pnl_pct": round(pnl / acc["initial"] * 100, 2) if acc["initial"] else None,
        "started_at": _iso(acc["started_at"]),
        "last_run": _iso(acc["last_run"]),
        "positions": items,
        "curve": [{"date": d.isoformat(), "equity": round(float(e), 2)}
                  for d, e in curve],
        "last_review": last_review,
    }


def list_trades():
    con = connect()
    try:
        ensure(con)
        rows = con.execute(
            "SELECT trade_date, symbol, side, shares, price, amount, fee "
            "FROM paper_trades ORDER BY trade_date DESC, id DESC").fetchall()
    finally:
        con.close()
    return [{"date": r[0].isoformat(), "symbol": r[1], "side": r[2],
             "shares": int(r[3]), "price": float(r[4]), "amount": float(r[5]),
             "fee": float(r[6])} for r in rows]


def run_if_due(period=1):
    """距上次运行 ≥ period 天才复盘；供计划任务调用。"""
    period = int(period)
    if period < 1:
        raise ValueError("period 必须为正整数")
    con = connect()
    try:
        ensure(con)
        acc = _load_account(con)
    finally:
        con.close()
    today = date.today()
    last_run = acc["last_run"]
    if last_run is not None:
        days = (today - last_run.date()).days
        if days < period:
            return {"executed": False, "date": today.isoformat(),
                    "last_run": _iso(last_run), "period": period, "days_since": days,
                    "reason": f"距上次复盘 {days} 天，未满 {period} 天，跳过"}
    result = rebalance(today)
    return {"executed": True, **result}
