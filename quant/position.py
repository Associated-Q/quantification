"""持仓与账户资金管理。

表（与 web/app.py 的 /api/positions、/api/account 对接）：
    positions(symbol PK, shares INT, cost DOUBLE, last_buy_date DATE, updated_at TIMESTAMP)
    account(id PK=1, cash DOUBLE)

写入的表只有 positions 与 account（首次调用时自动建表）。
市价取自腾讯实时行情；行情缺失时价格与派生字段为 None，不静默填假数据。
"""
import math
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_source
import realtime

DB = data_source.DB

_COLS = "symbol, shares, cost, last_buy_date, updated_at"


def connect(read_only=False, retries=30, delay=0.1):
    """连接 DuckDB。8000 端口实例可能短暂持有写锁，这里做有限重试。"""
    last = None
    for _ in range(retries):
        try:
            return duckdb.connect(DB, read_only=read_only)
        except Exception as e:  # duckdb.IOException: Could not set lock on file
            last = e
            msg = str(e).lower()
            if read_only or not ("lock" in msg or "conflict" in msg or "being used" in msg):
                raise
            time.sleep(delay)
    raise last


def ensure(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol VARCHAR PRIMARY KEY,
            shares INTEGER NOT NULL,
            cost DOUBLE NOT NULL,
            last_buy_date DATE,
            updated_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY,
            cash DOUBLE NOT NULL
        )
    """)


def normalize_symbol(code):
    """校验证券代码并返回带前缀的 symbol；非法时抛 ValueError。"""
    value = str(code).strip()
    if not value:
        raise ValueError("证券代码不能为空")
    if value[:2] in ("sh", "sz", "bj"):
        if len(value) != 8 or not value[2:].isdigit():
            raise ValueError(f"无效证券代码: {code}")
        return value
    if not value.isdigit() or len(value) != 6:
        raise ValueError(f"无效证券代码: {code}")
    return data_source.to_symbol(value)


def _validate_shares(shares):
    if isinstance(shares, bool) or not isinstance(shares, (int, float)):
        raise ValueError("shares 必须为正整数且是 100 的整数倍")
    if not math.isfinite(float(shares)) or float(shares) != int(shares):
        raise ValueError("shares 必须为正整数且是 100 的整数倍")
    shares = int(shares)
    if shares <= 0 or shares % 100 != 0:
        raise ValueError("shares 必须为正整数且是 100 的整数倍")
    return shares


def _validate_cost(cost):
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        raise ValueError("cost 必须为正数") from None
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("cost 必须为正数")
    return cost


def upsert_position(code, shares, cost):
    """新增或覆盖一笔持仓；返回规范化后的 symbol。"""
    sym = normalize_symbol(code)
    shares = _validate_shares(shares)
    cost = _validate_cost(cost)
    con = connect()
    try:
        ensure(con)
        con.execute("""
            INSERT INTO positions (symbol, shares, cost, last_buy_date, updated_at)
            VALUES (?, ?, ?, current_date, current_timestamp)
            ON CONFLICT (symbol) DO UPDATE SET
                shares = excluded.shares,
                cost = excluded.cost,
                last_buy_date = excluded.last_buy_date,
                updated_at = excluded.updated_at
        """, [sym, shares, cost])
    finally:
        con.close()
    return sym


def remove_position(sym):
    sym = normalize_symbol(sym)
    con = connect()
    try:
        ensure(con)
        con.execute("DELETE FROM positions WHERE symbol = ?", [sym])
    finally:
        con.close()
    return sym


def list_positions(with_quotes=True):
    """列出持仓并按实时行情计算市值/盈亏（pnl 未扣卖出费用）。

    返回字段：symbol, code, name, shares, cost, last_buy_date,
              price, market_value, pnl, pnl_pct（百分比）
    行情拿不到时 price/market_value/pnl/pnl_pct 均为 None。
    """
    con = connect()
    try:
        ensure(con)
        rows = con.execute(
            f"SELECT {_COLS} FROM positions ORDER BY symbol").fetchall()
    finally:
        con.close()

    out = []
    for sym, shares, cost, last_buy, updated in rows:
        out.append({
            "symbol": sym,
            "code": sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym,
            "name": sym,
            "shares": int(shares),
            "cost": float(cost),
            "last_buy_date": str(last_buy)[:10] if last_buy else None,
            "updated_at": str(updated) if updated else None,
            "price": None, "market_value": None, "pnl": None, "pnl_pct": None,
        })

    if not with_quotes or not out:
        return out

    try:
        quotes = realtime.fetch_quotes([r["symbol"] for r in out])
    except Exception:
        quotes = {}
    for r in out:
        q = quotes.get(r["symbol"]) or {}
        if q:
            r["name"] = q.get("name") or r["name"]
            r["code"] = q.get("code") or r["code"]
        price = q.get("price")
        if price is None:
            continue
        r["price"] = float(price)
        r["market_value"] = round(float(price) * r["shares"], 2)
        r["pnl"] = round((float(price) - r["cost"]) * r["shares"], 2)
        r["pnl_pct"] = (round((float(price) - r["cost"]) / r["cost"] * 100, 2)
                        if r["cost"] else None)
    return out


def get_cash():
    con = connect()
    try:
        ensure(con)
        row = con.execute("SELECT cash FROM account WHERE id = 1").fetchone()
    finally:
        con.close()
    return float(row[0]) if row else 0.0


def set_cash(v):
    try:
        cash = float(v)
    except (TypeError, ValueError):
        raise ValueError("cash 必须为非负数") from None
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash 必须为非负数")
    con = connect()
    try:
        ensure(con)
        con.execute("""
            INSERT INTO account (id, cash) VALUES (1, ?)
            ON CONFLICT (id) DO UPDATE SET cash = excluded.cash
        """, [cash])
    finally:
        con.close()
    return cash
