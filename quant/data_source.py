"""日线数据源：腾讯财经 fqkline 接口（零鉴权，实测不封 IP）。

腾讯字段顺序为 [日期, 开, 收, 高, 低, 量]，与通行的 OHLC 顺序不同，勿直接套用。
"""
import os
import duckdb
import pandas as pd
import requests

URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SINA_URL = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "CN_MarketData.getKLineData")
HEADERS = {"User-Agent": "Mozilla/5.0"}
SINA_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market.duckdb")

COLS = ["trade_date", "open", "close", "high", "low", "volume"]


def to_symbol(code: str) -> str:
    code = code.strip()
    if code[:2] in ("sh", "sz", "bj"):
        return code
    if code[0] in "659":
        return "sh" + code
    if code[0] in "4278":
        return "bj" + code
    return "sz" + code


def fetch_kline(code: str, period: str = "day", count: int = 800,
                adjust: str = "qfq") -> pd.DataFrame:
    """period: day / week / month。返回前复权 K 线。"""
    sym = to_symbol(code)
    r = requests.get(URL, params={"param": f"{sym},{period},,,{count},{adjust}"},
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    node = r.json()["data"][sym]
    bars = node.get(f"{adjust}{period}") or node.get(period) or []
    if not bars:
        raise ValueError(f"{sym} 无 {period} K 线数据（接口返回空）")
    df = pd.DataFrame([b[:6] for b in bars], columns=COLS)
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "symbol", sym)
    return df.dropna(subset=["open", "close"]).reset_index(drop=True)


def fetch_daily(code: str, count: int = 800, adjust: str = "qfq") -> pd.DataFrame:
    return fetch_kline(code, "day", count, adjust)


def fetch_kline_history(code: str, period: str = "day", count: int = 1600,
                        adjust: str = "qfq", page: int = 800) -> pd.DataFrame:
    """按 end 日期向前翻页，取更长的历史；只读，不写库。

    腾讯接口对大 count 会截断（不同标的截断长度还不一样），但带 end 日期时
    每页能稳定返回 page 根，所以用「上一页最早日期 - 1 天」继续往前翻。
    """
    sym = to_symbol(code)
    frames = []
    end = None
    for _ in range(4):
        param = f"{sym},{period},,{end or ''},{page},{adjust}"
        r = requests.get(URL, params={"param": param}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        node = r.json().get("data", {}).get(sym) or {}
        bars = node.get(f"{adjust}{period}") or node.get(period) or []
        if not bars:
            break
        df = pd.DataFrame([b[:6] for b in bars], columns=COLS)
        for c in COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.insert(0, "symbol", sym)
        df = df.dropna(subset=["open", "close"])
        if df.empty:
            break
        frames.append(df)
        if sum(len(x) for x in frames) >= count or len(df) < page:
            break
        first = str(df["trade_date"].iloc[0])[:10]
        end = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not frames:
        raise ValueError(f"{sym} 无 {period} K 线数据（接口返回空）")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
    return out.tail(count).reset_index(drop=True)


def save(df: pd.DataFrame, db: str = DB, table: str = "kline_daily") -> int:
    con = duckdb.connect(db)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            symbol VARCHAR, trade_date DATE,
            open DOUBLE, close DOUBLE, high DOUBLE, low DOUBLE, volume DOUBLE,
            PRIMARY KEY (symbol, trade_date))
    """)
    con.execute(f"DELETE FROM {table} WHERE symbol = ?", [df["symbol"].iloc[0]])
    con.execute(f"INSERT INTO {table} SELECT symbol, CAST(trade_date AS DATE), "
                "open, close, high, low, volume FROM df")
    n = con.execute(f"SELECT count(*) FROM {table} WHERE symbol = ?",
                    [df["symbol"].iloc[0]]).fetchone()[0]
    con.close()
    return n


def fetch_minute(code: str, scale: int = 5, datalen: int = 1023) -> pd.DataFrame:
    """分钟级 K 线（新浪，不复权）。scale 取值 5/15/30/60。"""
    if scale not in (5, 15, 30, 60):
        raise ValueError(f"不支持的分钟周期: {scale}")
    sym = to_symbol(code)
    r = requests.get(SINA_URL, params={"symbol": sym, "scale": str(scale),
                                       "ma": "no", "datalen": str(datalen)},
                     headers=SINA_HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"{sym} 无 {scale} 分钟数据")
    df = pd.DataFrame([{"trade_time": d["day"], "open": float(d["open"]),
                        "high": float(d["high"]), "low": float(d["low"]),
                        "close": float(d["close"]), "volume": float(d["volume"])}
                       for d in data])
    df.insert(0, "symbol", sym)
    return df


if __name__ == "__main__":
    import sys
    codes = sys.argv[1:] or ["510300", "600519"]
    for c in codes:
        d = fetch_daily(c)
        n = save(d)
        print(f"{to_symbol(c)}: {n} bars  {d['trade_date'].iloc[0]} -> {d['trade_date'].iloc[-1]}  "
              f"last_close={d['close'].iloc[-1]}")
