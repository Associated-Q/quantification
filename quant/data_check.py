"""只读扫描日线数据：自然日缺口、零成交量、板块涨跌幅超限。

用法：python -I data_check.py [--db market.duckdb] [--report data_check_report.csv]
数据库名、表名和日期必须有效，发现非法数据不静默跳过。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_source
from engine import infer_limit, is_holiday_gap, tick_size

COLS = ["symbol", "anomaly", "trade_date", "previous_date", "gap_days",
        "volume", "close", "previous_close", "change_pct", "limit_pct",
        "gap_type", "detail"]


def scan(df: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    required = {"symbol", "trade_date", "open", "close", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"日线数据缺少字段: {sorted(missing)}")
    if max_gap_days < 1:
        raise ValueError("max_gap_days 必须 >= 1")
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    if d["trade_date"].isna().any():
        raise ValueError("存在无法解析的 trade_date")
    if d.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("存在重复 symbol/trade_date")
    d = d.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    records = []
    for symbol, g in d.groupby("symbol", sort=True):
        names = g["name"].dropna().unique() if "name" in g else []
        if len(names) > 1:
            raise ValueError(f"{symbol} 有多个名称，无法判定历史 ST 状态")
        limit = infer_limit(symbol, names[0] if len(names) else None)
        tick = tick_size(symbol)
        dates = g["trade_date"]
        previous_date = dates.shift()
        gaps = (dates - previous_date).dt.days
        close = pd.to_numeric(g["close"], errors="coerce")
        previous_close = close.shift()
        change = close / previous_close - 1
        volume = pd.to_numeric(g["volume"], errors="coerce")
        # 指数无涨跌停限制；ETF 最小报价单位 0.001，股票 0.01。
        if tick is None:
            breach = pd.Series(False, index=g.index)
        else:
            decimals = 3 if tick < 0.005 else 2
            upper = (previous_close * (1 + limit)).round(decimals)
            lower = (previous_close * (1 - limit)).round(decimals)
            breach = close.gt(upper + 1e-10) | close.lt(lower - 1e-10)
        for i in range(len(g)):
            common = {
                "symbol": symbol, "trade_date": dates.iloc[i],
                "previous_date": previous_date.iloc[i], "gap_days": np.nan,
                "volume": volume.iloc[i], "close": close.iloc[i],
                "previous_close": previous_close.iloc[i],
                "change_pct": change.iloc[i], "limit_pct": np.nan,
                "gap_type": "", "detail": "",
            }
            if pd.notna(gaps.iloc[i]) and gaps.iloc[i] > max_gap_days:
                holiday = is_holiday_gap(previous_date.iloc[i], dates.iloc[i])
                records.append({**common, "anomaly": "gap", "gap_days": int(gaps.iloc[i]),
                                "gap_type": "holiday" if holiday else "suspension",
                                "detail": ("节假日休市（非停牌）" if holiday
                                           else "非节假日缺口，疑似停牌或数据缺失")})
            if pd.notna(volume.iloc[i]) and volume.iloc[i] == 0:
                records.append({**common, "anomaly": "zero_volume",
                                "detail": "当日成交量为零（手）"})
            if breach.iloc[i]:
                records.append({**common, "anomaly": "limit_breach", "limit_pct": limit,
                                "detail": "收盘较前收盘涨跌幅超板块限价；前复权可能产生假阳性"})
    return pd.DataFrame(records, columns=COLS)


def summarize(df: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    counts = report.groupby(["symbol", "anomaly"]).size().unstack(fill_value=0)
    for kind in ("gap", "zero_volume", "limit_breach"):
        if kind not in counts:
            counts[kind] = 0
    summary = df.groupby("symbol").agg(
        bars=("trade_date", "size"), first_date=("trade_date", "min"),
        last_date=("trade_date", "max")).join(counts[["gap", "zero_volume", "limit_breach"]])
    summary = summary.fillna({"gap": 0, "zero_volume": 0, "limit_breach": 0})
    summary["anomalies"] = summary[["gap", "zero_volume", "limit_breach"]].sum(axis=1)
    return summary.reset_index().sort_values("symbol").reset_index(drop=True)


def load(db: str, table: str = "kline_daily") -> pd.DataFrame:
    import duckdb
    if not table.isidentifier():
        raise ValueError(f"非法 DuckDB 表名: {table!r}")
    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(f'SELECT * FROM "{table}" ORDER BY symbol, trade_date').df()
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=data_source.DB)
    ap.add_argument("--table", default="kline_daily")
    ap.add_argument("--max-gap-days", type=int, default=5)
    ap.add_argument("--report", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "data_check_report.csv"))
    args = ap.parse_args()
    df = load(args.db, args.table)
    report = scan(df, args.max_gap_days)
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    summary = summarize(df, report)
    print(f"扫描 {len(summary)} 只标的、{len(df)} 根日线；异常记录 {len(report)} 条")
    gaps = report[report["anomaly"] == "gap"]
    if len(gaps):
        n_holiday = int((gaps["gap_type"] == "holiday").sum())
        print(f"缺口 {len(gaps)} 条：节假日休市 {n_holiday} 条，"
              f"疑似停牌/数据缺失 {len(gaps) - n_holiday} 条")
    print(summary.to_string(index=False))
    print(f"报告: {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
