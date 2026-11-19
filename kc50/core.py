"""科创50（000688）日线数据层：双源获取、完整性核验、指标与已确认转折点。

数据源
  * 东方财富 push2his —— OHLC / 成交量 / 成交额
  * 腾讯 fqkline      —— OHLC / 成交量，用于交叉核对

所有函数返回的序列均为**因果**（t 时点只包含 <= t 的信息）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
UA = {"User-Agent": "Mozilla/5.0"}

EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

KC50_SECID = "1.000688"
KC50_TX = "sh000688"
CAL_SECID = "1.000001"      # 上证指数，仅用于取交易日历


def _json(url: str, params: dict, timeout: int = 30, tries: int = 3) -> dict:
    err = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5)
    raise RuntimeError(f"行情接口失败 {url}: {err}")


def fetch_eastmoney(secid: str, beg: str, end: str) -> pd.DataFrame:
    """东财日线（前复权）。列：date open high low close volume amount"""
    j = _json(EM_URL, {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": "101",
        "fqt": "1",
        "beg": beg.replace("-", ""),
        "end": end.replace("-", ""),
        "lmt": "5000",
    })
    data = j.get("data")
    if not data or not data.get("klines"):
        raise RuntimeError(f"东财无数据: {secid} {beg}~{end}")
    rows = [ln.split(",") for ln in data["klines"]]
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "close", "high", "low", "volume", "amount", "amplitude"],
    )
    for c in ["open", "close", "high", "low", "volume", "amount", "amplitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return (df[["date", "open", "high", "low", "close", "volume", "amount"]]
            .sort_values("date").reset_index(drop=True))


def fetch_tencent(symbol: str, beg: str, end: str) -> pd.DataFrame:
    """腾讯日线（前复权）。列：date open high low close volume"""
    j = _json(TX_URL, {"param": f"{symbol},day,{beg},{end},2000,qfq"})
    if j.get("code") not in (0, None):
        raise RuntimeError(f"腾讯接口报错: {j.get('msg')}")
    node = j["data"][symbol]
    bars = node.get("qfqday") or node.get("day") or []
    if not bars:
        raise RuntimeError(f"腾讯无数据: {symbol} {beg}~{end}")
    # 腾讯字段顺序 [日期, 开, 收, 高, 低, 量]
    df = pd.DataFrame([b[:6] for b in bars],
                      columns=["date", "open", "close", "high", "low", "volume"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return (df[["date", "open", "high", "low", "close", "volume"]]
            .sort_values("date").reset_index(drop=True))


def fetch_calendar(beg: str, end: str) -> pd.DatetimeIndex:
    cal = fetch_eastmoney(CAL_SECID, beg, end)
    return pd.DatetimeIndex(cal["date"])


def verify(df: pd.DataFrame, cal: pd.DatetimeIndex, tx: pd.DataFrame | None = None) -> dict:
    """完整性核验。返回问题清单，空 dict 表示通过。"""
    issues: dict[str, list] = {}

    dup = df["date"][df["date"].duplicated()].dt.strftime("%Y-%m-%d").tolist()
    if dup:
        issues["重复日期"] = dup

    bad_ohlc = df[(df["high"] < df["low"])
                  | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-6)
                  | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-6)]
    if len(bad_ohlc):
        issues["OHLC自相矛盾"] = bad_ohlc["date"].dt.strftime("%Y-%m-%d").tolist()

    zero_vol = df[df["volume"] <= 0]
    if len(zero_vol):
        issues["成交量为0"] = zero_vol["date"].dt.strftime("%Y-%m-%d").tolist()

    nulls = df[df[["open", "high", "low", "close"]].isna().any(axis=1)]
    if len(nulls):
        issues["价格缺失"] = nulls["date"].dt.strftime("%Y-%m-%d").tolist()

    dset = set(df["date"])
    missing = [d for d in cal if d not in dset]
    if missing:
        issues["缺失交易日"] = [d.strftime("%Y-%m-%d") for d in missing]

    extra = [d for d in dset if d not in set(cal)]
    if extra:
        issues["非交易日出现K线"] = [d.strftime("%Y-%m-%d") for d in sorted(extra)]

    pct = df["close"].pct_change().abs()
    jump = df[pct > 0.10]
    if len(jump):
        issues["单日涨跌超10%"] = jump["date"].dt.strftime("%Y-%m-%d").tolist()

    if tx is not None:
        m = df.merge(tx, on="date", suffixes=("", "_tx"), how="inner")
        diff = m[(m["close"] - m["close_tx"]).abs() > 0.02]
        if len(diff):
            issues["双源收盘价不一致"] = (
                diff.assign(delta=(diff["close"] - diff["close_tx"]).round(3))
                [["date", "close", "close_tx", "delta"]]
                .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                .to_dict("records")
            )
    return issues


def load(beg: str = "2023-01-01", end: str | None = None, force: bool = False,
         verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """加载并核验。返回 (df, 核验报告)。结果缓存在 kc50/data/。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = os.path.join(DATA_DIR, "kc50_daily.csv")
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, parse_dates=["date"])
        if df["date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=4):
            report = verify(df, fetch_calendar(beg, end), fetch_tencent(KC50_TX, beg, end))
            return df, report

    df = fetch_eastmoney(KC50_SECID, beg, end)
    tx = fetch_tencent(KC50_TX, beg, end)
    cal = fetch_calendar(beg, end)
    report = verify(df, cal, tx)
    df.to_csv(path, index=False)
    if verbose:
        last = df.iloc[-1]
        print(f"[data] {len(df)} 根K线  {df['date'].iloc[0]:%Y-%m-%d} ~ "
              f"{last['date']:%Y-%m-%d}  收 {last['close']:.2f}")
        if report:
            for k, v in report.items():
                print(f"[data][警告] {k}: {v[:6]}{' ...' if len(v) > 6 else ''}")
        else:
            print("[data] 核验通过：无重复、无缺失交易日、无OHLC矛盾、双源收盘价一致")
    return df, report


# ---------------------------------------------------------------- indicators

def add_indicators(df: pd.DataFrame, atr_n: int = 14, ma_n: int = 20) -> pd.DataFrame:
    out = df.copy()
    h, l, c = out["high"], out["low"], out["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1.0 / atr_n, adjust=False).mean()
    out["ma20"] = c.rolling(ma_n).mean()
    out["ma20_slope5"] = out["ma20"] / out["ma20"].shift(5) - 1.0
    out["vol5"] = out["volume"].rolling(5).mean()
    out["vol20"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["vol5"] / out["vol20"]
    out["ret1"] = c.pct_change()
    out["ret5"] = c.pct_change(5)
    return out


# ---------------------------------------------------------------- pivots

def find_pivots(df: pd.DataFrame, k: int = 2, atr_mult: float = 1.0) -> pd.DataFrame:
    """5根K线分形转折点 + ATR过滤，因果构建。

    高低点需严格高于/低于左右各 k 根K线；右侧 k 根完成后（idx=j+k）才确认。
    相邻转折点价格差不足 atr_mult * ATR14(确认时) 的丢弃。

    返回列：idx, kind('H'/'L'), date, price, confirm_idx, confirm_date
    """
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    atr = df["atr14"].to_numpy(float)
    dates = df["date"].to_numpy()
    n = len(df)

    raw = []
    for j in range(k, n - k):
        h, lo = highs[j], lows[j]
        if all(h > highs[j - i] for i in range(1, k + 1)) and \
           all(h > highs[j + i] for i in range(1, k + 1)):
            raw.append((j, "H"))
        if all(lo < lows[j - i] for i in range(1, k + 1)) and \
           all(lo < lows[j + i] for i in range(1, k + 1)):
            raw.append((j, "L"))
    raw.sort()

    legs: list[dict] = []          # 当前有效转折点栈
    history: list[dict] = []       # 全部曾生效过的转折点（含被替换者）
    for j, kind in raw:
        px = highs[j] if kind == "H" else lows[j]
        cidx = min(j + k, n - 1)
        thr = atr_mult * (atr[cidx] if np.isfinite(atr[cidx]) else 0.0)
        rec = {"idx": j, "kind": kind, "price": px, "confirm_idx": cidx,
               "superseded_at": np.nan}
        if not legs:
            legs.append(rec)
            history.append(rec)
            continue
        prev = legs[-1]
        if kind == prev["kind"]:
            more_extreme = (px > prev["price"]) if kind == "H" else (px < prev["price"])
            if more_extreme:
                prev["superseded_at"] = cidx
                legs.pop()
                legs.append(rec)
                history.append(rec)
        elif abs(px - prev["price"]) >= thr:
            legs.append(rec)
            history.append(rec)

    out = pd.DataFrame(history)
    out["date"] = pd.to_datetime(dates[out["idx"].to_numpy()])
    out["confirm_date"] = pd.to_datetime(dates[out["confirm_idx"].to_numpy()])
    out["atr"] = atr[out["confirm_idx"].to_numpy()]
    return out.sort_values("idx").reset_index(drop=True)


def pivots_asof(piv: pd.DataFrame, t: int) -> pd.DataFrame:
    """t 时点（含）可用的转折点：已确认且尚未被后续更极端的同类点替换。"""
    m = (piv["confirm_idx"] <= t) & (piv["superseded_at"].isna() | (piv["superseded_at"] > t))
    return piv[m].reset_index(drop=True)
