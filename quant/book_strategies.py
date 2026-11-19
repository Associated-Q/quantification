"""把书里的规则转成逐日打分序列，供回测使用。

与 book_rules.py 的权重一一对应：
  吞噬 ±2、黄昏星/启明星 ∓2、锤子/流星 ±1.5、跳空 ±0.5、
  相对力度 ±1、均线排列 ±1、支撑 +1、道氏结构 ±1.5、成交量确认 ±0.5、斐波那契 +1

每个值只用当日及以前的数据，不含未来信息。
"""
import numpy as np
import pandas as pd


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def book_score(df: pd.DataFrame) -> pd.Series:
    o, c, h, l, v = (df[x].astype(float) for x in ("open", "close", "high", "low", "volume"))
    body = (c - o).abs()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    atr = _atr(df)
    score = pd.Series(0.0, index=df.index)

    # 吞噬
    prev_bear, prev_bull = c.shift(1) < o.shift(1), c.shift(1) > o.shift(1)
    bull_engulf = (c > o) & prev_bear & (c >= o.shift(1)) & (o <= c.shift(1))
    bear_engulf = (c < o) & prev_bull & (c <= o.shift(1)) & (o >= c.shift(1))
    score += bull_engulf * 2.0 - bear_engulf * 2.0

    # 黄昏星 / 启明星
    b1 = (c.shift(2) - o.shift(2)).abs()
    b2 = (c.shift(1) - o.shift(1)).abs()
    mid1 = (o.shift(2) + c.shift(2)) / 2
    evening = (c.shift(2) > o.shift(2)) & (b2 < b1 * 0.6) & (c < o) & (c <= mid1)
    morning = (c.shift(2) < o.shift(2)) & (b2 < b1 * 0.6) & (c > o) & (c >= mid1)
    score += morning * 2.0 - evening * 2.0

    # 锤子 / 流星
    low_pt = pd.concat([o, c], axis=1).min(axis=1)
    high_pt = pd.concat([o, c], axis=1).max(axis=1)
    lower, upper = low_pt - l, h - high_pt
    hammer = (body > 0) & (lower >= 2 * body) & (upper <= 0.5 * body)
    shooting = (body > 0) & (upper >= 2 * body) & (lower <= 0.5 * body)
    score += hammer * np.where(c < ma20, 1.5, 1.0)
    score -= shooting * np.where(c > ma20, 1.5, 1.0)

    # 跳空
    score += (l > h.shift(1)) * 0.5 - (h < l.shift(1)) * 0.5

    # 相对力度
    r = _rsi(c)
    score += (r <= 30) * 1.0 - (r >= 70) * 1.0

    # 均线排列
    score += ((c > ma20) & (ma20 > ma60)) * 1.0
    score -= ((c < ma20) & (ma20 < ma60)) * 1.0

    # 支撑
    support = l.rolling(20).min()
    near_support = (c - support >= 0) & (c - support <= atr)
    score += near_support * 1.0

    # 道氏结构
    rh, ph = h.rolling(10).max(), h.rolling(20).max().shift(10)
    rl, pl = l.rolling(10).min(), l.rolling(20).min().shift(10)
    score += ((rh > ph) & (rl > pl)) * 1.5
    score -= ((rh < ph) & (rl < pl)) * 1.5

    # 成交量确认
    vol_ma = v.rolling(20).mean()
    big_vol = v > vol_ma * 1.2
    score += (big_vol & (c > c.shift(1))) * 0.5
    score -= (big_vol & (c < c.shift(1))) * 0.5

    # 斐波那契回撤
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    span = (hi - lo).replace(0, np.nan)
    retr = (hi - c) / span
    score += ((retr >= 0.382) & (retr <= 0.618)) * 1.0

    return score


def book_entry_exit(df: pd.DataFrame, entry_th=2.0, exit_th=-1.0):
    s = book_score(df)
    return s >= entry_th, s <= exit_th, s


def tf_trend_daily(df: pd.DataFrame):
    """周线/月线均线方向映射回日线：周线 ±0.75，月线 ±1.0。同样只用已完成周期。"""
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    s = d.set_index("trade_date")["close"]

    w = s.resample("W-FRI").last().dropna()
    wm20, wm60 = w.rolling(20).mean(), w.rolling(60).mean()
    wv = pd.Series(np.where(wm20 > wm60, 0.75, -0.75), index=w.index)
    wv = wv.where(wm20.notna() & wm60.notna(), 0.0)

    m = s.resample("ME").last().dropna()
    mm12, mm24 = m.rolling(12).mean(), m.rolling(24).mean()
    mv = pd.Series(np.where(mm12 > mm24, 1.0, -1.0), index=m.index)
    mv = mv.where(mm12.notna() & mm24.notna(), 0.0)

    wv.index = wv.index + pd.Timedelta(days=3)
    mv.index = mv.index + pd.Timedelta(days=1)
    a = wv.reindex(d["trade_date"], method="ffill").fillna(0.0)
    b = mv.reindex(d["trade_date"], method="ffill").fillna(0.0)
    return (a + b).to_numpy()


def weekly_wave_daily(df: pd.DataFrame, pct=0.05):
    """把周线浪型判断映射回日线。

    每周只用该周及以前的周线数据计算，标注值从下一周首个交易日起生效，避免前视。
    """
    import wave as wv

    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    w = (d.set_index("trade_date")
           .resample("W-FRI")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna())
    if len(w) < 8:
        return pd.Series(0.0, index=d["trade_date"])

    vals = pd.Series(0.0, index=w.index)
    for i in range(6, len(w)):
        sub = w.iloc[:i + 1].reset_index()
        sub.columns = ["trade_date", "open", "high", "low", "close", "volume"]
        r = wv.analyze_wave(sub[["high", "low", "close", "open", "volume"]], pct)
        if r.get("valid") and r.get("sid"):
            vals.iloc[i] = r["sid"] * r.get("weight", 1.5) * 0.75

    vals.index = vals.index + pd.Timedelta(days=3)     # 周末收盘 → 下周一才可用
    return vals.reindex(d["trade_date"], method="ffill").fillna(0.0).to_numpy()
