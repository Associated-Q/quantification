"""从书里蒸馏出的可回测信号。

每个信号都必须能落到日线 OHLCV 上，且不含未来数据。
来源标注便于回溯：b1=日本蜡烛图技术新解，b2=期货市场技术分析，b5=艾略特波浪理论。
"""
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ma_trend(df: pd.DataFrame, fast: int = 20, slow: int = 60):
    """b2：均线多头排列 + 价在短均线之上，跌破短均线离场。"""
    mf, ms = df["close"].rolling(fast).mean(), df["close"].rolling(slow).mean()
    entry = (mf > ms) & (df["close"] > mf)
    exit_ = (df["close"] < mf) | (mf < ms)
    return entry, exit_


def breakout(df: pd.DataFrame, n: int = 20, vol_mult: float = 1.5):
    """b2：突破前 n 日高点且放量，跌破 10 日均线离场。"""
    hi = df["close"].rolling(n).max().shift(1)
    vol_ma = df["volume"].rolling(n).mean()
    entry = (df["close"] > hi) & (df["volume"] > vol_ma * vol_mult)
    exit_ = df["close"] < df["close"].rolling(10).mean()
    return entry, exit_


def bullish_engulfing(df: pd.DataFrame, trend_ma: int = 60):
    """b1：看涨吞没，且处于中期均线之上（形态需要趋势背景）。"""
    o, c = df["open"], df["close"]
    prev_bear = c.shift(1) < o.shift(1)
    engulf = (c > o) & prev_bear & (c >= o.shift(1)) & (o <= c.shift(1))
    entry = engulf & (c > c.rolling(trend_ma).mean())
    exit_ = c < c.rolling(5).mean()
    return entry, exit_


STRATEGIES = {"ma_trend": ma_trend, "breakout": breakout, "engulfing": bullish_engulfing}


def buy_hold(df: pd.DataFrame):
    entry = pd.Series(True, index=df.index)
    exit_ = pd.Series(False, index=df.index)
    return entry, exit_
