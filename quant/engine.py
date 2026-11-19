"""A股日线回测内核。

设计要点（都不能省）：
- 信号在收盘后产生、次日开盘成交，避免用当日收盘价成交的前视偏差；
- T+1：当日开盘买入的仓位，最早次日开盘才能卖出（循环结构天然满足）；
- 涨停不可买、跌停不可卖（按前收盘价推算）；
- 佣金双边、印花税仅卖出、过户费、滑点；
- 仓位按 ATR 风险单位法：单笔风险 = 权益 × risk_pct，手数 = 风险额 / (atr_stop × ATR)。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    commission: float = 0.00025      # 万 2.5，双边
    min_commission: float = 5.0
    stamp_tax: float = 0.0005        # 印花税，仅卖出
    transfer_fee: float = 0.00001
    slippage: float = 0.001
    lot: int = 100


def _fees(amount: float, c: Costs, sell: bool) -> float:
    fee = max(amount * c.commission, c.min_commission) + amount * c.transfer_fee
    if sell:
        fee += amount * c.stamp_tax
    return fee


# 2023-2026 年 A 股工作日休市日（仅列周一至周五；周末天然休市）。
# 用于把长假缺口与真正的停牌/数据缺失区分开。
CN_MARKET_HOLIDAYS = frozenset(pd.Timestamp(d).date() for d in (
    # 2023
    "2023-01-02", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26",
    "2023-01-27", "2023-04-05", "2023-05-01", "2023-05-02", "2023-05-03",
    "2023-06-22", "2023-06-23", "2023-09-29", "2023-10-02", "2023-10-03",
    "2023-10-04", "2023-10-05", "2023-10-06",
    # 2024
    "2024-01-01", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16", "2024-04-04", "2024-04-05", "2024-05-01",
    "2024-05-02", "2024-05-03", "2024-06-10", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07",
    # 2025
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-03", "2025-02-04", "2025-04-04", "2025-05-01", "2025-05-02",
    "2025-05-05", "2025-06-02", "2025-10-01", "2025-10-02", "2025-10-03",
    "2025-10-06", "2025-10-07", "2025-10-08",
    # 2026
    "2026-01-01", "2026-01-02", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-02-19", "2026-02-20", "2026-02-23", "2026-04-06", "2026-05-01",
    "2026-05-04", "2026-05-05", "2026-06-19", "2026-09-25", "2026-10-01",
    "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",
))


def is_holiday_gap(previous_date, trade_date) -> bool:
    """两根 K 线之间的自然日缺口是否完全由周末/法定休市造成（非停牌）。"""
    prev, nxt = pd.Timestamp(previous_date), pd.Timestamp(trade_date)
    if nxt <= prev:
        return False
    days = pd.date_range(prev + pd.Timedelta(days=1),
                         nxt - pd.Timedelta(days=1), freq="D")
    weekdays = [d for d in days if d.weekday() < 5]
    return all(d.date() in CN_MARKET_HOLIDAYS for d in weekdays)


def is_index(code) -> bool:
    """交易所指数代码（无涨跌停限制）。"""
    text = str(code).strip().lower()
    return text.startswith(("sh000", "sz399", "bj899"))


def tick_size(code) -> float | None:
    """最小报价单位：ETF/LOF 为 0.001，股票为 0.01，指数无涨跌停（None）。"""
    if is_index(code):
        return None
    text = str(code).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6 and digits.startswith(("5", "15", "16", "18")):
        return 0.001
    return 0.01


def infer_limit(code, name: str | None = None) -> float:
    """按证券代码及名称推断涨跌停比例。

    创业板/科创板（含 ST）为 20%；主板 ST 为 5%；其余主板/ETF 为 10%。
    """
    text = str(code).strip().upper()
    symbol_name = "" if name is None else str(name).upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    # 创业板、科创板股票及对应 ETF 为 20%（该板块 ST 股同样 20%）；588/589 为科创板 ETF。
    if digits.startswith(("300", "301", "688", "689", "588", "589")):
        return 0.20
    if (len(digits) == 6 and digits.startswith(("5", "15", "16", "18"))
            and ("创业板" in symbol_name or "科创" in symbol_name)):
        return 0.20
    if "ST" in text or "ST" in symbol_name:
        return 0.05
    # 代码段无法区分 ETF 跟踪的指数；无名称时用已知的创业板 ETF 代码兜底。
    if digits in ("159915", "159949"):
        return 0.20
    # 主板、其余 ETF，以及无法由代码识别的标的使用主板默认比例。
    return 0.10


def run(df: pd.DataFrame, entry: pd.Series, exit_: pd.Series, atr: pd.Series,
        equity0: float = 100_000, risk_pct: float = 0.005, atr_stop: float | None = 2.0,
        limit_pct: float | None = None, costs: Costs = Costs(), size_pct: float | None = None,
        limit_ticks: int = 0,
        max_gap_days: int | None = 5, max_volume_pct: float | None = 0.05):
    o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df else np.full(len(df), np.nan)
    a = atr.to_numpy(float)
    ent = entry.fillna(False).to_numpy(bool)
    ext = exit_.fillna(False).to_numpy(bool)
    dates = df["trade_date"].to_numpy()
    date_index = pd.to_datetime(df["trade_date"])
    if date_index.hasnans or not date_index.is_monotonic_increasing or date_index.duplicated().any():
        raise ValueError("trade_date 必须有效、按日期递增且不能重复")
    after_gap = np.zeros(len(df), dtype=bool)
    if max_gap_days is not None:
        gaps = date_index.diff().dt.days.to_numpy()
        for i in range(1, len(df)):
            if gaps[i] > max_gap_days and not is_holiday_gap(date_index.iloc[i - 1],
                                                            date_index.iloc[i]):
                after_gap[i] = True
    symbol = None
    if "symbol" in df and not df["symbol"].dropna().empty:
        symbols = df["symbol"].dropna().unique()
        if len(symbols) != 1:
            raise ValueError("engine.run 每次只能回测一个 symbol")
        symbol = symbols[0]
    if limit_pct is None:
        if symbol is None:
            raise ValueError("自动推断涨跌停比例需要非空 symbol；或显式传入 limit_pct")
        names = df["name"].dropna().unique() if "name" in df else []
        if len(names) > 1:
            raise ValueError("engine.run 无法从多个名称推断 ST 状态；请显式传入 limit_pct")
        limit_pct = infer_limit(symbol, names[0] if len(names) else None)
    tick = tick_size(symbol) if symbol is not None else 0.01
    decimals = 3 if tick is not None and tick < 0.005 else 2

    cash, pos, trades, equity = equity0, None, [], []

    for i in range(len(df)):
        # 缺口前的收盘信号无法跨停牌区间执行；复牌首根 K 线仅估值/更新止损。
        if i > 0 and not after_gap[i]:
            prev_close = c[i - 1]
            up = round(prev_close * (1 + limit_pct), decimals)
            down = round(prev_close * (1 - limit_pct), decimals)
            # 默认严格比较：价格不可能真的越过限价，任何「越过」都是前复权取整误差。
            # 严格比较偏保守——只会少成交，不会凭空造出成交。
            price_tick = 10 ** (-decimals)
            up_tol = up + price_tick * limit_ticks
            down_tol = down - price_tick * limit_ticks

            if pos is not None and (ext[i - 1] or (atr_stop is not None and o[i] <= pos["stop"])):
                if o[i] > down_tol + 1e-9:                       # 跌停开盘卖不出
                    px = o[i] - costs.slippage
                    amt = pos["shares"] * px
                    cash += amt - _fees(amt, costs, sell=True)
                    stopped = atr_stop is not None and o[i] <= pos["stop"]
                    trades.append({
                        "entry_date": dates[pos["i"]], "exit_date": dates[i],
                        "entry_px": pos["px"], "exit_px": px, "shares": pos["shares"],
                        "pnl": pos["shares"] * (px - pos["px"])
                               - _fees(amt, costs, True) - pos["fee"],
                        "reason": "stop" if stopped else "signal"})
                    pos = None

            if pos is None and ent[i - 1] and o[i] < up_tol - 1e-9:   # 涨停开盘买不进
                px = o[i] + costs.slippage
                if size_pct is not None:
                    size = int(cash * size_pct / px // costs.lot) * costs.lot
                else:
                    unit = (atr_stop or 2.0) * a[i - 1]
                    ok = unit > 0 and not np.isnan(unit)
                    size = int((equity0 * risk_pct) / unit // costs.lot) * costs.lot if ok else 0
                if max_volume_pct is not None:
                    # 用信号日（i-1）成交量，避免用执行日全天成交量造成前视；
                    # 腾讯日线 volume 单位为手，成交上限必须换算为股再取整手。
                    if not np.isfinite(v[i - 1]) or v[i - 1] <= 0:
                        size = 0
                    else:
                        cap = int(v[i - 1] * 100 * max_volume_pct // costs.lot) * costs.lot
                        size = min(size, cap)
                while size > 0 and size * px + _fees(size * px, costs, False) > cash:
                    size -= costs.lot
                if size > 0:
                    amt = size * px
                    fee = _fees(amt, costs, False)
                    cash -= amt + fee
                    pos = {"i": i, "shares": size, "px": px, "fee": fee,
                           "stop": px - atr_stop * a[i - 1] if atr_stop is not None else -np.inf}

        if pos is not None and atr_stop is not None:
            pos["stop"] = max(pos["stop"], c[i] - atr_stop * a[i])
        equity.append(cash + (pos["shares"] * c[i] if pos else 0.0))

    if pos is not None:
        px = c[-1] - costs.slippage
        amt = pos["shares"] * px
        cash += amt - _fees(amt, costs, sell=True)
        trades.append({"entry_date": dates[pos["i"]], "exit_date": dates[-1],
                       "entry_px": pos["px"], "exit_px": px, "shares": pos["shares"],
                       "pnl": pos["shares"] * (px - pos["px"])
                              - _fees(amt, costs, True) - pos["fee"], "reason": "eod"})
        equity[-1] = cash

    return (pd.DataFrame({"trade_date": dates, "equity": equity}),
            pd.DataFrame(trades, columns=["entry_date", "exit_date", "entry_px",
                                          "exit_px", "shares", "pnl", "reason"]))


def metrics(eq: pd.DataFrame, trades: pd.DataFrame, risk_free: float = 0.015):
    s = eq["equity"].astype(float)
    years = len(s) / 243
    total = s.iloc[-1] / s.iloc[0] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    r = s.pct_change().dropna()
    sharpe = ((r.mean() * 243 - risk_free) / (r.std() * np.sqrt(243))) if r.std() > 0 else np.nan
    dd = (s / s.cummax() - 1).min()
    n = len(trades)
    gain = trades.loc[trades["pnl"] > 0, "pnl"].sum() if n else 0.0
    loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum() if n else 0.0
    return {"total_return": round(total, 4), "annual": round(ann, 4),
            "max_drawdown": round(dd, 4), "sharpe": round(sharpe, 2), "trades": n,
            "win_rate": round((trades["pnl"] > 0).mean(), 3) if n else np.nan,
            "profit_factor": round(gain / loss, 2) if loss > 0 else np.nan}
