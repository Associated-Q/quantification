"""自选标的的加减仓建议。

规则来自书里的趋势跟随与 ATR 风险管理，全部可解释：
  1. 中期趋势向下（MA20 < MA60）→ 减仓至 0；
  2. 跌破 20 日均线超过 0.5 倍 ATR → 减仓；
  3. 趋势向上且价格贴近均线（偏离 ≤0.5 ATR）→ 加仓；
  4. 趋势向上但偏离均线超过 2 倍 ATR → 减仓（过热）；
  5. 其余情况 → 持有。

注意：该规则组在 2023-07~2026-10 的 13 只 ETF 样本外回测中未跑赢买入持有，
输出只能作为纪律参照，不构成确定性信号。
"""
import numpy as np
import pandas as pd


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def analyze(df: pd.DataFrame, price: float | None = None):
    if len(df) < 61:
        return {"action": "数据不足", "target_position": None, "reasons": [],
                "metrics": {"bars": len(df)}}

    close = df["close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    atr = _atr(df).iloc[-1]
    px = float(price) if price else float(close.iloc[-1])

    if not np.isfinite(atr) or atr <= 0:
        return {"action": "数据异常", "target_position": None, "reasons": [],
                "metrics": {"ma20": ma20, "ma60": ma60, "atr": atr}}

    dev = (px - ma20) / atr
    trend_up = ma20 > ma60
    reasons, action, target = [], "", 0.0

    if not trend_up:
        action, target = "减仓", 0.0
        reasons.append(f"MA20（{ma20:.3f}）低于 MA60（{ma60:.3f}），中期趋势向下")
        if px < ma20:
            reasons.append(f"现价 {px:.3f} 位于 20 日均线下方")
    elif px < ma20 - 0.5 * atr:
        action, target = "减仓", 0.25
        reasons.append(f"现价 {px:.3f} 跌破 20 日均线（{ma20:.3f}）超过 0.5 倍 ATR")
        reasons.append("趋势虽未破坏，但短线转弱，先降仓")
    elif dev > 2.0:
        action, target = "减仓", 0.3
        reasons.append(f"现价高于 20 日均线 {dev:.1f} 倍 ATR，短线过热")
        reasons.append("趋势仍在，减仓而非清仓")
    elif abs(dev) <= 0.5:
        action, target = "加仓", 0.6
        reasons.append(f"中期趋势向上（MA20 {ma20:.3f} > MA60 {ma60:.3f}）")
        reasons.append(f"现价贴近 20 日均线（偏离 {dev:+.1f} 倍 ATR），回踩不破")
    else:
        action, target = "持有", 0.4
        reasons.append(f"中期趋势向上（MA20 {ma20:.3f} > MA60 {ma60:.3f}）")
        reasons.append(f"现价偏离 20 日均线 {dev:+.1f} 倍 ATR，未到加减仓条件")

    return {
        "action": action,
        "target_position": target,
        "reasons": reasons,
        "metrics": {
            "price": round(px, 3),
            "ma20": round(float(ma20), 3),
            "ma60": round(float(ma60), 3),
            "atr": round(float(atr), 4),
            "deviation_atr": round(float(dev), 2),
            "stop_price": round(float(px - 2 * atr), 3),
            "trend": "向上" if trend_up else "向下",
        },
    }
