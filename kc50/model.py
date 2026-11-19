"""特征、扩展窗口滚动训练、校准概率与回测。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import pivots_asof
from .structs import (CYCLES, analyze_wave, fib_levels, fib_swing)

FEATURES = [
    "gann_days", "gann_hit", "gann_reso", "gann_angle",
    "fib_pos", "fib_dist", "wave_code",
    "trend", "ma20_slope", "ret1", "ret5", "atr_pct", "vol_ratio",
]

ANCHORS = 5          # 江恩只用最近5个已确认阶段转折点（"阶段高低点"）
GAP = 5              # 训练标签与测试窗口之间至少隔离的交易日数
WARMUP = 60


def _row(df: pd.DataFrame, p: pd.DataFrame, t: int) -> dict | None:
    close = float(df["close"].iat[t])
    atr = float(df["atr14"].iat[t])
    ma20 = float(df["ma20"].iat[t])
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(ma20):
        return None

    last = p.iloc[-1]
    days = t - int(last["idx"])
    anchors = p.tail(ANCHORS)

    hit, reso = 0.0, 0.0
    for _, a in anchors.iterrows():
        for c in CYCLES:
            ctr = int(a["idx"]) + c
            if ctr - 1 <= t <= ctr + 1:
                reso += 1.0
    hit = 1.0 if reso >= 2 else 0.0          # >=2 个周期窗口重合才算时间共振

    a = last
    direction = 1 if a["kind"] == "L" else -1
    line = float(a["price"]) + direction * float(a["atr"]) * days
    angle = (close - line) / atr

    sw = fib_swing(p)
    if sw and sw["range"] > 0:
        fib_pos = float(np.clip((close - sw["low"]) / sw["range"], -1.0, 2.0))
        lv = fib_levels(sw)["price"].to_numpy()
        fib_dist = float(np.min(np.abs(lv - close)) / atr)
    else:
        fib_pos, fib_dist = 0.5, np.nan

    w = analyze_wave(p, close)

    return {
        "date": df["date"].iat[t],
        "close": close,
        "gann_days": min(days, 60) / 60.0,
        "gann_hit": hit,
        "gann_reso": min(reso, 4.0) / 4.0,
        "gann_angle": angle,
        "fib_pos": fib_pos,
        "fib_dist": fib_dist,
        "wave_code": w["label_code"] / 5.0,
        "trend": close / ma20 - 1.0,
        "ma20_slope": float(df["ma20_slope5"].iat[t]),
        "ret1": float(df["ret1"].iat[t]),
        "ret5": float(df["ret5"].iat[t]),
        "atr_pct": atr / close,
        "vol_ratio": float(df["vol_ratio"].iat[t]),
    }


def build_features(df: pd.DataFrame, piv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in range(WARMUP, len(df)):
        p = pivots_asof(piv, t)
        if len(p) < 2:
            continue
        r = _row(df, p, t)
        if r:
            rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    c = df["close"].to_numpy(float)
    idx = {d: i for i, d in enumerate(df["date"])}
    ii = out["date"].map(idx).to_numpy()
    for h in (3, 5):
        fwd = np.where(ii + h < len(c), c[np.minimum(ii + h, len(c) - 1)] / c[ii] - 1.0, np.nan)
        out[f"fwd{h}"] = fwd
        out[f"y{h}"] = (fwd > 0).astype(float)
        out.loc[~np.isfinite(fwd), f"y{h}"] = np.nan
    out["idx"] = ii
    return out


def _make_model() -> CalibratedClassifierCV:
    base = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.2, max_iter=2000, solver="lbfgs")),
    ])
    return CalibratedClassifierCV(base, method="sigmoid",
                                  cv=TimeSeriesSplit(n_splits=3))


def walk_forward(feat: pd.DataFrame, horizon: int = 3,
                 oos_start: str = "2025-01-01", oos_end: str | None = None,
                 train_start: str = "2024-01-01") -> pd.DataFrame:
    """扩展窗口滚动训练。每个 t 只用 label_end <= t - GAP 的样本训练。"""
    ycol = f"y{horizon}"
    data = feat[feat[ycol].notna()].reset_index(drop=True)
    oos_end = oos_end or data["date"].max().strftime("%Y-%m-%d")

    rows = []
    for t in range(len(data)):
        row = data.iloc[t]
        if row["date"] < pd.Timestamp(oos_start):
            continue
        if row["date"] > pd.Timestamp(oos_end):
            continue
        ok = ((data["date"] >= pd.Timestamp(train_start))
              & (data["date"] <= row["date"] - pd.Timedelta(days=1))
              & ((data["idx"] + horizon) <= row["idx"] - GAP))
        tr = data[ok]
        if len(tr) < 120 or tr[ycol].nunique() < 2:
            continue
        m = _make_model()
        m.fit(tr[FEATURES].to_numpy(float), tr[ycol].to_numpy(int))
        p = float(m.predict_proba(row[FEATURES].to_numpy(float).reshape(1, -1))[0, 1])
        rows.append({"date": row["date"], "idx": int(row["idx"]), "p_up": p,
                     "y": int(row[ycol]), "close": row["close"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics

def _max_dd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0)) if len(eq) else 0.0


def trade_stats(close: np.ndarray, signal: np.ndarray, cost: float = 0.0005) -> dict:
    """日频调仓：t 收盘信号 -> t+1 起持仓。signal 0/1。"""
    r = np.diff(close) / close[:-1]
    pos = signal[:-1]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * r - turn * cost
    eq = np.cumprod(1.0 + strat)
    ann = (eq[-1] ** (250.0 / max(len(strat), 1)) - 1.0) if len(eq) else 0.0
    return {"cum": float(eq[-1] - 1.0) if len(eq) else 0.0,
            "ann": float(ann), "mdd": _max_dd(eq),
            "avg_daily": float(np.mean(strat)) if len(strat) else 0.0,
            "exposure": float(np.mean(pos)) if len(pos) else 0.0}


def bootstrap_ci(x: np.ndarray, stat=np.mean, n: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(pred: pd.DataFrame, close_all: np.ndarray, base_p: float, horizon: int) -> dict:
    y = pred["y"].to_numpy(bool)
    p = pred["p_up"].to_numpy(float)
    d = p > 0.5
    acc = float(np.mean(d == y))
    long_mask, short_mask = d, ~d
    long_acc = float(np.mean(y[long_mask])) if long_mask.any() else float("nan")
    short_acc = float(np.mean(~y[short_mask])) if short_mask.any() else float("nan")
    bacc = float(np.nanmean([long_acc, short_acc]))
    brier = float(np.mean((p - y) ** 2))
    base_brier = float(np.mean((base_p - y) ** 2))
    lo, hi = bootstrap_ci((d == y).astype(float))
    bl, bh = bootstrap_ci((np.full_like(y, base_p > 0.5) == y).astype(float))
    sig = pred.copy()
    sig["signal"] = d.astype(int)
    c = sig["close"].to_numpy(float)
    ts = trade_stats(c, sig["signal"].to_numpy(int))
    return {
        "horizon": horizon, "n": len(pred),
        "acc": acc, "acc_ci": (lo, hi),
        "long_acc": long_acc, "short_acc": short_acc, "balanced": bacc,
        "coverage": float(np.mean(np.abs(p - 0.5) >= 0.05)),
        "brier": brier, "base_brier": base_brier,
        "base_rate": float(np.mean(y)),
        "always_up_acc": float(np.mean(y)),
        "always_down_acc": float(np.mean(~y)),
        "always_up_bacc": 0.5, "always_up_brier": float(np.mean((1.0 - y) ** 2)),
        "always_down_brier": float(np.mean((0.0 - y) ** 2)),
        "base_brier_sig": float(np.mean((base_p - y) ** 2)),
        "long_brier": base_brier,
        "trade": ts,
        "long_ci": bootstrap_ci(y[long_mask].astype(float)) if long_mask.sum() > 5 else (np.nan, np.nan),
        "short_ci": bootstrap_ci((~y[short_mask]).astype(float)) if short_mask.sum() > 5 else (np.nan, np.nan),
        "balanced_ci": (float(np.nanmean([bootstrap_ci(y[long_mask].astype(float))[0],
                                          bootstrap_ci((~y[short_mask]).astype(float))[0]])),
                        float(np.nanmean([bootstrap_ci(y[long_mask].astype(float))[1],
                                          bootstrap_ci((~y[short_mask]).astype(float))[1]])))
        if long_mask.sum() > 5 and short_mask.sum() > 5 else (np.nan, np.nan),
    }


def baseline_ma20(df: pd.DataFrame, start: str, end: str) -> dict:
    d = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    sig = (d["close"] > d["ma20"]).astype(int).to_numpy()
    ts = trade_stats(d["close"].to_numpy(float), sig)
    return {"signal": sig, "trade": ts, "n": len(d)}


def random_baseline(y: np.ndarray, n_sim: int = 10000, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    base = float(np.mean(y))
    sims = rng.random((n_sim, len(y))) < base
    acc = (sims == y).mean(axis=1)
    return {"acc": float(acc.mean()), "ci": (float(np.percentile(acc, 2.5)),
                                            float(np.percentile(acc, 97.5))),
            "best": float(acc.max()), "n_sim": n_sim}
