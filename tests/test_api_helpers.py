"""服务层纯函数测试：标的规范化、来源策略、持仓建议冲突标记。"""
import pytest
from fastapi import HTTPException

import app as webapp


def test_effective_sources_defaults_to_books():
    class Req:
        sources = None

    assert webapp._effective_sources(Req()) == ["book"]


def test_effective_sources_respects_explicit_choice():
    class Req:
        sources = ["qa", "comment"]

    assert webapp._effective_sources(Req()) == ["qa", "comment"]


def test_normalize_symbol_accepts_code_and_prefixed():
    assert webapp._normalize_symbol("510300") == "sh510300"
    assert webapp._normalize_symbol("sz159915") == "sz159915"


def test_normalize_symbol_rejects_garbage():
    with pytest.raises(HTTPException) as exc:
        webapp._normalize_symbol("abc")
    assert exc.value.status_code == 422


def test_position_suggestion_flags_conflict():
    """规则说减仓、但仓位低于目标需要加仓时，必须标记冲突。"""
    text, conflict = webapp._position_suggestion("减仓", 10)
    assert text == "加 10 手" and conflict is True

    text, conflict = webapp._position_suggestion("减仓", -2)
    assert text == "减 2 手" and conflict is False


def test_position_suggestion_clear_position():
    text, conflict = webapp._position_suggestion("清仓", -3)
    assert text.startswith("清仓") and conflict is False


def test_position_suggestion_no_data():
    text, conflict = webapp._position_suggestion("观望", None)
    assert text == "暂无建议" and conflict is False


def _ohlc(points, steps=8):
    """按转折点构造单调走势，保证 ZigZag 在每个点转向。"""
    import numpy as np
    import pandas as pd

    closes = []
    for a, b in zip(points, points[1:]):
        closes.extend(np.linspace(a, b, steps, endpoint=False))
    closes.append(points[-1])
    rows, prev = [], closes[0]
    for c in closes:
        rows.append({"open": prev, "close": c, "high": max(prev, c),
                     "low": min(prev, c), "volume": 1.0})
        prev = c
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.date_range("2026-01-01", periods=len(df), freq="D")
    return df


def test_wave_overlay_aligns_pivots_with_timestamps():
    df = _ohlc([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    res = webapp.wv.analyze_wave(df)
    assert res["valid"] is True
    ov = webapp._wave_overlay(df, res, "trade_date")
    assert ov["valid"] is True and ov["candidate"] == []
    assert ov["groups"] and any(g.get("current") for g in ov["groups"])
    for g in ov["groups"]:
        for p in g["pivots"]:
            assert p["t"] == str(df["trade_date"].iloc[p["i"]])
    assert ov["fib"]["source"] == "wave"
    assert len(ov["fib"]["levels"]) == len(webapp.wv.FIB_RATIOS)


def test_wave_overlay_keeps_real_reason_and_swing_fib_when_invalid():
    df = _ohlc([1.00, 0.90, 0.99, 0.79, 0.92, 0.77])
    res = webapp.wv.analyze_wave(df)
    assert res["valid"] is False
    ov = webapp._wave_overlay(df, res, "trade_date")
    assert ov["valid"] is False and "浪4" in ov["why"]
    assert len(ov["candidate"]) == 6
    assert ov["fib"]["source"] in ("group", "swing")
    assert len(ov["fib"]["levels"]) == len(webapp.wv.FIB_RATIOS)
