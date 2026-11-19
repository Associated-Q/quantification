"""条件式计划：价位必须来自真实数据，方向跟随周月趋势，失效位在加仓区下方。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant"))

import plan  # noqa: E402


def make_df(n=320, slope=0.25, amp=4.0):
    i = np.arange(n)
    close = 50 + slope * i + amp * np.sin(i / 9.0)
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-02", periods=n),
        "open": close,
        "high": close + 1.2,
        "low": close - 1.2,
        "close": close,
        "volume": np.full(n, 1e6),
    })


def fake_analysis(up=True, action="加仓"):
    return {
        "week": {"trend": {"up": up, "ma_fast": 100.0, "ma_slow": 95.0, "bars": 80}},
        "month": {"trend": {"up": up, "ma_fast": 90.0, "ma_slow": 80.0, "bars": 40}},
        "action": action,
        "target_position": 0.6,
    }


def fake_gann(side=1):
    return {"asof": "2026-11-19", "current": {"side": side, "head": "处于江恩 7 循环窗口"},
            "anchors": [{"date": "2026-09-01", "price": 70.0, "label": "低点"}]}


def test_levels_are_real_and_split_around_price():
    day, week = make_df(), make_df(n=120, slope=0.5, amp=3.0)
    price = float(day["close"].iloc[-1])
    levels = plan.build_levels(price, plan._atr(day), day, week, {}, {}, fake_gann())
    assert levels["support"] and levels["resistance"]
    assert all(x["price"] < price and x["price"] > 0 for x in levels["support"])
    assert all(x["price"] > price and x["price"] > 0 for x in levels["resistance"])
    assert all(x["sources"] for x in levels["support"] + levels["resistance"])
    lo, hi = float(day["low"].min()), float(day["high"].max())
    for x in levels["support"] + levels["resistance"]:
        assert lo * 0.9 <= x["price"] <= hi * 1.1


def test_uptrend_plan_adds_on_pullback_with_stop_below():
    day, week = make_df(), make_df(n=120, slope=0.5, amp=3.0)
    price = float(day["close"].iloc[-1])
    out = plan.build("x", "测试", day, week, price, fake_analysis(True),
                     fake_gann(1), {}, {"valid": True, "direction": "上升", "pattern": "推动浪"})
    assert out["long"]["bias"] == "偏多"
    assert out["short"]["bias"] in ("偏多", "偏空", "分歧")
    assert out["levels"]["support"] and out["levels"]["resistance"]
    if out["long"]["add_zone"]:
        z = out["long"]["add_zone"]
        assert z["price"] < price
        if out["long"]["stop"]:
            assert out["long"]["stop"]["price"] <= z["low"] + 1e-9
    assert out["conditions"]
    assert "不构成投资建议" in out["disclaimer"]


def test_downtrend_plan_prefers_reducing_resistance():
    day = make_df(n=200, slope=-0.12, amp=4.0)
    week = make_df(n=120, slope=-0.30, amp=3.0)
    price = float(day["close"].iloc[-1])
    out = plan.build("x", "测试", day, week, price, fake_analysis(False, "减仓"),
                     fake_gann(-1), {}, {"valid": True, "direction": "下降", "pattern": "推动浪"})
    assert out["long"]["bias"] == "偏空"
    assert out["short"]["bias"] in ("偏多", "偏空", "分歧")
    if out["long"]["reduce_zone"]:
        assert out["long"]["reduce_zone"]["price"] > price


def test_missing_gann_and_invalid_waves_do_not_crash():
    day, week = make_df(), make_df(n=120)
    price = float(day["close"].iloc[-1])
    out = plan.build("x", "测试", day, week, price, fake_analysis(True),
                     None, {"valid": False, "why": "结构陈旧"},
                     {"valid": False, "why": "结构陈旧"})
    assert out["long"]["bias"] in ("偏多", "偏空", "分歧")
    assert any("江恩" not in w for w in out["warnings"])
    assert out["levels"]["support"]


def test_invalid_price_hard_errors():
    with pytest.raises(ValueError):
        plan.build_levels(0, 1.0, make_df(), make_df(), {}, {})
