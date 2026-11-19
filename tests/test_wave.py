"""浪型识别回归：推动浪、收缩终结楔形、失效原因与转折点清洗。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant"))

import wave  # noqa: E402


def _df(points, steps=8):
    """按给定转折点价格构造单调走势，保证 ZigZag 能在每个点转向。"""
    closes = []
    for a, b in zip(points, points[1:]):
        closes.extend(np.linspace(a, b, steps, endpoint=False))
    closes.append(points[-1])
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append({"open": prev, "close": c,
                     "high": max(prev, c), "low": min(prev, c), "volume": 1.0})
        prev = c
    return pd.DataFrame(rows)


def test_standard_impulse_is_recognized():
    # 1.00 → 0.80 → 0.91 → 0.66 → 0.78 → 0.60，浪3 明显长于其余两段
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    r = wave.analyze_wave(df)
    assert r["valid"] is True
    assert r["pattern"] == "推动浪"
    assert r["direction"] == "下降"
    assert r["wave_ends"]["浪5"] == 0.6


def test_failed_fifth_is_labelled_and_downweighted():
    # 浪5 只到 0.68，没有跌破浪3 终点 0.66
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.68])
    r = wave.analyze_wave(df)
    assert r["valid"] is True
    assert r["failed"] is True
    assert r["weight"] == 0.75
    assert "失败五浪" in r["position"] or "五浪下跌失败" in r["position"]


def test_contracting_ending_diagonal_is_recognized():
    # 浪1 0.20 > 浪3 0.12 > 浪5 0.075，浪4 回到浪1 区域内但不到浪2 终点
    df = _df([1.00, 0.80, 0.91, 0.79, 0.86, 0.785])
    r = wave.analyze_wave(df)
    assert r["valid"] is True
    assert r["pattern"] == "终结楔形"
    assert r["direction"] == "下降"
    assert r["position"] == "处于下降五浪（终结楔形）中"
    assert r["sid"] == -1


def test_overlap_without_contraction_is_rejected():
    # 浪1 0.10 < 浪3 0.20，不满足收缩条件，浪4 又重叠浪1 → 不成立
    df = _df([1.00, 0.90, 0.99, 0.79, 0.92, 0.77])
    r = wave.analyze_wave(df)
    assert r["valid"] is False
    assert "浪4" in r["why"]


def test_stale_structure_keeps_real_reason():
    # 结构成立后价格一路走高，结构终点已超过 15 根，必须返回真实原因而不是空字符串
    points = [1.00, 0.80, 0.91, 0.66, 0.78, 0.60]
    df = _df(points + [0.60, 0.90], steps=8)
    r = wave.analyze_wave(df)
    assert r["ok"] is True
    assert r["valid"] is False
    assert "结构已陈旧" in r["why"]
    assert "超过 15 根" in r["why"]


def test_clean_zigzag_merges_same_direction():
    piv = [(1, 10.0, "H"), (3, 11.0, "H"), (5, 8.0, "L"), (7, 7.5, "L"), (9, 12.0, "H")]
    out = wave.clean_zigzag(piv)
    assert out == [(3, 11.0, "H"), (7, 7.5, "L"), (9, 12.0, "H")]


def test_quality_rejects_shallow_wave2():
    reason = wave._quality_reason("推动浪", 0.10, 0.3, (1.0, 2.0, 1.0))
    assert "浪2" in reason


def test_quality_rejects_deep_wave4():
    reason = wave._quality_reason("推动浪", 0.5, 0.80, (1.0, 2.0, 1.0))
    assert "浪4" in reason


def test_quality_rejects_short_wave3():
    reason = wave._quality_reason("推动浪", 0.5, 0.3, (1.0, 1.1, 1.0))
    assert "浪3" in reason


def test_diagonal_skips_wave4_ratio_but_keeps_wave2_ratio():
    assert wave._quality_reason("终结楔形", 0.5, 0.9, (1.0, 0.6, 0.3)) is None
    assert "浪2" in wave._quality_reason("终结楔形", 0.1, 0.9, (1.0, 0.6, 0.3))


def test_fib_levels_bracket_the_move():
    up = wave.fib_levels(10.0, 20.0)
    assert [x["ratio"] for x in up] == list(wave.FIB_RATIOS)
    assert up[1]["price"] == pytest.approx(20 - 10 * 0.382, abs=1e-3)
    down = wave.fib_levels(20.0, 10.0)
    assert all(10 < x["price"] < 20 for x in down)


def test_fib_levels_reject_zero_span():
    assert wave.fib_levels(10.0, 10.0) == []
    assert wave.fib_levels(None, 10.0) == []


def test_last_swing_returns_most_recent_leg():
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    a, b, th = wave.last_swing(df)
    assert a[2] == "H" and b[2] == "L"
    assert b[1] == 0.60
    assert 0.02 <= th <= 0.08


def test_wave_groups_finds_multiple_structures_without_overlap():
    # 两段连续的推动浪：下降五浪后接上升五浪，第二段从第一段终点起步
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60,
              0.80, 0.72, 1.02, 0.94, 1.10])
    groups = wave.wave_groups(df)
    assert len(groups) >= 2
    assert min(g["pivots"][0][0] for g in groups) == 0
    by_degree = {}
    for g in groups:
        by_degree.setdefault(g["degree"], []).append(g)
    for same in by_degree.values():
        same.sort(key=lambda g: g["pivots"][0][0])
        for a, b in zip(same, same[1:]):
            assert b["pivots"][0][0] >= a["pivots"][-1][0]


def test_wave_groups_degrees_are_ordered_and_named():
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    groups = wave.wave_groups(df)
    assert groups
    for g in groups:
        assert 0 <= g["degree"] < len(wave.DEGREE_NAMES)
        assert g["degree_name"] == wave.DEGREE_NAMES[g["degree"]]
        assert len(g["pivots"]) == 6


def test_wave_groups_relaxed_is_superset_of_strict():
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60,
              0.80, 0.72, 1.02, 0.94, 1.10])
    strict = wave.wave_groups(df, strict=True)
    relaxed = wave.wave_groups(df, strict=False)
    assert len(relaxed) >= len(strict)


def test_zigzag_handles_negative_adjusted_prices():
    # 高分红股前复权后早期价格可能为负，ZigZag 不能因此丢掉整段转折点
    df = _df([-120.0, -100.0, -110.0, -80.0, -90.0, -70.0])
    piv = wave.clean_zigzag(wave.zigzag(df["high"], df["low"], 0.05))
    assert len(piv) >= 5
    assert wave.wave_groups(df)
