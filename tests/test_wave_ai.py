"""AI 选浪的候选构造与输出校验：模型只能在规则候选里选，不能凭空点位。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant"))

import wave  # noqa: E402
import wave_ai  # noqa: E402


def _df(points, steps=8):
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


def test_payload_lists_real_candidates_and_points():
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    result = wave.analyze_wave(df)
    payload = wave_ai.build_payload(df, result, "trade_date", "sh000001", "day")
    assert payload["candidates"]
    cand = payload["candidates"][0]
    assert cand["id"] == 0 and len(cand["points"]) == 6
    assert cand["start"] == cand["points"][0]["date"]
    assert cand["end"] == cand["points"][-1]["date"]
    dates = {str(x) for x in df["trade_date"]}
    assert all(p["date"] in dates for p in cand["points"])
    assert payload["current"]["valid"] is True


def test_validate_choice_rejects_unknown_id():
    with pytest.raises(ValueError) as exc:
        wave_ai.validate_choice({"chosen": 99}, [0, 1])
    assert "99" in str(exc.value)


def test_validate_choice_coerces_and_filters():
    out = wave_ai.validate_choice(
        {"chosen": "1", "alternatives": ["0", 1, 99, 0], "confidence": 5, "reason": "  ok  "},
        [0, 1])
    assert out["chosen"] == 1
    assert out["alternatives"] == [0]
    assert out["confidence"] == 1.0
    assert out["reason"] == "ok"


def test_validate_choice_accepts_none():
    out = wave_ai.validate_choice({"chosen": None, "alternatives": [], "reason": "无"}, [0])
    assert out["chosen"] is None and out["reason"] == "无"


def test_assemble_builds_fib_for_chosen_candidate():
    df = _df([1.00, 0.80, 0.91, 0.66, 0.78, 0.60])
    payload = wave_ai.build_payload(df, wave.analyze_wave(df), "trade_date", "x", "day")
    out = wave_ai.assemble(payload, {"chosen": 0, "alternatives": [], "confidence": 0.5,
                                     "reason": "测试"}, model="m")
    assert out["chosen"]["id"] == 0
    assert out["chosen"]["fib"]["label"].startswith("AI 主数法")
    assert len(out["chosen"]["fib"]["levels"]) == len(wave.FIB_RATIOS)
    assert out["model"] == "m"


def test_assemble_rejects_choice_outside_candidates():
    payload = {"symbol": "x", "period": "day", "candidates": []}
    with pytest.raises(ValueError):
        wave_ai.assemble(payload, {"chosen": 0}, model="m")


def test_prompt_forbids_inventing_points():
    prompt = wave_ai.build_prompt({"symbol": "x", "period": "day", "candidates": []})
    assert "禁止发明" in prompt
    assert "chosen" in prompt
