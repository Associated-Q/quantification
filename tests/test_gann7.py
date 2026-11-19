"""江恩 7 循环：窗口投影、因果性、统计口径与接口契约。"""
import numpy as np
import pandas as pd
import pytest

import gann7


def make_df(n=180):
    i = np.arange(n)
    close = 100 + 10 * np.sin(i / 8.0) + i * 0.02
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2025-01-02", periods=n),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1e6),
    })


def test_analyze_projects_7_multiples():
    df = make_df()
    r = gann7.analyze(df)
    assert r["bars"] == len(df)
    assert r["asof"] == str(df["trade_date"].iloc[-1])[:10]
    assert r["windows"], "应至少投影出一个窗口"
    assert {w["cycle"] for w in r["windows"]} <= set(gann7.CYCLES)
    for w in r["windows"]:
        assert w["center_idx"] >= 0
        assert w["status"] in ("已实现", "窗口内", "待验证", "已失效")


def test_future_windows_have_no_invented_date():
    df = make_df()
    r = gann7.analyze(df)
    now = len(df) - 1
    for w in r["windows"]:
        if w["center_idx"] > now:
            assert w["center_date"] is None
            assert w["status"] in ("窗口内", "待验证", "已失效")
        else:
            assert w["center_date"] is not None


def test_cycle_position_and_next_node_consistent():
    df = make_df()
    r = gann7.analyze(df)
    c = r["cycle"]
    assert c["position"] == c["bars_since_anchor"] % 7
    assert c["next_node_in"] == (7 - c["position"]) % 7


def test_current_verdict_shape():
    df = make_df()
    r = gann7.analyze(df)
    cur = r["current"]
    assert cur["action"] in ("留意反弹", "留意回调", "观察")
    assert cur["side"] in (-1, 0, 1)
    assert isinstance(cur["head"], str) and cur["head"]


def test_stats_compare_with_base_rate():
    df = make_df()
    st = gann7.analyze(df)["stats"]
    assert st["samples"] > 0
    assert 0 <= st["hit_rate"] <= 1
    assert 0 <= st["base_rate"] <= 1
    assert 0 <= st["p_value"] <= 1
    assert st["by_cycle"]


def test_rejects_short_and_incomplete_data():
    with pytest.raises(ValueError, match="至少需要"):
        gann7.analyze(make_df(60))
    bad = make_df().drop(columns=["low"])
    with pytest.raises(ValueError, match="缺少字段"):
        gann7.analyze(bad)


def test_endpoint_contract_without_network(monkeypatch):
    import app as webapp

    df = make_df()
    monkeypatch.setattr(webapp.rt, "history", lambda *a, **k: df)
    monkeypatch.setattr(webapp, "cached_quotes", lambda syms: {})
    out = webapp.detail_gann7("600000")
    assert out["symbol"] == "sh600000"
    assert out["name"] == "sh600000"
    assert out["stats"]["samples"] > 0


def test_binom_two_sided_matches_small_sample_hand_calc():
    # n=4, k=4, p=0.5：双侧概率 = 2 × (0.5^4) = 0.125
    assert gann7._binom_two_sided(4, 4, 0.5) == pytest.approx(0.125, abs=1e-12)


def test_binom_two_sided_large_sample_does_not_overflow():
    # 药明这类标的样本数会到几百上千，math.comb(n, i) 转 float 会溢出
    assert 0.9 < gann7._binom_two_sided(500, 1000, 0.5) <= 1.0
    assert gann7._binom_two_sided(600, 1000, 0.5) < 0.05
