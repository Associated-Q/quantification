"""AI 研判的新鲜度：价格大幅变动时必须标记过期，避免拿早上的结论当现在的。"""
import pytest

import ai_analyst as ai


def _cached(price):
    return {"facts": {"current_price": price}, "description": {}, "advice": {},
            "passages": [], "model": "m", "created_at": "2026-11-03 09:10:00"}


@pytest.fixture
def patch_cache(monkeypatch):
    def _apply(used_price):
        monkeypatch.setattr(ai.data_source, "to_symbol", lambda s: "sh600000")
        monkeypatch.setattr(ai, "_load_latest_cache",
                            lambda sym: ("2026-11-03", _cached(used_price)))
    return _apply


def test_latest_flags_stale_when_price_moved(monkeypatch, patch_cache):
    patch_cache(100.0)
    monkeypatch.setattr(ai.realtime, "fetch_quotes",
                        lambda syms: {"sh600000": {"price": 110.0}})
    out = ai.latest("600000")
    assert out["stale"] is True
    assert out["price_drift_pct"] == pytest.approx(10.0)


def test_latest_not_stale_for_small_move(monkeypatch, patch_cache):
    patch_cache(100.0)
    monkeypatch.setattr(ai.realtime, "fetch_quotes",
                        lambda syms: {"sh600000": {"price": 100.5}})
    out = ai.latest("600000")
    assert out["stale"] is False


def test_latest_tolerates_quote_failure(monkeypatch, patch_cache):
    """行情拿不到时仍返回缓存内容，不能因此报错。"""
    patch_cache(100.0)

    def boom(syms):
        raise RuntimeError("network down")

    monkeypatch.setattr(ai.realtime, "fetch_quotes", boom)
    out = ai.latest("600000")
    assert out["stale"] is False and out["facts"]["current_price"] == 100.0
