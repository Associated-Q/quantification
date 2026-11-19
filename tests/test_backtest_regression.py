"""回测数值回归：固定标的池与参数下，样本外指标不能漂移。

引擎、规则或数据的任何改动若改变了结论，这里会立刻失败。
数值基准来自 2026-11-04 的确认结果。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant"))

import book_strategies as bs  # noqa: E402
import data_source  # noqa: E402
import duckdb  # noqa: E402
import engine  # noqa: E402
import pandas as pd  # noqa: E402
import strategies  # noqa: E402
from run_book_backtest import IS, OOS, UNIVERSE  # noqa: E402

BASELINE = {"annual": 0.037, "max_drawdown": -0.126, "profit_factor": 1.202}


@pytest.fixture(scope="module")
def oos_metrics():
    con = duckdb.connect(data_source.DB, read_only=True)
    df = con.execute("SELECT * FROM kline_daily ORDER BY symbol, trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    rows = []
    for sym in UNIVERSE:
        g = df[df.symbol == sym].reset_index(drop=True)
        if g.empty:
            continue
        s = g[(g.trade_date >= OOS[0]) & (g.trade_date <= OOS[1])].reset_index(drop=True)
        if len(s) < 90:
            continue
        score = bs.book_score(s)
        eq, tr = engine.run(s, score >= 2.0, score <= -1.0, strategies.atr(s),
                            size_pct=0.5, atr_stop=2.0)
        rows.append(engine.metrics(eq, tr))
    assert len(rows) >= 10, f"可用标的只有 {len(rows)} 只，回归基准不成立"
    return pd.DataFrame(rows).mean(numeric_only=True)


def test_oos_annual_matches_baseline(oos_metrics):
    assert oos_metrics["annual"] == pytest.approx(BASELINE["annual"], abs=0.01)


def test_oos_drawdown_matches_baseline(oos_metrics):
    assert oos_metrics["max_drawdown"] == pytest.approx(BASELINE["max_drawdown"], abs=0.02)


def test_oos_profit_factor_matches_baseline(oos_metrics):
    assert oos_metrics["profit_factor"] == pytest.approx(BASELINE["profit_factor"], abs=0.1)


def test_universe_is_pinned():
    """标的池写死，避免数据库里多抓了标的导致数字漂移。"""
    assert len(UNIVERSE) == 13
    assert IS[0] < OOS[0]
