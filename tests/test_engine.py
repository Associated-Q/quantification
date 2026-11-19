"""回测内核的行为测试：费用、涨跌停、T+1。

用构造的确定性行情，断言可手算的结论，避免依赖真实数据。
"""
import pandas as pd
import pytest

import engine


def make_df(rows):
    """rows: [(date, open, high, low, close, volume)]"""
    df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def flat_atr(df, value=1.0):
    return pd.Series([value] * len(df))


def test_buy_and_hold_matches_price_return_net_of_costs():
    """满仓买入持有的收益应等于价格涨幅减去交易成本。"""
    df = make_df([
        ("2024-01-02", 10.0, 10.2, 9.9, 10.1, 1e6),
        ("2024-01-03", 10.1, 10.5, 10.0, 10.4, 1e6),
        ("2024-01-04", 10.4, 11.0, 10.3, 10.9, 1e6),
        ("2024-01-05", 10.9, 11.2, 10.8, 11.1, 1e6),
    ])
    entry = pd.Series([True] * len(df))
    exit_ = pd.Series([False] * len(df))
    eq, trades = engine.run(df, entry, exit_, flat_atr(df),
                            atr_stop=None, size_pct=1.0, limit_pct=0.1)
    price_ret = df["close"].iloc[-1] / df["open"].iloc[1] - 1
    total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
    assert len(trades) == 1
    # 差额来自佣金/过户费/印花税/滑点，应在 1 个百分点以内
    assert total_ret == pytest.approx(price_ret, abs=0.01)
    assert total_ret < price_ret          # 成本一定是负贡献


def test_limit_up_open_blocks_entry():
    """次日一字涨停开盘时买不进。"""
    df = make_df([
        ("2024-01-02", 10.0, 10.0, 10.0, 10.0, 1e6),
        ("2024-01-03", 11.0, 11.0, 11.0, 11.0, 1e6),   # 较前收 +10%，涨停开盘
        ("2024-01-04", 11.0, 11.5, 10.9, 11.2, 1e6),
    ])
    entry = pd.Series([True, False, False])
    exit_ = pd.Series([False, False, False])
    _, trades = engine.run(df, entry, exit_, flat_atr(df), limit_pct=0.1)
    assert trades.empty, "涨停开盘不应成交"


def test_limit_down_open_blocks_exit():
    """跌停开盘时卖不出，仓位被锁住。"""
    df = make_df([
        ("2024-01-02", 10.0, 10.0, 10.0, 10.0, 1e6),
        ("2024-01-03", 10.0, 10.2, 9.9, 10.1, 1e6),    # 正常买入
        ("2024-01-04", 9.09, 9.09, 9.09, 9.09, 1e6),   # 较前收 -10%，跌停开盘
        ("2024-01-05", 9.5, 9.6, 9.4, 9.5, 1e6),
    ])
    entry = pd.Series([True, False, False, False])
    exit_ = pd.Series([False, True, True, True])       # 第 2 根收盘发出离场
    _, trades = engine.run(df, entry, exit_, flat_atr(df), limit_pct=0.1)
    assert len(trades) == 1
    # 跌停日卖不出，成交日期必然晚于跌停日
    assert str(trades.iloc[0]["exit_date"])[:10] > "2024-01-04"


def test_exit_never_before_entry_plus_one_bar():
    """T+1：买入当天不可能卖出。"""
    df = make_df([
        ("2024-01-02", 10.0, 10.2, 9.9, 10.1, 1e6),
        ("2024-01-03", 10.1, 10.5, 10.0, 10.4, 1e6),
        ("2024-01-04", 10.4, 10.6, 10.2, 10.5, 1e6),
        ("2024-01-05", 10.5, 10.7, 10.3, 10.6, 1e6),
    ])
    entry = pd.Series([True, True, False, False])
    exit_ = pd.Series([True, True, True, False])       # 刻意在买入当日也给离场信号
    _, trades = engine.run(df, entry, exit_, flat_atr(df), limit_pct=0.1)
    for _, t in trades.iterrows():
        assert t["exit_date"] > t["entry_date"], "卖出日期必须晚于买入日期"
