"""Synthetic regression tests for trading constraints; no network or database needed."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import engine


def fixture(dates, volumes=100000, price=10.0):
    n = len(dates)
    return pd.DataFrame({
        "symbol": ["600000"] * n,
        "trade_date": pd.to_datetime(dates),
        "open": [price] * n,
        "close": [price] * n,
        "high": [price] * n,
        "low": [price] * n,
        "volume": [volumes] * n if np.isscalar(volumes) else volumes,
    })


def run(df, entries, exits=None, **kw):
    n = len(df)
    entry = pd.Series(entries, index=df.index)
    exit_ = pd.Series(exits if exits is not None else [False] * n, index=df.index)
    atr = pd.Series(1.0, index=df.index)
    return engine.run(df, entry, exit_, atr, atr_stop=None, size_pct=1.0, **kw)


class LimitTests(unittest.TestCase):
    def test_board_and_etf_limits(self):
        for code in ("600000", "000001", "510300", "159558", "sh510300", "sz159558"):
            self.assertEqual(engine.infer_limit(code), 0.10, code)
        # 创业板 ETF 与科创板 ETF 实际按 20% 限价（159915/159949/588000 有 20% 涨跌记录）。
        for code in ("300750", "301001", "688001", "689009", "sz300750", "sh688001",
                     "588000", "sh588000", "159915", "sz159915", "159949"):
            self.assertEqual(engine.infer_limit(code), 0.20, code)

    def test_etf_tick_size_blocks_buy_at_limit_up(self):
        # 512480 的 ETF 报价单位是 0.001：前收 0.397，涨停价 0.437。
        df = fixture(["2024-10-07", "2024-10-08", "2024-10-09"], price=0.397)
        df["symbol"] = "sh512480"
        df.loc[1, ["open", "close", "high", "low"]] = 0.437
        _, tr = run(df, [True, False, False])
        self.assertEqual(len(tr), 0)
        # 同一价格按两位小数取整时（旧逻辑）会被误判为未涨停。
        df["symbol"] = "sh600000"
        df.loc[1, ["open", "close", "high", "low"]] = 0.437
        _, tr = run(df, [True, False, False])
        self.assertEqual(len(tr), 1)

    def test_st_limit_depends_on_board(self):
        # 创业板/科创板 ST 股仍为 20%（实测 300313、300338 有 ±20% 交易日）。
        self.assertEqual(engine.infer_limit("300001", "*ST某某"), 0.20)
        self.assertEqual(engine.infer_limit("688001", "*ST某某"), 0.20)
        self.assertEqual(engine.infer_limit("600000", "ST某某"), 0.05)
        self.assertEqual(engine.infer_limit("000001", "*ST某某"), 0.05)

    def test_default_infers_limit_on_a_regular_board(self):
        df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"], price=10.0)
        df["symbol"] = "300750"
        df.loc[1, ["open", "close", "high", "low"]] = 11.0
        _, tr = run(df, [True, False, False])
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr.iloc[0].entry_px, 11.001)

    def test_explicit_narrow_limit_can_override_inferred_limit(self):
        df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"], price=10.0)
        df["symbol"] = "300750"
        df.loc[1, ["open", "close", "high", "low"]] = 11.0
        _, tr = run(df, [True, False, False], limit_pct=0.10)
        self.assertEqual(len(tr), 0)
        _, tr = run(df, [True, False, False], limit_pct=0.20)
        self.assertEqual(len(tr), 1)


class SuspensionTests(unittest.TestCase):
    def test_gap_discards_entry_and_exit_signals(self):
        dates = ["2024-01-02", "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"]
        df = fixture(dates)
        _, tr = run(df, [True, False, False, False, False], max_gap_days=None)
        self.assertEqual(len(tr), 1)
        _, tr = run(df, [True, False, False, False, False])
        self.assertEqual(len(tr), 0)

        df = fixture(["2024-01-02", "2024-01-12", "2024-01-15", "2024-01-16",
                      "2024-01-17", "2024-01-18"])
        entry = pd.Series([False, True, False, False, False, False])
        exit_ = pd.Series([False, True, False, True, False, False])
        atr = pd.Series(1.0, index=df.index)
        _, tr = engine.run(df, entry, exit_, atr, atr_stop=None, size_pct=1.0)
        self.assertEqual(len(tr), 1)
        self.assertEqual(pd.Timestamp(tr.iloc[0].exit_date), pd.Timestamp("2024-01-17"))
        self.assertEqual(tr.iloc[0].reason, "signal")

    def test_normal_weekend_is_not_suspension(self):
        df = fixture(["2024-01-05", "2024-01-08", "2024-01-09"])
        _, tr = run(df, [True, False, False])
        self.assertEqual(len(tr), 1)

    def test_long_holiday_gap_is_not_suspension(self):
        self.assertTrue(engine.is_holiday_gap("2024-02-08", "2024-02-19"))
        self.assertTrue(engine.is_holiday_gap("2025-09-30", "2025-10-09"))
        self.assertFalse(engine.is_holiday_gap("2024-01-02", "2024-01-12"))
        # 春节后第一个交易日的开盘仍应执行节前收盘信号。
        df = fixture(["2024-02-08", "2024-02-19", "2024-02-20"])
        _, tr = run(df, [True, False, False])
        self.assertEqual(len(tr), 1)
        self.assertEqual(pd.Timestamp(tr.iloc[0].entry_date), pd.Timestamp("2024-02-19"))


class VolumeTests(unittest.TestCase):
    def test_daily_lots_are_converted_to_shares_and_capped(self):
        # 上限用信号日 i-1 的成交量：100 手 = 10000 股，5% = 500 股。
        df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"],
                     volumes=[100, 100000, 100000])
        _, tr = run(df, [True, False, False], max_volume_pct=0.05)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.iloc[0].shares, 500)

    def test_cap_smaller_than_one_lot_blocks_entry(self):
        df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"],
                     volumes=[1, 100000, 100000])
        _, tr = run(df, [True, False, False], max_volume_pct=0.05)
        self.assertEqual(len(tr), 0)

    def test_missing_or_negative_volume_blocks_entry_and_none_disables_cap(self):
        for volume in (np.nan, 0.0, -1.0):
            df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"],
                         volumes=[volume, 100000, 100000])
            _, tr = run(df, [True, False, False])
            self.assertEqual(len(tr), 0, volume)
        df = fixture(["2024-01-02", "2024-01-03", "2024-01-04"], volumes=[1, 1, 1])
        _, tr = run(df, [True, False, False], max_volume_pct=None)
        self.assertEqual(tr.iloc[0].shares, 9900)


if __name__ == "__main__":
    unittest.main(verbosity=2)
