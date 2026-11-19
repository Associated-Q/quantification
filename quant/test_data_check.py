"""Synthetic tests for the read-only daily data quality scanner."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import data_check


def bars(symbol, dates, closes, volumes=1000):
    return pd.DataFrame({
        "symbol": symbol,
        "trade_date": pd.to_datetime(dates),
        "open": closes,
        "close": closes,
        "high": closes,
        "low": closes,
        "volume": volumes,
    })


class ScanTests(unittest.TestCase):
    def test_gap_zero_volume_and_board_limit_breach(self):
        df = pd.concat([
            bars("600000", ["2024-01-02", "2024-01-03", "2024-01-12"],
                 [10.0, 12.0, 12.0], [1000, 0, 1000]),
            bars("300750", ["2024-01-02", "2024-01-03"], [10.0, 12.0]),
        ], ignore_index=True)
        report = data_check.scan(df, max_gap_days=5)
        actual = set(zip(report.symbol, report.anomaly, report.trade_date.astype(str)))
        self.assertIn(("600000", "gap", "2024-01-12"), actual)
        self.assertIn(("600000", "zero_volume", "2024-01-03"), actual)
        self.assertIn(("600000", "limit_breach", "2024-01-03"), actual)
        self.assertNotIn(("300750", "limit_breach", "2024-01-03"), actual)
        self.assertEqual(len(report), 3)

    def test_price_tick_rounding_is_not_a_limit_breach(self):
        df = bars("600000", ["2024-01-02", "2024-01-03"], [10.05, 11.06])
        self.assertTrue(data_check.scan(df).empty)

    def test_etf_tick_and_board_limit_are_respected(self):
        # ETF 最小变动价位 0.001：0.878 -> 涨停价 0.966，不应报超限。
        self.assertTrue(data_check.scan(
            bars("sh512480", ["2024-10-17", "2024-10-18"], [0.878, 0.966])).empty)
        # 588000 为科创板 ETF，20% 涨停价 0.950，不应按 10% 报超限。
        self.assertTrue(data_check.scan(
            bars("sh588000", ["2024-09-27", "2024-09-30"], [0.792, 0.950])).empty)

    def test_holiday_gap_is_classified_separately(self):
        report = data_check.scan(
            bars("600000", ["2024-02-08", "2024-02-19"], [10.0, 10.0]))
        gap = report[report.anomaly == "gap"].iloc[0]
        self.assertEqual(gap.gap_type, "holiday")
        report = data_check.scan(
            bars("600000", ["2024-01-02", "2024-01-12"], [10.0, 10.0]))
        gap = report[report.anomaly == "gap"].iloc[0]
        self.assertEqual(gap.gap_type, "suspension")

    def test_summary_contains_every_symbol(self):
        df = pd.concat([
            bars("600000", ["2024-01-02", "2024-01-03"], [10.0, 10.0]),
            bars("159915", ["2024-01-02", "2024-01-03"], [1.0, 1.0]),
        ], ignore_index=True)
        summary = data_check.summarize(df, data_check.scan(df))
        self.assertEqual(summary.symbol.tolist(), ["159915", "600000"])
        self.assertEqual(summary.bars.tolist(), [2, 2])
        self.assertEqual(summary.anomalies.sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
