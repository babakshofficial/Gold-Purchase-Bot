"""Unit tests for market_indicators (no network)."""

import unittest

from market_indicators import calculate_market_indicators, trend_label_fa


class TestMarketIndicators(unittest.TestCase):
    def test_insufficient_data(self):
        self.assertEqual(
            calculate_market_indicators([100_000.0]),
            {"trend": "N/A", "rsi": "N/A", "volatility": "N/A"},
        )

    def test_trend_and_vol_with_three_points(self):
        # rising gaps → upward trend
        diffs = [100_000.0, 150_000.0, 200_000.0, 250_000.0]
        m = calculate_market_indicators(diffs)
        self.assertEqual(m["trend"], "UPWARD")
        self.assertNotEqual(m["volatility"], "N/A")
        self.assertEqual(m["rsi"], "N/A")  # need 14+ historical for RSI

    def test_rsi_with_enough_history(self):
        base = 100_000.0
        diffs = [base + i * 1_000 for i in range(16)]
        m = calculate_market_indicators(diffs)
        self.assertNotEqual(m["rsi"], "N/A")
        self.assertIsInstance(m["rsi"], float)

    def test_trend_label_fa(self):
        self.assertIn("صعودی", trend_label_fa("UPWARD"))


if __name__ == "__main__":
    unittest.main()
