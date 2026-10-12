"""Personalized BUY/HOLD/SELL from forecast returns."""

import unittest

from goal_engine import compute_signal


def _prediction(r1=0.0, r7=0.0, r30=0.0):
    return {
        "expected_return_1d": r1,
        "expected_return_7d": r7,
        "expected_return_30d": r30,
    }


class TestGoalEngine(unittest.TestCase):
    def test_short_aggressive_buy(self):
        # 0.4*1 + 0.6*2 = 1.6, aggressive threshold is 0.5
        info = compute_signal(_prediction(1.0, 2.0, 0.0), "short", "aggressive")
        self.assertEqual(info["signal"], "BUY")
        self.assertGreater(info["expected_return"], info["threshold"])

    def test_flat_forecast_holds(self):
        info = compute_signal(_prediction(0.1, 0.1, 0.1), "medium", "conservative")
        self.assertEqual(info["signal"], "HOLD")

    def test_negative_return_sells(self):
        info = compute_signal(_prediction(-3, -3, -3), "short", "medium")
        self.assertEqual(info["signal"], "SELL")

    def test_wedding_uses_wider_band(self):
        # Medium blends 7d and 30d; wedding looks only at 30d with a 1.5x band.
        plain = compute_signal(_prediction(0, 1.0, 1.2), "medium", "medium")
        wedding = compute_signal(_prediction(0, 1.0, 1.2), "wedding", "medium")
        self.assertEqual(plain["signal"], "BUY")
        self.assertEqual(wedding["signal"], "HOLD")
        self.assertAlmostEqual(wedding["threshold"], 1.5)


if __name__ == "__main__":
    unittest.main()
