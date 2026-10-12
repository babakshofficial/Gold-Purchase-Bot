"""Fair-price verdicts."""

import unittest

from pricing import GRAMS_PER_OUNCE_18K, fair_gold_price
from signals import analyze_market, determine_verdict


class TestSignals(unittest.TestCase):
    def test_fair_price_uses_shared_constant(self):
        usd, ounce = 250_000, 4_000
        self.assertEqual(fair_gold_price(usd, ounce), usd * ounce / GRAMS_PER_OUNCE_18K)

    def test_analyze_market_buy_sell_hold(self):
        usd, ounce = 250_000, 4_000
        fair = fair_gold_price(usd, ounce)
        buy_at = 100_000
        sell_at = 100_000

        _, var, _, _, status = analyze_market(fair - 150_000, usd, ounce, buy_at, sell_at)
        self.assertEqual(status, "BUY")
        self.assertLess(var, -buy_at)

        _, var, _, _, status = analyze_market(fair + 150_000, usd, ounce, buy_at, sell_at)
        self.assertEqual(status, "SELL")
        self.assertGreater(var, sell_at)

        _, _, _, _, status = analyze_market(fair, usd, ounce, buy_at, sell_at)
        self.assertEqual(status, "HOLD")

    def test_determine_verdict_matches_gap(self):
        _, _, status = determine_verdict(-200_000, 100_000, 100_000)
        self.assertEqual(status, "BUY")
        _, _, status = determine_verdict(200_000, 100_000, 100_000)
        self.assertEqual(status, "SELL")
        _, _, status = determine_verdict(0, 100_000, 100_000)
        self.assertEqual(status, "HOLD")


if __name__ == "__main__":
    unittest.main()
