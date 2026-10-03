"""Unit tests for usd_fetch parsing (no network)."""

import unittest

from usd_fetch import (
    is_plausible_market_prices,
    is_plausible_usd_toman,
    parse_nerkhedular_post,
    parse_tgjucurrency_post,
    parse_usd_post,
)

NERKHEDULAR_POST = """
💸 دلار فردایی تهران 💵 254,000 خـرید 💸

💸 دلار فردایی تهران 💵 254,050 فروش 💵

💸 دلار فردایی تهران 💵 254,000 معامله ✅

🚗@nerkhedular ◀️
"""

TGJU_POST = """
قیمت ارزهای آزاد
🇺🇸 دلار : 2,540,000 ریال
"""


class TestUsdFetch(unittest.TestCase):
    def test_nerkhedular_prefers_deal(self):
        self.assertEqual(parse_nerkhedular_post(NERKHEDULAR_POST), 254_000.0)

    def test_nerkhedular_skips_unrelated(self):
        self.assertIsNone(parse_nerkhedular_post("قیمت طلا امروز"))

    def test_tgjucurrency_rial_to_toman(self):
        self.assertEqual(parse_tgjucurrency_post(TGJU_POST), 254_000.0)

    def test_parse_usd_post_primary_first(self):
        self.assertEqual(parse_usd_post(NERKHEDULAR_POST), 254_000.0)

    def test_nerkhedular_buy_only(self):
        text = "💸 دلار فردایی تهران 💵 255,500 خـرید 💸"
        self.assertEqual(parse_nerkhedular_post(text), 255_500.0)

    def test_plausible_usd_rejects_800_toman_mistake(self):
        self.assertFalse(is_plausible_usd_toman(800))
        self.assertTrue(is_plausible_usd_toman(254_000))

    def test_market_sanity_rejects_usd_800_with_real_gold(self):
        tala = 26_122_000
        ounce = 4133.0
        self.assertFalse(is_plausible_market_prices(tala, 800, ounce))
        self.assertTrue(is_plausible_market_prices(tala, 262_000, ounce))


if __name__ == "__main__":
    unittest.main()
