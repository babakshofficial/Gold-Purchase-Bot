"""Unit tests for crypto_fetch parsing (no network)."""

import unittest

from crypto_fetch import (
    MARKET_POST_MARKER,
    find_market_post,
    find_cryptopricefeed_post,
    parse_arz247_coin,
    parse_cryptopricefeed_coin,
    parse_ecogold_usdt,
    parse_toman_value,
)

ARZ247_MARKET_POST = """
📊 قیمت لحظه‌ای ارزهای دیجیتال
🕐 1405/05/26 12:00 | 15 ارز منتخب بازار
2. بیت کوین / Bitcoin (BTC) 💵 $63,324 | 💰 11.76 میلیارد تومان 📊 حجم: $230,269.62 | 🟢 +0.31%
7. اتریوم / Ethereum (ETH) 💵 $1,891 | 💰 351.16 میلیون تومان 📊 حجم: $184,652.83 | 🟢 +0.89%
5. ترون / TRON (TRX) 💵 $0.3344 | 💰 62,100 تومان 📊 حجم: $81,399.18 | 🟢 +0.34%
"""

ARZ247_RANDOM_POST = """
📊 قیمت لحظه‌ای ارزهای دیجیتال
🕐 1405/05/26 12:00 | انتخاب تصادفی
20. فور / Four (FORM) 💵 $0.2060 | 💰 38,422 تومان 📊 حجم: $14,663.51 | 🟢 +1.78%
"""

ECOGOLD_POST = """
🔻طلای 18 عیار: 19,112,000 تومان
🔻تتر: 186,077 تومان
🔻اونس طلا: 4,396$
"""

CRYPTOPRICEFEED_POST = """
🟢 #BTC: $79,664.00
🟢 #ETH: $2,520.16
🟢 #DOGE: $0.091
🟢 #TON: $1.41

@CryptoPriceFeed
"""


class TestCryptoFetch(unittest.TestCase):
    def test_parse_toman_plain(self):
        self.assertEqual(parse_toman_value("62100", None), 62100.0)

    def test_parse_toman_million(self):
        self.assertEqual(parse_toman_value("351.16", "میلیون"), 351_160_000.0)

    def test_parse_toman_billion(self):
        self.assertEqual(parse_toman_value("11.76", "میلیارد"), 11_760_000_000.0)

    def test_find_market_post_skips_random(self):
        posts = [ARZ247_RANDOM_POST, ARZ247_MARKET_POST]
        found = find_market_post(posts)
        self.assertIn(MARKET_POST_MARKER, found)

    def test_parse_btc(self):
        data = parse_arz247_coin(ARZ247_MARKET_POST, "BTC")
        self.assertIsNotNone(data)
        self.assertEqual(data["usd"], 63324.0)
        self.assertEqual(data["toman"], 11_760_000_000.0)
        self.assertEqual(data["source"], "arz_247")

    def test_parse_eth(self):
        data = parse_arz247_coin(ARZ247_MARKET_POST, "ETH")
        self.assertIsNotNone(data)
        self.assertEqual(data["usd"], 1891.0)
        self.assertEqual(data["toman"], 351_160_000.0)

    def test_parse_trx(self):
        data = parse_arz247_coin(ARZ247_MARKET_POST, "TRX")
        self.assertIsNotNone(data)
        self.assertAlmostEqual(data["usd"], 0.3344)
        self.assertEqual(data["toman"], 62100.0)
        self.assertEqual(data["change_24h_pct"], 0.34)

    def test_parse_ecogold_usdt(self):
        toman = parse_ecogold_usdt(ECOGOLD_POST)
        self.assertEqual(toman, 186077.0)

    def test_missing_symbol_returns_none(self):
        self.assertIsNone(parse_arz247_coin(ARZ247_MARKET_POST, "USDT"))

    def test_parse_cryptopricefeed_btc(self):
        usd = parse_cryptopricefeed_coin(CRYPTOPRICEFEED_POST, "BTC")
        self.assertEqual(usd, 79664.0)

    def test_parse_cryptopricefeed_eth(self):
        usd = parse_cryptopricefeed_coin(CRYPTOPRICEFEED_POST, "ETH")
        self.assertEqual(usd, 2520.16)

    def test_find_cryptopricefeed_post(self):
        found = find_cryptopricefeed_post(["noise", CRYPTOPRICEFEED_POST])
        self.assertIsNotNone(found)
        self.assertIn("#BTC", found)


if __name__ == "__main__":
    unittest.main()
