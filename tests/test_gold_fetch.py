"""Gold channel post parser (no network)."""

import unittest

from gold_fetch import parse_gold_post


SAMPLE = """
قیمت طلا
طلای 18 عیار : 7,850,000
اونس طلا : 2,650.50
"""

PERSIAN_DIGITS = """
طلای ۱۸ عیار : ۷٬۸۵۰٬۰۰۰
اونس طلا : ۲٬۶۵۰.۵۰
"""


class TestGoldFetch(unittest.TestCase):
    def test_parse_ascii_post(self):
        self.assertEqual(parse_gold_post(SAMPLE), (7_850_000, 2650.50))

    def test_parse_persian_digits(self):
        self.assertEqual(parse_gold_post(PERSIAN_DIGITS), (7_850_000, 2650.50))

    def test_missing_fields(self):
        self.assertIsNone(parse_gold_post("فقط یک متن کوتاه بدون قیمت"))


if __name__ == "__main__":
    unittest.main()
