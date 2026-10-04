"""Unit tests for metals_fetch (no network)."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from metals_fetch import (
    MESGHAL_GRAMS,
    MetalsPrices,
    is_plausible_metals_prices,
    last_metals_from_db,
    parse_zarpay_post,
    save_metals_price_history,
)

ZARPAY_POST = """
زرپی | معامله‌ی 24 ساعته طلا، نقره و مس

🌕 طلای ۱۸ عیار: 26,272,444 تومان

🪙 نقره: 516,150 تومان

🟤 مس: 3,190 تومان

🕰 زمان: 1405/07/12 - 06:43
"""


class TestMetalsFetch(unittest.TestCase):
    def test_parse_zarpay_post(self):
        parsed = parse_zarpay_post(ZARPAY_POST)
        self.assertIsNotNone(parsed)
        gold, silver, copper = parsed
        self.assertEqual(gold, 26_272_444.0)
        self.assertEqual(silver, 516_150.0)
        self.assertEqual(copper, 3_190.0)

    def test_parse_skips_incomplete(self):
        self.assertIsNone(parse_zarpay_post("فقط طلا: 1000"))

    def test_plausibility_rejects_bad_usd_scale_gold(self):
        self.assertFalse(is_plausible_metals_prices(800, 516_150, 3_190))

    def test_metals_prices_derived_units(self):
        m = MetalsPrices(26_272_444, 516_150, 3_190)
        self.assertAlmostEqual(m.copper_per_kg, 3_190_000.0)
        self.assertAlmostEqual(m.silver_per_mesghal, 516_150 * MESGHAL_GRAMS, places=0)

    def test_db_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "test.db")
            conn = sqlite3.connect(db)
            conn.execute(
                """CREATE TABLE metals_price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    gold_price REAL,
                    silver_price REAL,
                    copper_price REAL,
                    source TEXT
                )"""
            )
            conn.commit()
            conn.close()
            save_metals_price_history(
                26_000_000, 500_000, 3_000, db_path=db, source="test"
            )
            row = last_metals_from_db(db)
            self.assertIsNotNone(row)
            self.assertTrue(row.stale)
            self.assertEqual(row.silver_per_gram, 500_000.0)


if __name__ == "__main__":
    unittest.main()
