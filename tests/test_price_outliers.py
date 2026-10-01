"""Unit tests for price_outliers detection."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from price_outliers import delete_price_outliers, detect_price_outliers


class TestPriceOutliers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        conn = sqlite3.connect(self.db)
        conn.execute(
            """
            CREATE TABLE price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                tala_price INTEGER,
                usd_price REAL,
                ounce_price REAL,
                fair_price REAL,
                difference REAL,
                source TEXT DEFAULT 'bot'
            )
            """
        )
        # Good row: usd 254k, ounce 4396 → fair ~27M, tala close to fair
        fair = 254_000 * 4396 / 41.5
        tala = int(fair + 200_000)
        conn.execute(
            "INSERT INTO price_history (timestamp, tala_price, usd_price, ounce_price, fair_price, difference, source) VALUES (?,?,?,?,?,?,?)",
            ("2026-01-01 10:00:00", tala, 254_000, 4396, fair, tala - fair, "bot"),
        )
        # Bad USD (USDT-scale mistake)
        conn.execute(
            "INSERT INTO price_history (timestamp, tala_price, usd_price, ounce_price, fair_price, difference, source) VALUES (?,?,?,?,?,?,?)",
            ("2026-01-01 11:00:00", tala, 186_077, 4396, fair, 5_000_000, "bot"),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_bad_usd(self):
        outliers = detect_price_outliers(str(self.db))
        self.assertGreaterEqual(len(outliers), 1)
        bad = [o for o in outliers if o.usd_price == 186_077]
        self.assertEqual(len(bad), 1)
        self.assertTrue(
            {"fair_mismatch", "difference_mismatch", "usd_out_of_range"} & set(bad[0].reasons)
        )

    def test_delete(self):
        outliers = detect_price_outliers(str(self.db))
        deleted = delete_price_outliers(str(self.db), outliers=outliers)
        self.assertEqual(deleted, len(outliers))
        self.assertEqual(len(detect_price_outliers(str(self.db))), 0)


if __name__ == "__main__":
    unittest.main()
