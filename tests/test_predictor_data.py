"""Unit tests for predictor training data loading."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from predictor import build_feature_matrix, load_daily_arrays, load_daily_bars, load_series_arrays


class TestPredictorData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "test.db")
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE price_history (
                timestamp TEXT,
                tala_price REAL,
                usd_price REAL,
                ounce_price REAL,
                fair_price REAL,
                difference REAL,
                source TEXT
            )
            """
        )
        rows = [
            ("2026-01-01 10:00:00", 100.0, 50.0, 2000.0, 90.0, 10.0, "bot"),
            ("2026-01-01 12:00:00", 110.0, 50.0, 2000.0, 90.0, 20.0, "unknown"),
            ("2026-01-02 10:00:00", 102.0, 51.0, 2010.0, 91.0, 11.0, "crawler"),
            ("2026-01-03 10:00:00", 103.0, 52.0, 2020.0, 92.0, 11.0, "crawler"),
        ]
        c.executemany(
            "INSERT INTO price_history VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_series_uses_all_rows_and_sources(self):
        data = load_series_arrays(self.db)
        self.assertEqual(len(data["tala"]), 4)

    def test_load_daily_bars_uses_full_day_range(self):
        bars = load_daily_bars(self.db)
        self.assertEqual(len(bars["tala"]), 3)
        self.assertAlmostEqual(bars["open"][0], 100.0)
        self.assertAlmostEqual(bars["high"][0], 110.0)
        self.assertAlmostEqual(bars["low"][0], 100.0)
        self.assertAlmostEqual(bars["tala"][0], 110.0)  # close
        self.assertAlmostEqual(bars["ticks"][0], 2.0)

    def test_load_daily_keeps_last_close_per_day(self):
        data = load_daily_arrays(self.db)
        self.assertEqual(len(data["tala"]), 3)
        self.assertAlmostEqual(data["tala"][0], 110.0)

    def test_return_targets(self):
        bars = load_daily_bars(self.db)
        _X, _names, targets = build_feature_matrix(bars)
        # 1d return from day0 close 110 -> day1 close 102
        self.assertAlmostEqual(targets[1][0], (102.0 - 110.0) / 110.0)


if __name__ == "__main__":
    unittest.main()
