"""Holdout metric definitions."""

import unittest

import numpy as np

from predictor import _score_holdout


class TestHoldoutMetrics(unittest.TestCase):
    def test_mape_is_not_mae(self):
        pred = np.array([0.02, -0.01])
        actual = np.array([0.04, -0.02])
        scores = _score_holdout(pred, actual)
        mae_pct = float(np.mean(np.abs(pred - actual)) * 100.0)
        self.assertAlmostEqual(scores["mae"], mae_pct)
        # |0.02-0.04|/0.04 = 0.5, | -0.01-(-0.02) | / 0.02 = 0.5 → MAPE 50
        self.assertAlmostEqual(scores["mape"], 50.0)
        self.assertNotAlmostEqual(scores["mape"], scores["mae"])

    def test_directional_accuracy_ignores_flat_actuals(self):
        pred = np.array([0.01, -0.02, 0.03])
        actual = np.array([0.02, -0.01, 0.0])
        scores = _score_holdout(pred, actual)
        self.assertAlmostEqual(scores["dir_acc"], 100.0)


if __name__ == "__main__":
    unittest.main()
