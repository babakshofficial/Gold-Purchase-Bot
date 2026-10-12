"""Edge-trigger and cooldown rules for price alerts."""

import unittest
from datetime import datetime, timedelta

from alert_policy import (
    ALERT_COOLDOWN,
    market_transition,
    significant_side,
    significant_transition,
)


class TestAlertPolicy(unittest.TestCase):
    def test_same_buy_does_not_repeat(self):
        now = datetime(2026, 10, 12, 9, 0)
        send, observed, kind, sent_at = market_transition("BUY", "BUY", "BUY", now, now)
        self.assertFalse(send)
        self.assertEqual(observed, "BUY")
        self.assertEqual(kind, "BUY")
        self.assertEqual(sent_at, now)

    def test_hold_then_buy_sends(self):
        now = datetime(2026, 10, 12, 9, 0)
        send, observed, kind, sent_at = market_transition("HOLD", "BUY", None, None, now)
        self.assertTrue(send)
        self.assertEqual((observed, kind, sent_at), ("BUY", "BUY", now))

    def test_flicker_inside_cooldown_waits(self):
        sent = datetime(2026, 10, 12, 9, 0)
        now = sent + timedelta(hours=1)
        send, observed, kind, sent_at = market_transition(
            "HOLD", "BUY", "BUY", sent, now, ALERT_COOLDOWN
        )
        self.assertFalse(send)
        self.assertEqual(observed, "HOLD")
        self.assertEqual(sent_at, sent)
        self.assertEqual(kind, "BUY")

    def test_flicker_after_cooldown_sends(self):
        sent = datetime(2026, 10, 12, 9, 0)
        now = sent + ALERT_COOLDOWN
        send, observed, _, _ = market_transition("HOLD", "BUY", "BUY", sent, now, ALERT_COOLDOWN)
        self.assertTrue(send)
        self.assertEqual(observed, "BUY")

    def test_significant_move_both_sides(self):
        self.assertEqual(significant_side(800_000, 700_000), "high")
        self.assertEqual(significant_side(-800_000, 700_000), "low")
        self.assertEqual(significant_side(100_000, 700_000), "none")

    def test_significant_edge(self):
        now = datetime(2026, 10, 12, 9, 0)
        send, observed, side, _ = significant_transition("none", "low", None, None, now)
        self.assertTrue(send)
        self.assertEqual(observed, "low")
        self.assertEqual(side, "low")
        send_again, _, _, _ = significant_transition("low", "low", "low", now, now)
        self.assertFalse(send_again)


if __name__ == "__main__":
    unittest.main()
