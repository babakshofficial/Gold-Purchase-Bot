"""Edge-triggered price alerts with a per-signal cooldown.

A BUY/SELL or significant-move alert is sent when the actionable state
changes. Repeating the same state on the next monitor tick does not send
again. If the state flickers back inside the cooldown, the send waits
until the cooldown has passed and the edge is still pending.
"""

from __future__ import annotations

from datetime import datetime, timedelta

ALERT_COOLDOWN = timedelta(hours=6)
MARKET_ACTIONS = frozenset({"BUY", "SELL"})
SIG_HIGH = "high"
SIG_LOW = "low"
SIG_NONE = "none"


def significant_side(var: float, threshold: float) -> str:
    """Both sides of the fair-price gap. ``none`` when inside the band."""
    thr = abs(float(threshold))
    if var > thr:
        return SIG_HIGH
    if var < -thr:
        return SIG_LOW
    return SIG_NONE


def market_transition(
    prev_observed: str | None,
    current: str | None,
    last_sent_kind: str | None,
    last_sent_at: datetime | None,
    now: datetime,
    cooldown: timedelta = ALERT_COOLDOWN,
) -> tuple[bool, str, str | None, datetime | None]:
    """Return (send, next_observed, next_sent_kind, next_sent_at)."""
    current_state = current if current in MARKET_ACTIONS else "HOLD"
    prev = prev_observed or "HOLD"
    if current_state == prev:
        return False, prev, last_sent_kind, last_sent_at
    if current_state not in MARKET_ACTIONS:
        return False, current_state, last_sent_kind, last_sent_at
    if (
        last_sent_kind == current_state
        and last_sent_at is not None
        and (now - last_sent_at) < cooldown
    ):
        return False, prev, last_sent_kind, last_sent_at
    return True, current_state, current_state, now


def significant_transition(
    prev_observed: str | None,
    current: str | None,
    last_sent_side: str | None,
    last_sent_at: datetime | None,
    now: datetime,
    cooldown: timedelta = ALERT_COOLDOWN,
) -> tuple[bool, str, str | None, datetime | None]:
    """Return (send, next_observed, next_sent_side, next_sent_at)."""
    current_side = current if current in (SIG_HIGH, SIG_LOW) else SIG_NONE
    prev = prev_observed or SIG_NONE
    if current_side == prev:
        return False, prev, last_sent_side, last_sent_at
    if current_side == SIG_NONE:
        return False, SIG_NONE, last_sent_side, last_sent_at
    if (
        last_sent_side == current_side
        and last_sent_at is not None
        and (now - last_sent_at) < cooldown
    ):
        return False, prev, last_sent_side, last_sent_at
    return True, current_side, current_side, now
