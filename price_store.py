"""price_history writes shared by the bot and the crawler."""

from __future__ import annotations

import logging

from db import connection

logger = logging.getLogger("gold_bot")

# Skip a second insert of the same prices inside this window.
SNAPSHOT_DEDUPE_SEC = 300


def save_market_snapshot(
    tala,
    usd,
    ounce,
    fair,
    diff,
    *,
    source: str,
    rsi=None,
    volatility=None,
    trend=None,
    db_path: str | None = None,
) -> bool:
    """Insert one price row. Identical prices inside five minutes stay one row.

    If the existing row has no RSI and this write includes indicators, those
    fields are filled in. Returns True when a new row was inserted.
    """
    window = f"-{int(SNAPSHOT_DEDUPE_SEC)} seconds"
    with connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, rsi FROM price_history
            WHERE tala_price = ? AND usd_price = ? AND ounce_price = ?
              AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (tala, usd, ounce, window),
        )
        existing = cursor.fetchone()
        if existing:
            row_id, existing_rsi = existing
            if rsi is not None and existing_rsi is None:
                cursor.execute(
                    """
                    UPDATE price_history
                    SET fair_price = ?, difference = ?, rsi = ?, volatility = ?, trend = ?, source = ?
                    WHERE id = ?
                    """,
                    (fair, diff, rsi, volatility, trend, source, row_id),
                )
                logger.info("Updated recent price row %s with %s indicators", row_id, source)
            else:
                logger.info("Skipped duplicate %s price snapshot", source)
            return False

        cursor.execute(
            """
            INSERT INTO price_history
                (tala_price, usd_price, ounce_price, fair_price, difference,
                 rsi, volatility, trend, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tala, usd, ounce, fair, diff, rsi, volatility, trend, source),
        )
        return True
