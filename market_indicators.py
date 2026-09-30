"""RSI, volatility, and trend from price-difference history."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("gold_bot")

MIN_HISTORY_FOR_RSI = 14
MIN_HISTORY_FOR_TREND = 2
TREND_SLOPE_THRESHOLD = 100

TREND_LABELS_FA = {
    "UPWARD": "صعودی ↗️",
    "DOWNWARD": "نزولی ↘️",
    "FLAT": "خنثی ➡️",
}


def trend_label_fa(trend: str) -> str:
    if trend in TREND_LABELS_FA:
        return TREND_LABELS_FA[trend]
    return trend


def calculate_market_indicators(differences: list[float]) -> dict[str, str | float]:
    """
    Compute trend, RSI, and volatility from chronological differences.
    The last element is treated as the current (live) point.
    """
    empty: dict[str, str | float] = {"trend": "N/A", "rsi": "N/A", "volatility": "N/A"}
    if len(differences) < 2:
        logger.debug("market_indicators: need at least 2 points, have %s", len(differences))
        return empty

    historical = differences[:-1] if len(differences) > 1 else list(differences)

    trend: str | float = "N/A"
    if len(historical) >= MIN_HISTORY_FOR_TREND:
        x = np.arange(len(historical))
        y = np.array(historical, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        if slope > TREND_SLOPE_THRESHOLD:
            trend = "UPWARD"
        elif slope < -TREND_SLOPE_THRESHOLD:
            trend = "DOWNWARD"
        else:
            trend = "FLAT"

    volatility: str | float = "N/A"
    if len(historical) >= MIN_HISTORY_FOR_TREND:
        volatility = round(float(np.std(historical)), 2)

    rsi: str | float = "N/A"
    if len(historical) >= MIN_HISTORY_FOR_RSI:
        deltas = np.diff(historical[-MIN_HISTORY_FOR_RSI:])
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = float(gains.mean()) if len(gains) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        else:
            rsi_val = 100.0 if avg_gain > 0 else 0.0
        rsi = round(rsi_val, 2)

    return {"trend": trend, "rsi": rsi, "volatility": volatility}
