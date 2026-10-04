"""Fetch gold/silver/copper Toman prices from @Zarpay724 Telegram channel."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("gold_bot")

METALS_CHANNEL_USERNAME = "Zarpay724"
METALS_CHANNEL_URL = f"https://t.me/s/{METALS_CHANNEL_USERNAME}"
DEFAULT_DB = "gold_bot.db"
DEFAULT_MESSAGES_TO_SCAN = 10

GOLD_TOMAN_MIN = 4_000_000
GOLD_TOMAN_MAX = 60_000_000
SILVER_TOMAN_MIN = 50_000
SILVER_TOMAN_MAX = 5_000_000
COPPER_TOMAN_MIN = 500
COPPER_TOMAN_MAX = 100_000

MESGHAL_GRAMS = 4.6083


def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")


def _parse_int_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


@dataclass(frozen=True)
class MetalsPrices:
    gold_per_gram: float
    silver_per_gram: float
    copper_per_gram: float
    source: str = "zarpay724"
    stale: bool = False

    @property
    def silver_per_mesghal(self) -> float:
        return self.silver_per_gram * MESGHAL_GRAMS

    @property
    def copper_per_kg(self) -> float:
        return self.copper_per_gram * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "gold_per_gram": self.gold_per_gram,
            "silver_per_gram": self.silver_per_gram,
            "copper_per_gram": self.copper_per_gram,
            "silver_per_mesghal": self.silver_per_mesghal,
            "copper_per_kg": self.copper_per_kg,
            "source": self.source,
            "stale": self.stale,
        }


def is_plausible_gold_toman(value: float | None) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return GOLD_TOMAN_MIN <= v <= GOLD_TOMAN_MAX


def is_plausible_silver_toman(value: float | None) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return SILVER_TOMAN_MIN <= v <= SILVER_TOMAN_MAX


def is_plausible_copper_toman(value: float | None) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return COPPER_TOMAN_MIN <= v <= COPPER_TOMAN_MAX


def is_plausible_metals_prices(gold: float, silver: float, copper: float) -> bool:
    return (
        is_plausible_gold_toman(gold)
        and is_plausible_silver_toman(silver)
        and is_plausible_copper_toman(copper)
    )


def parse_zarpay_post(text: str) -> tuple[float, float, float] | None:
    """Parse Zarpay724 post: gold 18k, silver, copper — all Toman per gram."""
    text = normalize(text)
    gold_m = re.search(
        r"طل(?:ای|ا)\s*18\s*عیار\s*:\s*([\d,]+)\s*تومان",
        text,
    )
    silver_m = re.search(r"نقره\s*:\s*([\d,]+)\s*تومان", text)
    copper_m = re.search(r"مس\s*:\s*([\d,]+)\s*تومان", text)
    if not gold_m or not silver_m or not copper_m:
        return None
    gold = _parse_int_amount(gold_m.group(1))
    silver = _parse_int_amount(silver_m.group(1))
    copper = _parse_int_amount(copper_m.group(1))
    if not is_plausible_metals_prices(gold, silver, copper):
        return None
    return gold, silver, copper


def _messages_from_channel_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    return [m.get_text("\n", strip=True) for m in msgs]


def scan_messages_for_metals(
    messages: list[str],
    *,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
) -> tuple[float, float, float] | None:
    limit = min(messages_to_scan, len(messages))
    checked = 0
    for idx in range(len(messages) - 1, -1, -1):
        if checked >= limit:
            break
        msg_text = messages[idx]
        if not msg_text or len(msg_text) < 30:
            continue
        checked += 1
        parsed = parse_zarpay_post(msg_text)
        if parsed is not None:
            return parsed
    return None


def fetch_metals_from_channel(
    session: requests.Session,
    timeout: float | tuple[float, float],
    *,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
) -> tuple[float, float, float] | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = session.get(METALS_CHANNEL_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    messages = _messages_from_channel_html(r.text)
    if not messages:
        return None
    return scan_messages_for_metals(messages, messages_to_scan=messages_to_scan)


def save_metals_price_history(
    gold: float,
    silver: float,
    copper: float,
    *,
    db_path: str = DEFAULT_DB,
    source: str = "zarpay724",
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            """INSERT INTO metals_price_history
               (gold_price, silver_price, copper_price, source)
               VALUES (?, ?, ?, ?)""",
            (gold, silver, copper, source),
        )
        conn.commit()
    finally:
        conn.close()


def last_metals_from_db(db_path: str = DEFAULT_DB) -> MetalsPrices | None:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            """SELECT gold_price, silver_price, copper_price, source
               FROM metals_price_history
               ORDER BY timestamp DESC LIMIT 30"""
        )
        for gold, silver, copper, source in c.fetchall():
            if is_plausible_metals_prices(gold, silver, copper):
                return MetalsPrices(
                    gold_per_gram=float(gold),
                    silver_per_gram=float(silver),
                    copper_per_gram=float(copper),
                    source=source or "db",
                    stale=True,
                )
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return None


def fetch_metals_prices(
    session: requests.Session | None = None,
    *,
    db_path: str = DEFAULT_DB,
    max_attempts: int = 3,
    timeout: float | tuple[float, float] = 30,
    backoff_factor: float = 2,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
    persist: bool = True,
) -> MetalsPrices:
    """Live fetch from @Zarpay724, else last plausible row in metals_price_history."""
    own_session = session is None
    if own_session:
        session = requests.Session()

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            triple = fetch_metals_from_channel(
                session,
                timeout,
                messages_to_scan=messages_to_scan,
            )
            if triple is not None:
                gold, silver, copper = triple
                if persist:
                    save_metals_price_history(gold, silver, copper, db_path=db_path)
                logger.info(
                    "Metals from @%s: gold=%s silver=%s copper=%s (attempt %s)",
                    METALS_CHANNEL_USERNAME,
                    gold,
                    silver,
                    copper,
                    attempt + 1,
                )
                return MetalsPrices(
                    gold_per_gram=gold,
                    silver_per_gram=silver,
                    copper_per_gram=copper,
                    source="zarpay724",
                    stale=False,
                )
            logger.warning(
                "No valid @%s metals post in last %s messages (attempt %s)",
                METALS_CHANNEL_USERNAME,
                messages_to_scan,
                attempt + 1,
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("Metals fetch request error attempt %s: %s", attempt + 1, e)
        except Exception as e:
            last_error = e
            logger.error("Metals fetch unexpected error attempt %s: %s", attempt + 1, e)

        if attempt < max_attempts - 1:
            time.sleep(backoff_factor**attempt)

    fallback = last_metals_from_db(db_path)
    if fallback is not None:
        logger.info("Using last plausible metals from DB")
        return fallback

    if last_error:
        raise RuntimeError(f"Failed to fetch metals prices: {last_error}")
    raise RuntimeError(
        f"Metals prices not found on @{METALS_CHANNEL_USERNAME} and no DB fallback"
    )
