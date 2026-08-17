"""Fetch and parse crypto prices from Telegram public channel pages."""

import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("gold_bot")

CRYPTO_CHANNEL_USERNAME = "arz_247"
CRYPTO_CHANNEL_URL = f"https://t.me/s/{CRYPTO_CHANNEL_USERNAME}"
GOLD_CHANNEL_URL = "https://t.me/s/ecogold_ir"

STAGE1_SYMBOLS = ("BTC", "ETH", "TRX", "USDT")
MARKET_POST_MARKER = "ارز منتخب بازار"
RANDOM_POST_MARKER = "انتخاب تصادفی"

REQUEST_TIMEOUT = 30
MAX_FETCH_ATTEMPTS = 5
RETRY_BACKOFF_FACTOR = 2
MESSAGES_TO_SCAN = 5


def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")


def parse_toman_value(raw: str, unit: str | None) -> float:
    """Convert parsed Toman text (with optional million/billion unit) to float."""
    value = float(raw.replace(",", ""))
    if unit == "میلیارد":
        return value * 1_000_000_000
    if unit == "میلیون":
        return value * 1_000_000
    return value


def find_market_post(messages: list[str]) -> str | None:
    """Return the newest post containing the market marker (not random selection)."""
    for text in messages:
        if not text or len(text) < 20:
            continue
        if MARKET_POST_MARKER in text and RANDOM_POST_MARKER not in text:
            return text
    return None


def parse_arz247_coin(text: str, symbol: str) -> dict[str, Any] | None:
    """Parse one coin from an arz_247 market post."""
    text = normalize(text)
    pattern = (
        rf"\({re.escape(symbol)}\)"
        rf".*?"
        rf"💵\s*\$([\d,.]+)"
        rf"\s*\|\s*💰\s*([\d,.]+)"
        rf"\s*(میلیارد|میلیون)?"
        rf"\s*تومان"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    usd = float(match.group(1).replace(",", ""))
    toman = parse_toman_value(match.group(2), match.group(3))

    change_pct = None
    line_start = match.start()
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    change_match = re.search(r"[🔴🟢]\s*([+-]?[\d.]+)%?", line)
    if change_match:
        change_pct = float(change_match.group(1))

    return {
        "usd": usd,
        "toman": toman,
        "change_24h_pct": change_pct,
        "source": "arz_247",
    }


def parse_ecogold_usdt(text: str) -> float | None:
    """Parse USDT (تتر) Toman price from ecogold_ir post."""
    text = normalize(text)
    match = re.search(r"تتر:\s*([\d,]+)\s*تومان", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def fetch_recent_posts(url: str, limit: int = MESSAGES_TO_SCAN, session: requests.Session | None = None) -> list[str]:
    """Fetch the last N message texts from a Telegram public channel page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    http = session or requests
    r = http.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    if not msgs:
        raise RuntimeError(f"No messages found at {url}")

    texts = []
    for i in range(min(limit, len(msgs))):
        msg_text = msgs[-(i + 1)].get_text("\n", strip=True)
        if msg_text:
            texts.append(msg_text)
    return texts


def _fetch_with_retries(url: str, limit: int, session: requests.Session) -> list[str]:
    last_error = None
    for attempt in range(MAX_FETCH_ATTEMPTS):
        try:
            return fetch_recent_posts(url, limit=limit, session=session)
        except Exception as e:
            last_error = e
            logger.warning(f"Fetch attempt {attempt + 1}/{MAX_FETCH_ATTEMPTS} failed for {url}: {e}")
            if attempt < MAX_FETCH_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_FACTOR ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_FETCH_ATTEMPTS} attempts: {last_error}")


def fetch_arz247_market_prices(session: requests.Session | None = None) -> dict[str, dict[str, Any]]:
    """Fetch BTC/ETH/TRX/USDT from the latest arz_247 market post."""
    session = session or requests.Session()
    messages = _fetch_with_retries(CRYPTO_CHANNEL_URL, MESSAGES_TO_SCAN, session)
    market_post = find_market_post(messages)
    if not market_post:
        raise RuntimeError(
            f"No post containing '{MARKET_POST_MARKER}' found in last {MESSAGES_TO_SCAN} messages"
        )

    result: dict[str, dict[str, Any]] = {}
    for symbol in STAGE1_SYMBOLS:
        if symbol == "USDT":
            continue
        parsed = parse_arz247_coin(market_post, symbol)
        if parsed:
            result[symbol] = parsed
        else:
            logger.warning(f"{symbol} not found in arz_247 market post")
    return result


def fetch_ecogold_usdt_price(session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch USDT Toman price from ecogold_ir (fallback source)."""
    session = session or requests.Session()
    messages = _fetch_recent_posts_safe(GOLD_CHANNEL_URL, MESSAGES_TO_SCAN, session)
    for text in messages:
        toman = parse_ecogold_usdt(text)
        if toman is not None:
            return {"toman": toman, "source": "ecogold_ir"}
    raise RuntimeError("USDT (تتر) not found in ecogold_ir posts")


def _fetch_recent_posts_safe(url: str, limit: int, session: requests.Session) -> list[str]:
    try:
        return _fetch_with_retries(url, limit, session)
    except Exception:
        return fetch_recent_posts(url, limit=limit, session=session)


def fetch_crypto_prices(usd_toman: float | None = None) -> dict[str, dict[str, Any]]:
    """
    Fetch stage-1 crypto prices.
    Primary: arz_247 market post for BTC/ETH/TRX.
    Fallback: ecogold_ir for USDT when missing from arz_247.
    """
    session = requests.Session()
    prices: dict[str, dict[str, Any]] = {}

    try:
        prices.update(fetch_arz247_market_prices(session))
    except Exception as e:
        logger.error(f"Failed to fetch arz_247 crypto prices: {e}")

    usdt_from_arz = None
    if "USDT" not in prices:
        try:
            messages = _fetch_recent_posts_safe(CRYPTO_CHANNEL_URL, MESSAGES_TO_SCAN, session)
            market_post = find_market_post(messages)
            if market_post:
                usdt_from_arz = parse_arz247_coin(market_post, "USDT")
                if usdt_from_arz:
                    prices["USDT"] = usdt_from_arz
        except Exception:
            pass

    if "USDT" not in prices:
        try:
            eco = fetch_ecogold_usdt_price(session)
            usdt_entry: dict[str, Any] = {
                "toman": eco["toman"],
                "source": eco["source"],
                "usd": None,
                "change_24h_pct": None,
            }
            if usd_toman and usd_toman > 0:
                usdt_entry["usd"] = eco["toman"] / usd_toman
            prices["USDT"] = usdt_entry
        except Exception as e:
            logger.warning(f"USDT fallback from ecogold_ir failed: {e}")

    return prices
