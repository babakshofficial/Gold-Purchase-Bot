"""Fetch and parse USD/Toman rate from Telegram public channel pages."""

import logging
import re
import time
from typing import Callable

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("gold_bot")

USD_CHANNEL_PRIMARY = "nerkhedular"
USD_CHANNEL_FALLBACK = "tgjucurrency"
USD_CHANNEL_PRIMARY_URL = f"https://t.me/s/{USD_CHANNEL_PRIMARY}"
USD_CHANNEL_FALLBACK_URL = f"https://t.me/s/{USD_CHANNEL_FALLBACK}"

NERKHEDULAR_MARKER = "دلار فردایی تهران"
TGJU_MARKER = "قیمت ارزهای آزاد"

DEFAULT_MESSAGES_TO_SCAN = 10

# Free-market USD in Toman (reject Rial/10 mistakes like 800, USDT-scale, etc.)
USD_TOMAN_MIN = 50_000
USD_TOMAN_MAX = 900_000
OUNCE_USD_MIN = 1_500
OUNCE_USD_MAX = 8_000
TALA_TOMAN_MIN = 4_000_000
TALA_TOMAN_MAX = 50_000_000
FAIR_TO_TALA_MIN_RATIO = 0.55
FAIR_TO_TALA_MAX_RATIO = 1.85


def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")


def _parse_toman_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def is_plausible_usd_toman(value: float | None) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return USD_TOMAN_MIN <= v <= USD_TOMAN_MAX


def is_plausible_market_prices(tala: float, usd_toman: float, ounce: float) -> bool:
    """Gold market, USD, and fair = USD×ounce/41.5 must be internally consistent."""
    if not is_plausible_usd_toman(usd_toman):
        return False
    try:
        tala_f = float(tala)
        ounce_f = float(ounce)
    except (TypeError, ValueError):
        return False
    if not (TALA_TOMAN_MIN <= tala_f <= TALA_TOMAN_MAX):
        return False
    if not (OUNCE_USD_MIN <= ounce_f <= OUNCE_USD_MAX):
        return False
    fair = float(usd_toman) * ounce_f / 41.5
    if fair <= 0:
        return False
    ratio = fair / tala_f
    return FAIR_TO_TALA_MIN_RATIO <= ratio <= FAIR_TO_TALA_MAX_RATIO


def parse_nerkhedular_post(text: str) -> float | None:
    """Parse tomorrow Tehran USD from @nerkhedular (prices already in Toman)."""
    text = normalize(text)
    if NERKHEDULAR_MARKER not in text:
        return None

    deal = re.search(
        rf"{NERKHEDULAR_MARKER}.*?([\d,]+)\s*معامله",
        text,
        re.DOTALL,
    )
    if deal:
        return _parse_toman_amount(deal.group(1))

    buy = re.search(
        rf"{NERKHEDULAR_MARKER}.*?([\d,]+)\s*خ\S*رید",
        text,
        re.DOTALL,
    )
    if buy:
        return _parse_toman_amount(buy.group(1))

    sell = re.search(
        rf"{NERKHEDULAR_MARKER}.*?([\d,]+)\s*فروش",
        text,
        re.DOTALL,
    )
    if sell:
        return _parse_toman_amount(sell.group(1))

    return None


def parse_tgjucurrency_post(text: str) -> float | None:
    """Parse free-market USD from @tgjucurrency (line is in Rial, converted to Toman)."""
    text = normalize(text)
    if TGJU_MARKER not in text:
        return None

    usd_line_match = re.search(r"🇺🇸\s*دلار\s*[:\s]*\s*([\d,]+)\s*ریال", text)
    if not usd_line_match:
        return None

    usd_rial = int(usd_line_match.group(1).replace(",", ""))
    return usd_rial / 10


def parse_usd_post(text: str) -> float | None:
    """Try primary then fallback parsers on a single message body."""
    return parse_nerkhedular_post(text) or parse_tgjucurrency_post(text)


def _messages_from_channel_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    return [m.get_text("\n", strip=True) for m in msgs]


def _scan_messages(
    messages: list[str],
    parser: Callable[[str], float | None],
    *,
    min_len: int = 20,
    newest_first: bool = True,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
) -> float | None:
    indices = range(len(messages) - 1, -1, -1) if newest_first else range(len(messages))
    limit = min(messages_to_scan, len(messages))
    checked = 0
    for idx in indices:
        if checked >= limit:
            break
        msg_text = messages[idx]
        if not msg_text or len(msg_text) < min_len:
            continue
        checked += 1
        result = parser(msg_text)
        if result is not None:
            if not is_plausible_usd_toman(result):
                logger.warning(
                    "Skipping implausible USD %s Toman (expected %s–%s)",
                    result,
                    USD_TOMAN_MIN,
                    USD_TOMAN_MAX,
                )
                continue
            return result
    return None


def fetch_usd_from_channel(
    url: str,
    parser: Callable[[str], float | None],
    session: requests.Session,
    timeout: tuple[float, float] | float,
    *,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
) -> float | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    messages = _messages_from_channel_html(r.text)
    if not messages:
        return None
    return _scan_messages(messages, parser, messages_to_scan=messages_to_scan)


def fetch_usd_toman(
    session: requests.Session | None = None,
    *,
    max_attempts: int = 5,
    timeout: tuple[float, float] | float = 30,
    backoff_factor: float = 2,
    messages_to_scan: int = DEFAULT_MESSAGES_TO_SCAN,
) -> float:
    """Fetch USD in Toman: @nerkhedular first, then @tgjucurrency."""
    own_session = session is None
    if own_session:
        session = requests.Session()

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            price = fetch_usd_from_channel(
                USD_CHANNEL_PRIMARY_URL,
                parse_nerkhedular_post,
                session,
                timeout,
                messages_to_scan=messages_to_scan,
            )
            if price is not None:
                logger.info(
                    "USD from @%s: %s Toman (attempt %s)",
                    USD_CHANNEL_PRIMARY,
                    price,
                    attempt + 1,
                )
                return price

            logger.warning(
                "No @%s USD post in last %s messages, trying fallback (attempt %s)",
                USD_CHANNEL_PRIMARY,
                messages_to_scan,
                attempt + 1,
            )

            price = fetch_usd_from_channel(
                USD_CHANNEL_FALLBACK_URL,
                parse_tgjucurrency_post,
                session,
                timeout,
                messages_to_scan=messages_to_scan,
            )
            if price is not None:
                logger.info(
                    "USD from @%s (fallback): %s Toman (attempt %s)",
                    USD_CHANNEL_FALLBACK,
                    price,
                    attempt + 1,
                )
                return price

            logger.warning(
                "USD not found on @%s or @%s (attempt %s)",
                USD_CHANNEL_PRIMARY,
                USD_CHANNEL_FALLBACK,
                attempt + 1,
            )
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning("USD fetch timeout attempt %s: %s", attempt + 1, e)
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("USD fetch request error attempt %s: %s", attempt + 1, e)
        except Exception as e:
            last_error = e
            logger.error("USD fetch unexpected error attempt %s: %s", attempt + 1, e)

        if attempt < max_attempts - 1:
            wait_time = backoff_factor ** attempt
            logger.info("Waiting %s seconds before next USD fetch attempt...", wait_time)
            time.sleep(wait_time)

    if isinstance(last_error, requests.exceptions.Timeout):
        raise requests.exceptions.ReadTimeout(
            f"Failed to fetch USD after {max_attempts} attempts due to timeout."
        )
    if last_error:
        raise RuntimeError(f"Failed to fetch USD after {max_attempts} attempts: {last_error}")
    raise RuntimeError(
        f"USD price not found on @{USD_CHANNEL_PRIMARY} or @{USD_CHANNEL_FALLBACK} "
        f"after {max_attempts} attempts."
    )
