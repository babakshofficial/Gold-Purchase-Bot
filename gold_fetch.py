"""Fetch and parse 18k gold posts from the public Telegram channel."""

from __future__ import annotations

import logging
import re
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pricing import normalize

logger = logging.getLogger("gold_bot")

GOLD_CHANNEL_USERNAME = "ecogold_ir"
GOLD_CHANNEL_URL = f"https://t.me/s/{GOLD_CHANNEL_USERNAME}"
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_FACTOR = 2
DEFAULT_TIMEOUT = (10, 60)
MESSAGES_TO_SCAN = 10


def parse_gold_post(text: str) -> tuple[int, float] | None:
    """Return (tala toman per gram, ounce USD) from an ecogold-style post."""
    text = normalize(text)
    tala = re.search(r"طلای\s*18\s*عیار[\s\n]*:\s*([\d,]+)", text)
    ounce = re.search(r"اونس\s*طلا[\s\n]*:\s*([\d,.]+)", text)
    if not tala or not ounce:
        return None
    return (
        int(tala.group(1).replace(",", "")),
        float(ounce.group(1).replace(",", "")),
    )


def _session_with_retries(max_attempts: int, backoff_factor: float) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max(0, max_attempts - 1),
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_and_parse_gold(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    *,
    timeout: tuple[float, float] | float = DEFAULT_TIMEOUT,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    session: requests.Session | None = None,
    url: str = GOLD_CHANNEL_URL,
    messages_to_scan: int = MESSAGES_TO_SCAN,
) -> tuple[int, float]:
    """Fetch gold data, trying several recent posts and HTTP retries."""
    headers = {"User-Agent": "Mozilla/5.0"}
    session = session or _session_with_retries(max_attempts, backoff_factor)
    num_msgs_to_check = messages_to_scan

    for attempt in range(max_attempts):
        try:
            logger.info(
                "Attempt %s/%s to fetch Gold data from %s",
                attempt + 1,
                max_attempts,
                url,
            )
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            msgs = soup.select("div.tgme_widget_message_text")
            if not msgs:
                raise RuntimeError("No messages found in Gold channel")

            num_msgs_to_check = min(messages_to_scan, len(msgs))
            for i in range(num_msgs_to_check):
                msg_text = msgs[-(i + 1)].get_text("\n", strip=True)
                if msg_text and len(msg_text) > 20:
                    result = parse_gold_post(msg_text)
                    if result is not None:
                        logger.info(
                            "Successfully parsed Gold data (Tala: %s, Ounce: %s) from post #%s (latest being #1), attempt %s.",
                            result[0],
                            result[1],
                            i + 1,
                            attempt + 1,
                        )
                        return result
                else:
                    logger.debug(
                        "fetch_and_parse_gold: Skipping empty/short message #%s, attempt %s",
                        i + 1,
                        attempt + 1,
                    )

            logger.warning(
                "Gold price not found in the last %s posts, attempt %s.",
                num_msgs_to_check,
                attempt + 1,
            )
        except requests.exceptions.Timeout as exc:
            logger.warning("Request timeout on attempt %s for Gold data: %s", attempt + 1, exc)
            if attempt == max_attempts - 1:
                raise requests.exceptions.ReadTimeout(
                    f"Failed to fetch Gold data after {max_attempts} attempts due to timeout."
                ) from exc
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on attempt %s for Gold data: %s", attempt + 1, exc)
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to fetch Gold data after {max_attempts} attempts: {exc}"
                ) from exc
        except Exception as exc:
            logger.error("Unexpected error on attempt %s for Gold data: %s", attempt + 1, exc)
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to fetch Gold data after {max_attempts} attempts due to an unexpected error: {exc}"
                ) from exc

        if attempt < max_attempts - 1:
            wait_time = backoff_factor ** attempt
            logger.info("Waiting %s seconds before next Gold fetch attempt...", wait_time)
            time.sleep(wait_time)

    raise RuntimeError(
        f"Gold price not found in the last {num_msgs_to_check} posts after {max_attempts} attempts."
    )
