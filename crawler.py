# crawler_service.py
import logging
import time

import requests

from crypto_fetch import CRYPTO_CHANNEL_USERNAME, fetch_crypto_prices, save_crypto_price_history
from db import connect
from gold_fetch import fetch_and_parse_gold
from market_indicators import calculate_market_indicators
from metals_fetch import METALS_CHANNEL_USERNAME, fetch_metals_prices
from pricing import fair_gold_price
from price_store import save_market_snapshot
from usd_fetch import (
    USD_CHANNEL_FALLBACK,
    USD_CHANNEL_PRIMARY,
    fetch_usd_toman,
    is_plausible_market_prices,
    last_plausible_usd_from_db,
)

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_crawler")

# ================= CONFIG ==================
GOLD_CHANNEL_USERNAME = "ecogold_ir"
USD_CHANNEL_USERNAME = USD_CHANNEL_PRIMARY
GOLD_CHANNEL_URL = f"https://t.me/s/{GOLD_CHANNEL_USERNAME}"
REQUEST_TIMEOUT = 10
MAX_FETCH_ATTEMPTS = 5
RETRY_BACKOFF_FACTOR = 2
TREND_HOURS = 6
CRAWL_INTERVAL_SEC = 600


# ================= DATABASE HELPERS FOR CRAWLER =================
def save_price_history_crawler(tala, usd, ounce, fair, diff, rsi, volatility, trend):
    """Saves price data to the database with 'crawler' source."""
    save_market_snapshot(
        tala,
        usd,
        ounce,
        fair,
        diff,
        source="crawler",
        rsi=rsi,
        volatility=volatility,
        trend=trend,
    )


def get_price_history_for_analysis_crawler(hours=TREND_HOURS):
    window = f"-{int(hours)} hours"
    conn = connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, difference
               FROM price_history
               WHERE timestamp >= datetime('now', ?)
               AND source = 'crawler'
               ORDER BY timestamp ASC""",
            (window,),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def calculate_rsi_and_volatility_and_trend_crawler(differences):
    metrics = calculate_market_indicators([float(d) for d in differences])
    return metrics["rsi"], metrics["volatility"], metrics["trend"]


# ================= FETCH HELPERS =================
def fetch_and_parse_usd(max_attempts: int = MAX_FETCH_ATTEMPTS):
    return fetch_usd_toman(
        max_attempts=max_attempts,
        timeout=REQUEST_TIMEOUT,
        backoff_factor=RETRY_BACKOFF_FACTOR,
    )


def usd_toman_for_dependent_fetches() -> float | None:
    try:
        return fetch_and_parse_usd()
    except Exception as e:
        logger.warning("Crawler USD live fetch failed: %s", e)
    return last_plausible_usd_from_db()


def crawl_gold_market() -> bool:
    """Ecogold + USD → price_history (fair value, RSI, trend)."""
    try:
        tala, ounce = fetch_and_parse_gold()
        usd_toman = fetch_and_parse_usd()
        if not is_plausible_market_prices(tala, usd_toman, ounce):
            fair_probe = fair_gold_price(usd_toman, ounce)
            logger.error(
                "Crawler gold skipped: bad tala=%s usd=%s ounce=%s fair=%s",
                tala,
                usd_toman,
                ounce,
                fair_probe,
            )
            return False
        fair_price = fair_gold_price(usd_toman, ounce)
        difference = tala - fair_price
        recent_history = get_price_history_for_analysis_crawler(TREND_HOURS)
        recent_differences = [h[1] for h in recent_history]
        differences_for_analysis = recent_differences + [difference]
        rsi, volatility, trend = calculate_rsi_and_volatility_and_trend_crawler(
            differences_for_analysis
        )
        save_price_history_crawler(
            tala, usd_toman, ounce, fair_price, difference, rsi, volatility, trend
        )
        logger.info(
            "Crawler gold saved tala=%s usd=%s ounce=%s diff=%s rsi=%s trend=%s",
            tala,
            usd_toman,
            ounce,
            f"{difference:.2f}",
            rsi,
            trend,
        )
        return True
    except Exception as e:
        logger.error("Crawler gold failed: %s", e)
        return False


def crawl_metals_prices() -> bool:
    """@Zarpay724 → metals_price_history."""
    session = requests.Session()
    try:
        metals = fetch_metals_prices(
            session=session,
            max_attempts=MAX_FETCH_ATTEMPTS,
            timeout=REQUEST_TIMEOUT,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            persist=True,
            persist_source="crawler",
        )
        if metals.stale:
            logger.warning("Crawler metals used DB fallback")
        else:
            logger.info(
                "Crawler metals saved gold=%s silver=%s copper=%s",
                metals.gold_per_gram,
                metals.silver_per_gram,
                metals.copper_per_gram,
            )
        return True
    except Exception as e:
        logger.error("Crawler metals failed: %s", e)
        return False


def crawl_crypto_prices() -> bool:
    """arz_247 + fallbacks → crypto_price_history (BTC, ETH, TRX, USDT)."""
    try:
        usd_toman = usd_toman_for_dependent_fetches()
        prices = fetch_crypto_prices(usd_toman=usd_toman)
        if not prices:
            logger.warning("Crawler crypto: empty price map")
            return False
        n = save_crypto_price_history(prices)
        logger.info(
            "Crawler crypto saved %s symbols: %s (via @%s + fallbacks)",
            n,
            list(prices.keys()),
            CRYPTO_CHANNEL_USERNAME,
        )
        return True
    except Exception as e:
        logger.error("Crawler crypto failed: %s", e)
        return False


def run_crawl_cycle():
    logger.info("Crawler cycle: gold + metals + crypto")
    crawl_gold_market()
    crawl_metals_prices()
    crawl_crypto_prices()


# ================= MAIN CRAWLER LOOP =================
def main():
    logger.info(
        "Crawler started. Gold/fair: @%s + USD @%s (fallback @%s) | "
        "Metals: @%s | Crypto: @%s + fallbacks | interval=%ss",
        GOLD_CHANNEL_USERNAME,
        USD_CHANNEL_USERNAME,
        USD_CHANNEL_FALLBACK,
        METALS_CHANNEL_USERNAME,
        CRYPTO_CHANNEL_USERNAME,
        CRAWL_INTERVAL_SEC,
    )
    while True:
        run_crawl_cycle()
        time.sleep(CRAWL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
