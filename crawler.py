# crawler_service.py
import time
import sqlite3
import logging
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import numpy as np # For technical indicators
from usd_fetch import (
    USD_CHANNEL_FALLBACK,
    USD_CHANNEL_PRIMARY,
    fetch_usd_toman,
    is_plausible_market_prices,
)
from market_indicators import calculate_market_indicators
from metals_fetch import METALS_CHANNEL_USERNAME, fetch_metals_prices

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gold_crawler")

# ================= CONFIG ==================
GOLD_CHANNEL_USERNAME = "ecogold_ir"
USD_CHANNEL_USERNAME = USD_CHANNEL_PRIMARY
GOLD_CHANNEL_URL = f"https://t.me/s/{GOLD_CHANNEL_USERNAME}"
REQUEST_TIMEOUT = 10
MAX_USD_FETCH_ATTEMPTS = 5
USD_RETRY_BACKOFF_FACTOR = 2
# Trend Analysis Config (for crawler)
TREND_HOURS = 6 # Hours to look back for trend analysis
MIN_HISTORY_FOR_RSI = 14 # Minimum historical points needed for RSI
MIN_HISTORY_FOR_TREND = 2 # Minimum historical points needed for trend

# ================= DATABASE HELPERS FOR CRAWLER =================
def save_price_history_crawler(tala, usd, ounce, fair, diff, rsi, volatility, trend):
    """Saves price data to the database with 'crawler' source"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''INSERT INTO price_history (tala_price, usd_price, ounce_price, fair_price, difference, rsi, volatility, trend, source)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'crawler')''', (tala, usd, ounce, fair, diff, rsi, volatility, trend))
    conn.commit()
    conn.close()

def get_price_history_for_analysis_crawler(hours=TREND_HOURS):
    """Get price history for the last N hours from the database (for analysis)"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    # Only use crawler data for analysis
    c.execute('''SELECT timestamp, difference
                 FROM price_history
                 WHERE timestamp >= datetime('now', '-{} hours')
                 AND source = 'crawler'
                 ORDER BY timestamp ASC'''.format(hours))
    results = c.fetchall()
    conn.close()
    return results

def calculate_rsi_and_volatility_and_trend_crawler(differences):
    """Calculate RSI, Volatility, and Trend from a list of differences."""
    metrics = calculate_market_indicators([float(d) for d in differences])
    return metrics["rsi"], metrics["volatility"], metrics["trend"]


# ================= HELPERS FOR CRAWLER (copied from main bot script) =================
def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")

def fetch_latest_post(url: str, max_attempts: int = 10) -> str:
    """Fetch latest post with content, checking multiple posts if needed"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    if not msgs:
        raise RuntimeError("No messages found")

    # Try from latest to oldest (up to max_attempts)
    for i in range(min(max_attempts, len(msgs))):
        msg_text = msgs[-(i+1)].get_text("\n", strip=True)
        if msg_text and len(msg_text) > 20:  # Ensure it's not empty or too short
            return msg_text

    # If no valid message found, return the last one anyway
    return msgs[-1].get_text("\n", strip=True)

def parse_gold_post(text: str):
    text = normalize(text)
    tala = re.search(r"طلای\s*18\s*عیار[\s\n]*:\s*([\d,]+)", text)
    ounce = re.search(r"اونس\s*طلا[\s\n]*:\s*([\d,.]+)", text)
    if not tala or not ounce:
        return None
    return (
        int(tala.group(1).replace(",", "")),
        float(ounce.group(1).replace(",", ""))
    )

def fetch_and_parse_gold(max_attempts: int = 10):
    """Fetch gold data, trying multiple posts if needed"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(GOLD_CHANNEL_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    if not msgs:
        raise RuntimeError("No messages found")

    # Try from latest to oldest
    for i in range(min(max_attempts, len(msgs))):
        msg_text = msgs[-(i+1)].get_text("\n", strip=True)
        result = parse_gold_post(msg_text)
        if result:
            return result

    raise ValueError("Gold data not found in recent posts")

def fetch_and_parse_usd(max_attempts: int = MAX_USD_FETCH_ATTEMPTS):
    """USD in Toman from @nerkhedular (معامله), fallback @tgjucurrency — same as main bot."""
    return fetch_usd_toman(
        max_attempts=max_attempts,
        timeout=REQUEST_TIMEOUT,
        backoff_factor=USD_RETRY_BACKOFF_FACTOR,
    )

def crawl_metals_prices():
    """Fetch @Zarpay724 and append to metals_price_history (independent of gold crawl)."""
    session = requests.Session()
    try:
        metals = fetch_metals_prices(
            session=session,
            max_attempts=MAX_USD_FETCH_ATTEMPTS,
            timeout=REQUEST_TIMEOUT,
            backoff_factor=USD_RETRY_BACKOFF_FACTOR,
            persist=True,
            persist_source="crawler",
        )
        logger.info(
            "Crawler: Metals saved gold=%s silver=%s copper=%s (stale=%s)",
            metals.gold_per_gram,
            metals.silver_per_gram,
            metals.copper_per_gram,
            metals.stale,
        )
        return True
    except Exception as e:
        logger.error("Crawler: metals fetch failed: %s", e)
        return False


# ================= MAIN CRAWLER LOOP =================
def main():
    logger.info(
        "Crawler service started. Gold: @%s | Metals: @%s | USD: @%s (fallback @%s). "
        "Fetching every 10 minutes...",
        GOLD_CHANNEL_USERNAME,
        METALS_CHANNEL_USERNAME,
        USD_CHANNEL_USERNAME,
        USD_CHANNEL_FALLBACK,
    )
    while True:
        try:
            logger.info("Crawler: Fetching data...")
            tala, ounce = fetch_and_parse_gold()
            usd_toman = fetch_and_parse_usd()
            if not is_plausible_market_prices(tala, usd_toman, ounce):
                fair_probe = usd_toman * ounce / 41.5
                logger.error(
                    "Crawler skipped save: bad tala=%s usd=%s ounce=%s fair=%s",
                    tala,
                    usd_toman,
                    ounce,
                    fair_probe,
                )
                time.sleep(600)
                continue
            fair_price = usd_toman * ounce / 41.5
            difference = tala - fair_price
            # Fetch recent differences from the database for analysis
            recent_history = get_price_history_for_analysis_crawler(TREND_HOURS)
            recent_differences = [h[1] for h in recent_history] # Extract differences
            logger.debug(f"Crawler: Retrieved {len(recent_differences)} historical differences from DB.")
            # Add the current difference to the list for analysis
            differences_for_analysis = recent_differences + [difference]
            logger.debug(f"Crawler: Differences list for analysis now has {len(differences_for_analysis)} points (including current).")

            # Calculate RSI, Volatility, Trend based on database data + current diff
            rsi, volatility, trend = calculate_rsi_and_volatility_and_trend_crawler(differences_for_analysis)
            logger.debug(f"Crawler: Calculated RSI/Vol/Trend: {rsi}, {volatility}, {trend}")

            # Save the new data point with calculated values
            save_price_history_crawler(tala, usd_toman, ounce, fair_price, difference, rsi, volatility, trend)
            logger.info(f"Crawler: Data saved. Tala: {tala}, USD: {usd_toman}, Ounce: {ounce}, Diff: {difference:.2f}, RSI: {rsi}, Vol: {volatility}, Trend: {trend}")

        except Exception as e:
            logger.error(f"Crawler failed: {e}")

        try:
            crawl_metals_prices()
        except Exception as e:
            logger.error("Crawler metals-only pass failed: %s", e)

        # Wait for 10 minutes before the next fetch
        time.sleep(600) # 600 seconds = 10 minutes

if __name__ == "__main__":
    main()