# crawler_service.py
import time
import sqlite3
import logging
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import numpy as np # For technical indicators

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gold_crawler")

# ================= CONFIG ==================
GOLD_CHANNEL_USERNAME = "ecogold_ir"
USD_CHANNEL_USERNAME = "tgjucurrency"
GOLD_CHANNEL_URL = f"https://t.me/s/{GOLD_CHANNEL_USERNAME}"
USD_CHANNEL_URL = f"https://t.me/s/{USD_CHANNEL_USERNAME}"
REQUEST_TIMEOUT = 10
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
    """Calculate RSI, Volatility, and Trend from a list of differences"""
    # The function should only calculate if it has enough *historical* data points.
    # The current difference is added *after* fetching history for analysis.
    # So, if len(differences) < 3, we don't have enough history to calculate trend/indicators *for the current point*.
    # Let's say we need at least 2 historical points to calculate a slope and 14 for RSI.
    min_history_for_rsi = MIN_HISTORY_FOR_RSI
    min_history_for_trend = MIN_HISTORY_FOR_TREND

    if len(differences) < min_history_for_trend + 1: # +1 because current diff is included
        logger.debug(f"Crawler: Not enough history for analysis (len={len(differences)}). Returning N/A.")
        return "N/A", "N/A", "N/A"

    # Calculate trend (simple linear regression slope) on historical data only
    # Use differences[:-1] to exclude the *current* difference when calculating the slope for *historical* trend
    historical_differences = differences[:-1]
    if len(historical_differences) < min_history_for_trend:
         # If removing current diff leaves insufficient data for trend, return N/A
         logger.debug(f"Crawler: Insufficient historical data for trend after excluding current diff.")
         return "N/A", "N/A", "N/A"

    x = np.arange(len(historical_differences))
    y = np.array(historical_differences)
    slope, _ = np.polyfit(x, y, 1)

    # Calculate RSI (Relative Strength Index) - Simplified 14-period, using historical data only
    rsi = "N/A"
    if len(historical_differences) >= min_history_for_rsi:
        deltas = np.diff(historical_differences[-min_history_for_rsi:]) # Use last 14 historical points
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0] # Make losses positive for calculation
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_gain > 0 else 0 # RSI is 100 if no losses, 0 if no gains
    else:
        logger.debug(f"Crawler: Insufficient historical data for RSI (need {min_history_for_rsi}, have {len(historical_differences)}).")

    # Calculate Volatility (std of differences over the historical period)
    volatility = np.std(historical_differences)

    # Determine trend direction based on slope
    if slope > 100: # Threshold for "strong" trend
        trend = "UPWARD"
    elif slope < -100:
        trend = "DOWNWARD"
    else:
        trend = "FLAT"

    logger.debug(f"Crawler: Calculated - RSI: {rsi}, Vol: {volatility}, Trend: {trend} based on {len(historical_differences)} historical points.")
    return round(rsi, 2) if rsi != "N/A" else "N/A", round(volatility, 2), trend


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

def parse_usd_post(text: str):
    text = normalize(text)
    momentary_price_match = re.search(r"قیمت\s+لحظه\s+ای\s*[:\s]*\s*([\d,]+)\s*ریال", text)
    if not momentary_price_match:
        return None
    usd_rial = int(momentary_price_match.group(1).replace(",", ""))
    usd_toman = usd_rial / 10
    return usd_toman

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

def fetch_and_parse_usd(max_attempts: int = 10):
    """Fetch USD data, trying multiple posts if needed"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(USD_CHANNEL_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    if not msgs:
        raise RuntimeError("No messages found")

    # Try from latest to oldest
    for i in range(min(max_attempts, len(msgs))):
        msg_text = msgs[-(i+1)].get_text("\n", strip=True)
        result = parse_usd_post(msg_text)
        if result:
            return result

    raise ValueError("USD price not found in recent posts")

# ================= MAIN CRAWLER LOOP =================
def main():
    logger.info("Crawler service started. Fetching data every 10 minutes...")
    while True:
        try:
            logger.info("Crawler: Fetching data...")
            tala, ounce = fetch_and_parse_gold()
            usd_toman = fetch_and_parse_usd()
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

        # Wait for 10 minutes before the next fetch
        time.sleep(600) # 600 seconds = 10 minutes

if __name__ == "__main__":
    main()