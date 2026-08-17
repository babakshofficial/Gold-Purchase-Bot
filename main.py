import re
import os
import pytz
import logging
import requests
import asyncio
import sqlite3
import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime, timedelta, time
from bs4 import BeautifulSoup
import telegram.error
from dotenv import load_dotenv
from telegram.helpers import escape_markdown
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans']
from io import BytesIO
import numpy as np 
from telegram.helpers import escape_markdown 
import messages as msg

load_dotenv()
# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gold_bot")

# ================= CONFIG ==================
BOT_TOKEN = os.getenv('BOT_TOKEN')
GOLD_CHANNEL_USERNAME = "ecogold_ir"
USD_CHANNEL_USERNAME = "tgjucurrency"
GOLD_CHANNEL_URL = f"https://t.me/s/{GOLD_CHANNEL_USERNAME}"
USD_CHANNEL_URL = f"https://t.me/s/{USD_CHANNEL_USERNAME}"
PRIVATE_CHANNEL_ID = os.getenv('PRIVATE_CHANNEL_ID')
ADMIN_IDS = [int(x) for x in os.getenv('ADMIN_IDS', '').split(',') if x]
REQUEST_TIMEOUT = 30
REQUEST_CONNECT_TIMEOUT = 10 
REQUEST_READ_TIMEOUT = 60 
MAX_FETCH_ATTEMPTS = 5  
RETRY_BACKOFF_FACTOR = 2 
DEFAULT_BUY_THRESHOLD = 100_000
DEFAULT_WAIT_THRESHOLD = 500_000
ASK_AMOUNT = 1
ASK_BROADCAST = 2
ASK_DB_ACTION = 3
ASK_EXPORT_DAYS = 4
ASK_THRESHOLD_TYPE = 5 
ASK_THRESHOLD_VALUE = 6 
ASK_THRESHOLD_TYPE_SIGNIFICANT_MOVE = 7
TREND_HOURS = 6 
NOTIF_BUY = 1
NOTIF_SELL = 2
NOTIF_SIGNIFICANT_MOVE = 4
NOTIF_SUMMARY = 8
NOTIF_PORTFOLIO = 16
DEFAULT_NOTIFICATION_FLAGS = NOTIF_BUY
ASK_CHART_TIMEFRAME = 7
ASK_PORTFOLIO_GOLD = 8
ASK_PORTFOLIO_TOMAN = 9
ASK_PORTFOLIO_USD = 10
STORE_PREV_MENU = 'previous_menu'
TEHRAN_TZ = pytz.timezone('Asia/Tehran')

# Navigation menu IDs
NAV_MAIN = "main_menu"
NAV_SETTINGS = "settings"
NAV_HISTORY = "history_menu"
NAV_PORTFOLIO = "portfolio"
NAV_THRESHOLDS = "set_thresholds"
NAV_ADMIN = "admin_menu"
NAV_ADMIN_CHARTS = "admin_charts"
NAV_ADMIN_DB = "admin_db"
NAV_ADMIN_EXPORT = "admin_export"
NAV_ADMIN_BROADCAST = "admin_broadcast_menu"
# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        notifications INTEGER DEFAULT 1,
        buy_threshold INTEGER DEFAULT {DEFAULT_BUY_THRESHOLD},
        wait_threshold INTEGER DEFAULT {DEFAULT_WAIT_THRESHOLD},
        significant_move_threshold INTEGER DEFAULT 700000, -- Add this new column
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tala_price INTEGER,
        usd_price REAL, -- This was the raw USD price in Toman fetched from the channel
        ounce_price REAL, -- This was the raw Ounce price in USD fetched from the channel
        fair_price REAL,
        difference REAL
    )''')
    try:
        c.execute('ALTER TABLE price_history ADD COLUMN usd_raw_toman REAL')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE price_history ADD COLUMN ounce_raw_usd REAL')
    except sqlite3.OperationalError:
        pass

    _migrate_users_columns(c)

    conn.commit()
    conn.close()

def _migrate_users_columns(c):
    """Add columns that may be missing from older databases."""
    migrations = [
        'ALTER TABLE users ADD COLUMN notification_flags INTEGER DEFAULT 1',
        'ALTER TABLE users ADD COLUMN significant_move_threshold INTEGER DEFAULT 700000',
        'ALTER TABLE users ADD COLUMN gold_grams REAL DEFAULT NULL',
        'ALTER TABLE users ADD COLUMN cash_toman INTEGER DEFAULT 0',
        'ALTER TABLE users ADD COLUMN cash_usd REAL DEFAULT 0',
        'ALTER TABLE users ADD COLUMN baseline_gold_price INTEGER DEFAULT NULL',
        'ALTER TABLE users ADD COLUMN baseline_usd_toman REAL DEFAULT NULL',
        'ALTER TABLE users ADD COLUMN baseline_total_toman INTEGER DEFAULT NULL',
        'ALTER TABLE users ADD COLUMN baseline_total_usd REAL DEFAULT NULL',
        'ALTER TABLE users ADD COLUMN portfolio_updated_at TIMESTAMP DEFAULT NULL',
    ]
    for sql in migrations:
        try:
            c.execute(sql)
        except sqlite3.OperationalError:
            pass

    price_history_migrations = [
        'ALTER TABLE price_history ADD COLUMN source TEXT DEFAULT "bot"',
        'ALTER TABLE price_history ADD COLUMN rsi REAL',
        'ALTER TABLE price_history ADD COLUMN volatility REAL',
        'ALTER TABLE price_history ADD COLUMN trend TEXT',
    ]
    for sql in price_history_migrations:
        try:
            c.execute(sql)
        except sqlite3.OperationalError:
            pass

init_db()

# ================= DATABASE HELPERS =================
def add_or_update_user(user_id, username, first_name):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
    exists = c.fetchone()
    if exists:
        c.execute('''UPDATE users SET username = ?, first_name = ? WHERE user_id = ?''',
                  (username, first_name, user_id))
    else:
        c.execute('''INSERT INTO users (user_id, username, first_name, notifications, notification_flags, buy_threshold, wait_threshold, significant_move_threshold)
                     VALUES (?, ?, ?, 1, ?, ?, ?, ?)''',
                  (user_id, username, first_name, DEFAULT_NOTIFICATION_FLAGS, DEFAULT_BUY_THRESHOLD, DEFAULT_WAIT_THRESHOLD, 700000)) 
    conn.commit()
    conn.close()

def get_user_settings(user_id):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT notifications, notification_flags, buy_threshold, wait_threshold, significant_move_threshold FROM users WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return {
            'notifications': result[0],
            'notification_flags': result[1],
            'buy_threshold': result[2],
            'wait_threshold': result[3],
            'significant_move_threshold': result[4]
        }
    return {
        'notifications': 1,
        'notification_flags': DEFAULT_NOTIFICATION_FLAGS,
        'buy_threshold': DEFAULT_BUY_THRESHOLD,
        'wait_threshold': DEFAULT_WAIT_THRESHOLD,
        'significant_move_threshold': 700000
    }

def get_user_portfolio(user_id):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT gold_grams, cash_toman, cash_usd,
                        baseline_gold_price, baseline_usd_toman,
                        baseline_total_toman, baseline_total_usd, portfolio_updated_at
                 FROM users WHERE user_id = ?''', (user_id,))
    result = c.fetchone()
    conn.close()
    if not result or result[7] is None:
        return None
    return {
        'gold_grams': result[0] or 0.0,
        'cash_toman': result[1] or 0,
        'cash_usd': result[2] or 0.0,
        'baseline_gold_price': result[3],
        'baseline_usd_toman': result[4],
        'baseline_total_toman': result[5],
        'baseline_total_usd': result[6],
        'portfolio_updated_at': result[7],
    }

def user_has_portfolio(user_id):
    portfolio = get_user_portfolio(user_id)
    if not portfolio:
        return False
    return (
        (portfolio['gold_grams'] or 0) > 0
        or (portfolio['cash_toman'] or 0) > 0
        or (portfolio['cash_usd'] or 0) > 0
    )

def save_user_portfolio(user_id, gold_grams, cash_toman, cash_usd, tala_price, usd_toman):
    gold_value = gold_grams * tala_price
    total_toman = gold_value + cash_toman + (cash_usd * usd_toman)
    total_usd = total_toman / usd_toman if usd_toman else 0
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''UPDATE users SET
                 gold_grams = ?, cash_toman = ?, cash_usd = ?,
                 baseline_gold_price = ?, baseline_usd_toman = ?,
                 baseline_total_toman = ?, baseline_total_usd = ?,
                 portfolio_updated_at = CURRENT_TIMESTAMP
                 WHERE user_id = ?''',
              (gold_grams, cash_toman, cash_usd, tala_price, usd_toman,
               int(total_toman), total_usd, user_id))
    conn.commit()
    conn.close()
    return int(total_toman), total_usd

def get_users_with_portfolio_notifications():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT user_id, gold_grams, cash_toman, cash_usd,
                        baseline_total_toman, baseline_total_usd, notification_flags
                 FROM users
                 WHERE notifications = 1
                 AND portfolio_updated_at IS NOT NULL
                 AND (gold_grams > 0 OR cash_toman > 0 OR cash_usd > 0)''')
    results = c.fetchall()
    conn.close()
    return results

def fetch_current_prices():
    """Fetch live prices or fall back to latest DB record."""
    try:
        tala, ounce = fetch_and_parse_gold()
        usd_toman = fetch_and_parse_usd()
        return tala, ounce, usd_toman, False
    except Exception:
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT tala_price, ounce_price, usd_price FROM price_history ORDER BY timestamp DESC LIMIT 1')
        latest = c.fetchone()
        conn.close()
        if latest:
            return latest[0], latest[1], latest[2], True
        raise RuntimeError("No price data available")

def calculate_portfolio_values(portfolio, tala_price, usd_toman):
    gold_grams = portfolio['gold_grams'] or 0
    cash_toman = portfolio['cash_toman'] or 0
    cash_usd = portfolio['cash_usd'] or 0
    gold_value = gold_grams * tala_price
    total_toman = gold_value + cash_toman + (cash_usd * usd_toman)
    total_usd = total_toman / usd_toman if usd_toman else 0
    baseline_toman = portfolio['baseline_total_toman'] or 0
    baseline_usd = portfolio['baseline_total_usd'] or 0
    pnl_toman = total_toman - baseline_toman
    pnl_usd = total_usd - baseline_usd
    pnl_pct = (pnl_toman / baseline_toman * 100) if baseline_toman else 0
    return {
        'total_toman': total_toman,
        'total_usd': total_usd,
        'pnl_toman': pnl_toman,
        'pnl_usd': pnl_usd,
        'pnl_pct': pnl_pct,
    }

def update_user_settings(user_id, notifications=None, notification_flags=None, buy_threshold=None, wait_threshold=None, significant_move_threshold=None): 
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    if notifications is not None:
        c.execute('UPDATE users SET notifications = ? WHERE user_id = ?', (notifications, user_id))
    if notification_flags is not None:
        c.execute('UPDATE users SET notification_flags = ? WHERE user_id = ?', (notification_flags, user_id))
    if buy_threshold is not None:
        c.execute('UPDATE users SET buy_threshold = ? WHERE user_id = ?', (buy_threshold, user_id))
    if wait_threshold is not None:
        c.execute('UPDATE users SET wait_threshold = ? WHERE user_id = ?', (wait_threshold, user_id))
    if significant_move_threshold is not None: 
        c.execute('UPDATE users SET significant_move_threshold = ? WHERE user_id = ?', (significant_move_threshold, user_id))
    conn.commit()
    conn.close()

def save_price_history(tala, usd_raw_toman, ounce_raw_usd, fair, diff):
    """Save price data including raw USD and Ounce prices"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''INSERT INTO price_history (tala_price, usd_price, ounce_price, fair_price, difference)
                 VALUES (?, ?, ?, ?, ?)''', (tala, usd_raw_toman, ounce_raw_usd, fair, diff))
    conn.commit()
    conn.close()

def get_price_history(limit=24):
    """Fetch price history - Updated to include raw USD and Ounce"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, tala_price, usd_price, ounce_price, fair_price, difference
                 FROM price_history ORDER BY timestamp DESC LIMIT ?''', (limit,))
    results = c.fetchall()
    conn.close()
    return results[::-1]

def get_price_history_for_analysis_bot(hours=TREND_HOURS):
    """Get price history for the last N hours from the database (for bot analysis) - prioritizes 'crawler' data"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    sql_query = '''SELECT rsi, volatility, trend, timestamp
                   FROM price_history
                   WHERE timestamp >= datetime('now', '-{} hours')
                   AND source = 'crawler'
                   ORDER BY timestamp DESC LIMIT 1'''.format(hours)
    logger.debug(f"Bot analysis query: {sql_query}")
    try:
        c.execute(sql_query)
        latest_crawler_analysis = c.fetchone()
    except sqlite3.Error as e:
        logger.error(f"Database query error in get_price_history_for_analysis_bot: {e}")
        latest_crawler_analysis = None
    conn.close()

    if latest_crawler_analysis:
        rsi, volatility, trend, timestamp = latest_crawler_analysis
        logger.info(f"Bot analysis: Using crawler data from {timestamp}")
        return {"trend": trend, "rsi": rsi, "volatility": volatility}
    else:
        logger.info("Bot analysis: No recent crawler data found, using N/A")
        return {"trend": "N/A", "rsi": "N/A", "volatility": "N/A"}

def get_price_history_by_timeframe(start_time, end_time):
    """Get price history for a specific time range from the database"""

    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"get_price_history_by_timeframe: Querying DB for range (converted) {start_str} to {end_str}")
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, tala_price, fair_price, difference
                 FROM price_history
                 WHERE timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC''', (start_str, end_str))
    results = c.fetchall()
    conn.close()
    logger.info(f"get_price_history_by_timeframe: Retrieved {len(results)} rows from DB for range {start_str} to {end_str}")
    return results

def get_all_users_with_notifications():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT user_id, notification_flags, buy_threshold, wait_threshold, significant_move_threshold FROM users WHERE notifications = 1')
    results = c.fetchall()
    conn.close()
    return results

def get_user_count():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_recent_users(days=7):
    """Get users who joined in the last N days"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT COUNT(*) FROM users
                 WHERE created_at >= datetime('now', '-' || ? || ' days')''', (days,))
    count = c.fetchone()[0]
    conn.close()
    return count

def get_active_users(days=7):
    """Get count of users who have used the bot recently (simplified - based on notifications)"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users WHERE notifications = 1')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_price_stats():
    """Get price statistics"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT tala_price, fair_price, difference, timestamp, source
                 FROM price_history ORDER BY timestamp DESC LIMIT 1''')
    latest = c.fetchone()
    c.execute('''SELECT AVG(tala_price), AVG(fair_price), AVG(difference)
                 FROM price_history
                 WHERE timestamp >= datetime('now', '-1 day')''')
    avg_24h = c.fetchone()
    c.execute('''SELECT MIN(tala_price), MAX(tala_price)
                 FROM price_history
                 WHERE timestamp >= datetime('now', '-1 day')''')
    minmax_24h = c.fetchone()
    conn.close()
    return {
        'latest': latest,
        'avg_24h': avg_24h,
        'minmax_24h': minmax_24h
    }

def export_users_to_csv():
    """Export users to CSV format"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT user_id, username, first_name, notifications,
                 buy_threshold, wait_threshold, created_at FROM users''')
    users = c.fetchall()
    conn.close()
    csv_content = "user_id,username,first_name,notifications,buy_threshold,wait_threshold,created_at\n"
    for user in users:
        csv_content += ",".join(str(x) if x is not None else "" for x in user) + "\n"
    return csv_content

def export_price_history_to_csv(days=7):
    """Export price history to CSV format"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, tala_price, usd_price, ounce_price, fair_price, difference, source
                 FROM price_history
                 WHERE timestamp >= datetime('now', '-' || ? || ' days')
                 ORDER BY timestamp DESC''', (days,))
    prices = c.fetchall()
    conn.close()
    csv_content = "timestamp,tala_price,usd_price,ounce_price,fair_price,difference,source\n"
    for price in prices:
        csv_content += ",".join(str(x) for x in price) + "\n"
    return csv_content

def clear_old_price_history(days=30):
    """Clear price history older than N days"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''DELETE FROM price_history
                 WHERE timestamp < datetime('now', '-' || ? || ' days')''', (days,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted

def get_db_size():
    """Get database file size in MB"""
    import os
    if os.path.exists('gold_bot.db'):
        size_bytes = os.path.getsize('gold_bot.db')
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

# ================= HELPERS =================
def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")

def escape_for_markdown_v2(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    return escape_markdown(text, version=2)

def fetch_latest_post(url: str, max_attempts: int = 10) -> str:
    """Fetch latest post with content, checking multiple posts if needed"""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    msgs = soup.select("div.tgme_widget_message_text")
    if not msgs:
        raise RuntimeError("No messages found")

    for i in range(min(max_attempts, len(msgs))):
        msg_text = msgs[-(i+1)].get_text("\n", strip=True)
        if msg_text and len(msg_text) > 20:
            return msg_text

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
    if "قیمت ارزهای آزاد" not in text:
        logger.debug(f"parse_usd_post: Skipping post, title does not contain 'قیمت ارزهای آزاد'. Content: {text[:200]}...")
        return None

    usd_line_match = re.search(r"🇺🇸\s*دلار\s*[:\s]*\s*([\d,]+)\s*ریال", text)
    if not usd_line_match:
        logger.warning(f"parse_usd_post: Could not find '🇺🇸 دلار : ... ریال' line in the expected format within post titled 'قیمت ارزهای آزاد'. Content: {text[:500]}...") # Log for debugging
        return None

    usd_rial = int(usd_line_match.group(1).replace(",", ""))
    usd_toman = usd_rial / 10
    return usd_toman

def fetch_and_parse_gold():
    """Fetch gold data from posts titled 'قیمت طلا', trying multiple posts if needed."""
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests_session_with_retries()
    url = GOLD_CHANNEL_URL

    for attempt in range(MAX_FETCH_ATTEMPTS):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_FETCH_ATTEMPTS} to fetch Gold data from {url}")
            r = session.get(url, headers=headers, timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT))
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            msgs = soup.select("div.tgme_widget_message_text")
            if not msgs:
                raise RuntimeError("No messages found in Gold channel")

            num_msgs_to_check = min(10, len(msgs))
            for i in range(num_msgs_to_check):
                msg_text = msgs[-(i+1)].get_text("\n", strip=True)
                if msg_text and len(msg_text) > 20:  
                    result = parse_gold_post(msg_text)
                    if result is not None:
                        logger.info(f"Successfully parsed Gold data (Tala: {result[0]}, Ounce: {result[1]}) from post #{i+1} (latest being #1), attempt {attempt + 1}.")
                        return result
                else:
                    logger.debug(f"fetch_and_parse_gold: Skipping empty/short message #{i+1}, attempt {attempt + 1}")

            logger.warning(f"Gold price (from a post titled 'قیمت طلا' containing 'قیمت لحظه ای' and 'اونس:') not found in the last {num_msgs_to_check} posts, attempt {attempt + 1}.")

        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timeout on attempt {attempt + 1} for Gold data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                raise requests.exceptions.ReadTimeout(f"Failed to fetch Gold data after {MAX_FETCH_ATTEMPTS} attempts due to timeout.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1} for Gold data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                 raise RuntimeError(f"Failed to fetch Gold data after {MAX_FETCH_ATTEMPTS} attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1} for Gold data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                 raise RuntimeError(f"Failed to fetch Gold data after {MAX_FETCH_ATTEMPTS} attempts due to an unexpected error: {e}")

        if attempt < MAX_FETCH_ATTEMPTS - 1:
            wait_time = RETRY_BACKOFF_FACTOR ** attempt
            logger.info(f"Waiting {wait_time} seconds before next Gold fetch attempt...")
            time.sleep(wait_time)

    raise RuntimeError(f"Gold price not found in the last {num_msgs_to_check} posts after {MAX_FETCH_ATTEMPTS} attempts.")


def fetch_and_parse_usd():
    """Fetch USD data from posts titled 'قیمت ارزهای آزاد', trying multiple posts if needed."""
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests_session_with_retries()
    url = USD_CHANNEL_URL

    for attempt in range(MAX_FETCH_ATTEMPTS):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_FETCH_ATTEMPTS} to fetch USD data from {url}")
            r = session.get(url, headers=headers, timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT))
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            msgs = soup.select("div.tgme_widget_message_text")
            if not msgs:
                raise RuntimeError("No messages found in USD channel")

            num_msgs_to_check = min(10, len(msgs))
            for i in range(num_msgs_to_check):
                msg_text = msgs[-(i+1)].get_text("\n", strip=True)
                if msg_text and len(msg_text) > 20:
                    result = parse_usd_post(msg_text)
                    if result is not None:
                        logger.info(f"Successfully parsed USD data ({result} Toman) from post #{i+1} (latest being #1), attempt {attempt + 1}.")
                        return result
                else:
                    logger.debug(f"fetch_and_parse_usd: Skipping empty/short message #{i+1}, attempt {attempt + 1}")

            logger.warning(f"USD price (from a post titled 'قیمت ارزهای آزاد' containing '🇺🇸 دلار : ... ریال') not found in the last {num_msgs_to_check} posts, attempt {attempt + 1}.")

        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timeout on attempt {attempt + 1} for USD data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                raise requests.exceptions.ReadTimeout(f"Failed to fetch USD data after {MAX_FETCH_ATTEMPTS} attempts due to timeout.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1} for USD data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                 raise RuntimeError(f"Failed to fetch USD data after {MAX_FETCH_ATTEMPTS} attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1} for USD data: {e}")
            if attempt == MAX_FETCH_ATTEMPTS - 1: # Last attempt
                 raise RuntimeError(f"Failed to fetch USD data after {MAX_FETCH_ATTEMPTS} attempts due to an unexpected error: {e}")

        if attempt < MAX_FETCH_ATTEMPTS - 1:
            wait_time = RETRY_BACKOFF_FACTOR ** attempt
            logger.info(f"Waiting {wait_time} seconds before next USD fetch attempt...")
            time.sleep(wait_time)

    raise RuntimeError(f"USD price not found in the last {num_msgs_to_check} posts after {MAX_FETCH_ATTEMPTS} attempts.")


def analyze_market(tala, usd_toman, ounce, buy_threshold, wait_threshold):
    fair_price = usd_toman * ounce / 41.5
    var = tala - fair_price
    if var < buy_threshold:
        verdict = msg.verdict_buy()
        emoji = "🟢"
        status = "BUY"
    elif var < wait_threshold:
        verdict = msg.verdict_wait()
        emoji = "🟡"
        status = "WAIT"
    else:
        verdict = msg.verdict_sell()
        emoji = "🔴"
        status = "SELL"
    return fair_price, var, verdict, emoji, status

def generate_price_chart():
    """Generate price comparison chart with English labels"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    history = get_price_history_by_timeframe(start_time.isoformat(), end_time.isoformat())
    if len(history) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    tala_prices = [h[1] for h in history] # Market price
    fair_prices = [h[2] for h in history] # Fair price

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, tala_prices, label='Market Price', marker='o', linewidth=2)
    plt.plot(timestamps, fair_prices, label='Fair Price', marker='s', linewidth=2, linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title('Gold Price Comparison (Last 24 Hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_usd_price_chart():
    """Generate USD price chart (in Toman) - English Labels"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, usd_price
                 FROM price_history ORDER BY timestamp DESC LIMIT 24''')
    results = c.fetchall()
    conn.close()

    if len(results) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    usd_prices_toman = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, usd_prices_toman, label='USD Price (Toman)', marker='o', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title('USD Price in Toman')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_ounce_price_chart():
    """Generate Ounce price chart (in USD) - English Labels"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, ounce_price
                 FROM price_history ORDER BY timestamp DESC LIMIT 24''')
    results = c.fetchall()
    conn.close()

    if len(results) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    ounce_prices_usd = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ounce_prices_usd, label='Ounce Price (USD)', marker='s', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('Gold Ounce Price in USD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_price_chart_by_timeframe(start_time, end_time):
    """Generate price comparison chart for a specific time range."""
    logger.info(f"generate_price_chart_by_timeframe: Querying DB for range {start_time} to {end_time}")
    history = get_price_history_by_timeframe(start_time, end_time)
    logger.info(f"generate_price_chart_by_timeframe: Got {len(history)} data points from DB.")
    if len(history) < 2:
        logger.warning(f"generate_price_chart_by_timeframe: Insufficient data ({len(history)} points) for range {start_time} to {end_time}")
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    tala_prices = [h[1] for h in history]
    fair_prices = [h[2] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, tala_prices, label='Market Price', marker='o', linewidth=2)
    plt.plot(timestamps, fair_prices, label='Fair Price', marker='s', linewidth=2, linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title(f'Gold Price Comparison ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_usd_price_chart_by_timeframe(start_time, end_time):
    """Generate USD price chart for a specific time range."""
    logger.info(f"generate_usd_price_chart_by_timeframe: Querying DB for range {start_time} to {end_time}")
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, usd_price
                 FROM price_history
                 WHERE timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC''', (start_time, end_time))
    results = c.fetchall()
    conn.close()
    logger.info(f"generate_usd_price_chart_by_timeframe: Got {len(results)} data points from DB query.")
    if len(results) < 2:
        logger.warning(f"generate_usd_price_chart_by_timeframe: Insufficient data ({len(results)} points) for range {start_time} to {end_time}")
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    usd_prices_toman = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, usd_prices_toman, label='USD Price (Toman)', marker='o', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title(f'USD Price in Toman ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_ounce_price_chart_by_timeframe(start_time, end_time):
    """Generate Ounce price chart for a specific time range."""
    logger.info(f"generate_ounce_price_chart_by_timeframe: Querying DB for range {start_time} to {end_time}")
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, ounce_price
                 FROM price_history
                 WHERE timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC''', (start_time, end_time))
    results = c.fetchall()
    conn.close()
    logger.info(f"generate_ounce_price_chart_by_timeframe: Got {len(results)} data points from DB query.")
    if len(results) < 2:
        logger.warning(f"generate_ounce_price_chart_by_timeframe: Insufficient data ({len(results)} points) for range {start_time} to {end_time}")
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    ounce_prices_usd = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ounce_prices_usd, label='Ounce Price (USD)', marker='s', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title(f'Gold Ounce Price in USD ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def requests_session_with_retries():
    """Creates a requests session with retry strategy."""
    session = requests.Session()

    retry_strategy = Retry(
        total=MAX_FETCH_ATTEMPTS - 1,  
        status_forcelist=[429, 500, 502, 503, 504],  
        allowed_methods=["HEAD", "GET", "OPTIONS"], 
        backoff_factor=RETRY_BACKOFF_FACTOR, 
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

async def send_daily_summary(context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.info("Starting daily summary process.")
        now = datetime.now()
        start_of_period = datetime.combine(now.date() - timedelta(days=1), datetime.min.time())
        end_of_period = datetime.combine(now.date() - timedelta(days=1), datetime.max.time())
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('''
            SELECT tala_price, timestamp
            FROM price_history
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        ''', (start_of_period.isoformat(), end_of_period.isoformat()))
        period_data = c.fetchall()
        conn.close()

        if not period_data:
            logger.info(f"No price data found for summary period {start_of_period} to {end_of_period}. Skipping summary.")
            return

        prices = [row[0] for row in period_data]
        timestamps = [datetime.fromisoformat(row[1]) for row in period_data]

        if not prices:
            logger.info(f"No price values found for summary period {start_of_period} to {end_of_period}. Skipping summary.")
            return

        open_price = prices[0]
        close_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)
        avg_price = sum(prices) / len(prices)
        total_change = close_price - open_price
        change_percentage = (total_change / open_price) * 100 if open_price != 0 else 0
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('''
            SELECT fair_price, timestamp
            FROM price_history
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (start_of_period.isoformat(), end_of_period.isoformat()))
        last_fair_result = c.fetchone()
        conn.close()

        last_fair_price = None
        if last_fair_result:
             last_fair_price = last_fair_result[0]
        else:
            conn = sqlite3.connect('gold_bot.db')
            c = conn.cursor()
            c.execute('''
                SELECT fair_price, timestamp
                FROM price_history
                WHERE timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (start_of_period.isoformat(),))
            last_fair_before_period = c.fetchone()
            conn.close()
            if last_fair_before_period:
                last_fair_price = last_fair_before_period[0]

        bubble_percentage = 0.0
        if last_fair_price and close_price > 0:
            bubble_percentage = ((close_price - last_fair_price) / last_fair_price) * 100

        summary_message = msg.daily_market_summary(
            start_of_period.strftime('%Y-%m-%d'),
            open_price, close_price, high_price, low_price,
            int(avg_price), total_change, change_percentage, bubble_percentage,
        )

        all_users = get_all_users_with_notifications()
        users_to_notify = [u for u in all_users if u[1] & NOTIF_SUMMARY]

        success_count = 0
        failed_count = 0
        for user_tuple in users_to_notify:
            user_id = user_tuple[0]
            try:
                await context.bot.send_message(chat_id=user_id, text=summary_message, parse_mode="Markdown")
                success_count += 1
                logger.debug(f"Daily summary sent to user {user_id}")
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.warning(f"Daily summary send failed for user {user_id}: {e}")
                failed_count += 1

        logger.info(f"Daily summary process finished. Sent to {success_count} users, failed for {failed_count} users.")

    except Exception as e:
        logger.exception("Daily summary process failed.")

def generate_user_growth_chart(days=30):
    """Generate user growth chart with English labels"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT DATE(created_at) as date, COUNT(*) as count
                 FROM users
                 WHERE created_at >= datetime('now', '-' || ? || ' days')
                 GROUP BY DATE(created_at)
                 ORDER BY date''', (days,))
    data = c.fetchall()
    conn.close()

    if len(data) < 2:
        return None

    dates = [datetime.strptime(d[0], '%Y-%m-%d') for d in data]
    counts = [d[1] for d in data]

    cumulative = []
    total = 0
    for count in counts:
        total += count
        cumulative.append(total)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, cumulative, marker='o', linewidth=2, color='#2196F3')
    plt.fill_between(dates, cumulative, alpha=0.3, color='#2196F3')

    plt.xlabel('Date')
    plt.ylabel('Number of Users')
    plt.title(f'User Growth ({days} Days Ago)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_price_difference_chart(days=7):
    """Generate price difference trend chart with English labels, fetching data from DB"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    history = get_price_history_by_timeframe(start_time.isoformat(), end_time.isoformat())
    if len(history) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    differences = [h[3] for h in history]

    colors = []
    for diff in differences:
        if diff < DEFAULT_BUY_THRESHOLD:
            colors.append('#4CAF50')
        elif diff < DEFAULT_WAIT_THRESHOLD:
            colors.append('#FFC107')
        else:
            colors.append('#F44336')

    plt.figure(figsize=(12, 6))
    plt.scatter(timestamps, differences, c=colors, s=50, alpha=0.6)
    plt.plot(timestamps, differences, linewidth=1, alpha=0.5, color='gray')

    plt.axhline(y=DEFAULT_BUY_THRESHOLD, color='green', linestyle='--', label='Buy Threshold', alpha=0.7)
    plt.axhline(y=DEFAULT_WAIT_THRESHOLD, color='red', linestyle='--', label='Sell Threshold', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Price Difference (Toman)')
    plt.title(f'Price Difference Trend ({days} Days Ago)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_detailed_history_chart(start_time, end_time):
    """Generate a chart for a specific time period with English labels, fetching data from DB"""
    if isinstance(start_time, str):
        start_time_dt = datetime.fromisoformat(start_time)
    else:
        start_time_dt = start_time
    if isinstance(end_time, str):
        end_time_dt = datetime.fromisoformat(end_time)
    else:
        end_time_dt = end_time

    history = get_price_history_by_timeframe(start_time_dt.isoformat(), end_time_dt.isoformat())
    if len(history) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    tala_prices = [h[1] for h in history]
    fair_prices = [h[2] for h in history]
    differences = [h[3] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(timestamps, tala_prices, label='Market Price', marker='o', linewidth=2)
    ax1.plot(timestamps, fair_prices, label='Fair Price', marker='s', linewidth=2, linestyle='--')
    ax1.set_ylabel('Price (Toman)')
    ax1.set_title('Price History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = []
    for diff in differences:
        if diff < DEFAULT_BUY_THRESHOLD:
            colors.append('#4CAF50')
        elif diff < DEFAULT_WAIT_THRESHOLD:
            colors.append('#FFC107')
        else:
            colors.append('#F44336')

    ax2.scatter(timestamps, differences, c=colors, s=50, alpha=0.6)
    ax2.plot(timestamps, differences, linewidth=1, alpha=0.5, color='gray')
    ax2.axhline(y=DEFAULT_BUY_THRESHOLD, color='green', linestyle='--', label='Buy Threshold', alpha=0.7)
    ax2.axhline(y=DEFAULT_WAIT_THRESHOLD, color='red', linestyle='--', label='Sell Threshold', alpha=0.7)
    ax2.set_ylabel('Price Difference (Toman)')
    ax2.set_xlabel('Time')
    ax2.set_title('Price Difference History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# ================= AUDIT LOGGING =================
async def audit_log(context: ContextTypes.DEFAULT_TYPE, user_id, username, command, response_summary):
    """Audit logging with command and response summary"""
    if not PRIVATE_CHANNEL_ID:
        logger.warning("PRIVATE_CHANNEL_ID not set - skipping audit log")
        return

    logger.debug(f"Audit Log Raw Username: '{username}', Raw Command: '{command}', Raw Response Summary: '{response_summary}'")

    username_display = escape_for_markdown_v2(username if username else "No username")

    max_msg_length = 3000
    if len(command) > max_msg_length:
        command = command[:max_msg_length] + "... (truncated)"
    if len(response_summary) > max_msg_length:
        response_summary = response_summary[:max_msg_length] + "... (truncated)"

    escaped_command = escape_for_markdown_v2(command)
    escaped_response_summary = escape_for_markdown_v2(response_summary)

    logger.debug(f"Audit Log Escaped Username: '{username_display}', Escaped Command: '{escaped_command}', Escaped Response Summary: '{escaped_response_summary}'")

    msg_part1 = (
        f"📨 **Interaction Log**\n"
        f"👤 User: {username_display} (`{user_id}`)\n"
        f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    msg_part2 = f"📥 **Command/Action:** `{escaped_command}`\n"
    msg_part3 = f"📤 **Response Summary:** {escaped_response_summary[:1000]}"

    msg = msg_part1 + msg_part2 + msg_part3

    try:
        await context.bot.send_message(
            chat_id=PRIVATE_CHANNEL_ID,
            text=msg,
            parse_mode="MarkdownV2")
        logger.info(f"Audit log sent for user {user_id}")
    except Exception as e:
        logger.error(f"Audit send failed for user {user_id}: {e}")
        try:
            simple_msg_part1 = (
                f"📨 Interaction Log\n"
                f"User: {username_display} ({user_id})\n" 
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            simple_msg_part2 = f"Command/Action: {command[:500]}\n" 
            simple_msg_part3 = f"Response Summary: {response_summary[:500]}" 
            simple_msg = simple_msg_part1 + simple_msg_part2 + simple_msg_part3

            await context.bot.send_message(
                chat_id=PRIVATE_CHANNEL_ID,
                text=simple_msg
            )
            logger.info(f"Audit log sent (fallback) for user {user_id}")
        except Exception as e2:
            logger.error(f"Audit fallback also failed for user {user_id}: {e2}")

# ================= NAVIGATION =================

def nav_back(target: str) -> str:
    return f"nav_back:{target}"


def back_row(target: str):
    return [InlineKeyboardButton(msg.BTN_BACK, callback_data=nav_back(target))]


def kb_back(target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([back_row(target)])


async def delete_message_safe(message):
    try:
        await message.delete()
    except Exception as e:
        logger.warning(f"Could not delete message: {e}")


def clear_nav_state(context):
    context.user_data.pop('waiting_for_calc', None)
    context.user_data.pop('setting_threshold', None)
    context.user_data.pop('portfolio_gold', None)
    context.user_data.pop('portfolio_toman', None)
    context.user_data.pop(STORE_PREV_MENU, None)
    context.user_data.pop('requested_chart_type', None)


async def send_portfolio_view(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user):
    add_or_update_user(user.id, user.username, user.first_name)
    portfolio = get_user_portfolio(user.id)
    if not portfolio or not user_has_portfolio(user.id):
        await context.bot.send_message(
            chat_id, msg.PORTFOLIO_NOT_SET, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
        return
    try:
        tala_price, _, usd_toman, stale = fetch_current_prices()
    except Exception:
        await context.bot.send_message(
            chat_id, msg.ERROR_FETCH, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
        return
    values = calculate_portfolio_values(portfolio, tala_price, usd_toman)
    stale_note = msg.STALE_DATA_NOTE if stale else ""
    text = msg.portfolio_view(
        gold_grams=portfolio['gold_grams'],
        cash_toman=portfolio['cash_toman'],
        cash_usd=portfolio['cash_usd'],
        total_toman=values['total_toman'],
        total_usd=values['total_usd'],
        pnl_toman=values['pnl_toman'],
        pnl_usd=values['pnl_usd'],
        pnl_pct=values['pnl_pct'],
        tala_price=tala_price,
        usd_toman=usd_toman,
        updated_at=portfolio['portfolio_updated_at'],
        stale_note=stale_note,
    )
    await context.bot.send_message(
        chat_id, text, parse_mode="Markdown", reply_markup=portfolio_keyboard()
    )


def main_menu_text(user_id: int) -> str:
    settings = get_user_settings(user_id)
    return msg.welcome_message(settings['buy_threshold'], settings['wait_threshold'])


async def show_menu(context: ContextTypes.DEFAULT_TYPE, chat_id: int, menu_id: str, user):
    if menu_id == NAV_MAIN:
        await context.bot.send_message(
            chat_id,
            main_menu_text(user.id),
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )
    elif menu_id == NAV_SETTINGS:
        settings = get_user_settings(user.id)
        await context.bot.send_message(
            chat_id,
            msg.settings_message(settings),
            parse_mode="Markdown",
            reply_markup=settings_menu_keyboard(settings['notifications'], settings['notification_flags']),
        )
    elif menu_id == NAV_HISTORY:
        await context.bot.send_message(
            chat_id, msg.HISTORY_MENU, parse_mode="Markdown", reply_markup=history_menu_keyboard()
        )
    elif menu_id == NAV_THRESHOLDS:
        await context.bot.send_message(
            chat_id, msg.THRESHOLDS_MENU, parse_mode="Markdown", reply_markup=thresholds_menu_keyboard()
        )
    elif menu_id == NAV_PORTFOLIO:
        await send_portfolio_view(context, chat_id, user)
    elif menu_id == NAV_ADMIN:
        if is_admin(user.id):
            await context.bot.send_message(
                chat_id, msg.ADMIN_PANEL, parse_mode="Markdown", reply_markup=admin_keyboard()
            )
    elif menu_id == NAV_ADMIN_CHARTS:
        await context.bot.send_message(
            chat_id,
            "📈 **نمودارهای تحلیلی**\nنمودار مورد نظر را انتخاب کنید:",
            parse_mode="Markdown",
            reply_markup=admin_charts_keyboard(),
        )
    elif menu_id == NAV_ADMIN_DB:
        db_size = get_db_size()
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM price_history')
        total_records = c.fetchone()[0]
        c.execute('''SELECT COUNT(*) FROM price_history WHERE timestamp < datetime('now', '-30 days')''')
        old_records = c.fetchone()[0]
        conn.close()
        response = (
            "💾 **مدیریت دیتابیس**\n"
            f"📊 حجم فایل: {db_size:.2f} MB\n"
            f"📈 کل رکوردها: {total_records}\n"
            f"🗑 رکوردهای قدیمی‌تر از 30 روز: {old_records}\n"
            "عملیات مورد نظر را انتخاب کنید:"
        )
        await context.bot.send_message(
            chat_id, response, parse_mode="Markdown", reply_markup=admin_db_keyboard()
        )
    elif menu_id == NAV_ADMIN_EXPORT:
        await context.bot.send_message(
            chat_id,
            "📤 **خروجی داده‌ها**\nنوع خروجی را انتخاب کنید:",
            parse_mode="Markdown",
            reply_markup=admin_export_keyboard(),
        )
    elif menu_id == NAV_ADMIN_BROADCAST:
        await context.bot.send_message(
            chat_id,
            msg.ADMIN_BROADCAST_MENU,
            parse_mode="Markdown",
            reply_markup=admin_broadcast_menu_keyboard(),
        )
    elif menu_id == "calc":
        await context.bot.send_message(
            chat_id, msg.CALC_PROMPT, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
    elif menu_id == "about_us":
        await context.bot.send_message(
            chat_id,
            msg.about_message(USD_CHANNEL_USERNAME, GOLD_CHANNEL_USERNAME),
            parse_mode="Markdown",
            reply_markup=kb_back(NAV_MAIN),
        )
    elif menu_id == "help":
        await context.bot.send_message(
            chat_id, msg.help_message(), parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )


async def navigate_back(update: Update, context: ContextTypes.DEFAULT_TYPE, target: str | None = None):
    query = update.callback_query
    if not query:
        return ConversationHandler.END
    await query.answer()
    if target is None:
        target = query.data.split(":", 1)[1]
    chat_id = query.message.chat_id
    user = query.from_user
    await delete_message_safe(query.message)
    clear_nav_state(context)
    await show_menu(context, chat_id, target, user)
    return ConversationHandler.END


async def handle_nav_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await navigate_back(update, context)


async def open_menu_from_query(query, context: ContextTypes.DEFAULT_TYPE, menu_id: str, toast: str | None = None):
    """Delete current message and open a menu from a callback query."""
    if toast:
        await query.answer(toast)
    else:
        await query.answer()
    chat_id = query.message.chat_id
    user = query.from_user
    await delete_message_safe(query.message)
    await show_menu(context, chat_id, menu_id, user)


async def open_menu_from_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, menu_id: str):
    """Delete current message and open a menu."""
    await open_menu_from_query(update.callback_query, context, menu_id)

# ================= INLINE KEYBOARDS =================
def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton(msg.BTN_ANALYSIS, callback_data="gold")],
        [InlineKeyboardButton(msg.BTN_CALC, callback_data="calc")],
        [InlineKeyboardButton(msg.BTN_PORTFOLIO, callback_data="portfolio")],
        [InlineKeyboardButton(msg.BTN_CHART_GOLD, callback_data="chart")],
        [InlineKeyboardButton(msg.BTN_CHART_USD, callback_data="usd_chart")],
        [InlineKeyboardButton(msg.BTN_CHART_OUNCE, callback_data="ounce_chart")],
        [InlineKeyboardButton(msg.BTN_HISTORY, callback_data="history_menu"),
         InlineKeyboardButton(msg.BTN_SETTINGS, callback_data="settings")],
        [InlineKeyboardButton(msg.BTN_ABOUT, callback_data="about_us")],
        [InlineKeyboardButton(msg.BTN_HELP, callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def portfolio_keyboard():
    keyboard = [
        [InlineKeyboardButton(msg.BTN_UPDATE_PORTFOLIO, callback_data="portfolio_update")],
        back_row(NAV_MAIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def portfolio_setup_keyboard(back_target=NAV_PORTFOLIO):
    return InlineKeyboardMarkup([back_row(back_target)])

def chart_timeframe_keyboard():
    """Keyboard for selecting chart timeframe."""
    keyboard = [
        [InlineKeyboardButton("7h", callback_data="tf_7h"),
         InlineKeyboardButton("24h", callback_data="tf_24h")],
        [InlineKeyboardButton("7d", callback_data="tf_7d"),
         InlineKeyboardButton("30d", callback_data="tf_30d")],
        [InlineKeyboardButton("6m", callback_data="tf_6m")],
        back_row(NAV_MAIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def settings_menu_keyboard(notifications_on, notification_flags):
    notif_text = msg.settings_notif_toggle_on() if notifications_on else msg.settings_notif_toggle_off()
    buy_on = "🟢" if notification_flags & NOTIF_BUY else "⚪️"
    sell_on = "🔴" if notification_flags & NOTIF_SELL else "⚪️"
    move_on = "📊" if notification_flags & NOTIF_SIGNIFICANT_MOVE else "⚪️"
    summ_on = "📅" if notification_flags & NOTIF_SUMMARY else "⚪️"
    port_on = "💼" if notification_flags & NOTIF_PORTFOLIO else "⚪️"

    keyboard = [
        [InlineKeyboardButton(notif_text, callback_data="toggle_notif")],
        [InlineKeyboardButton(f"{buy_on} اعلان خرید", callback_data="toggle_notif_buy")],
        [InlineKeyboardButton(f"{sell_on} اعلان فروش", callback_data="toggle_notif_sell")],
        [InlineKeyboardButton(f"{move_on} حرکت قیمت", callback_data="toggle_notif_move")],
        [InlineKeyboardButton(f"{summ_on} خلاصه روزانه بازار", callback_data="toggle_notif_summary")],
        [InlineKeyboardButton(f"{port_on} گزارش روزانه دارایی", callback_data="toggle_notif_portfolio")],
        [InlineKeyboardButton(msg.BTN_SET_THRESHOLDS, callback_data="set_thresholds")],
        back_row(NAV_MAIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def history_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton(msg.BTN_HISTORY_24H, callback_data="history_24h")],
        [InlineKeyboardButton(msg.BTN_HISTORY_7D, callback_data="history_7d")],
        [InlineKeyboardButton(msg.BTN_HISTORY_30D, callback_data="history_30d")],
        back_row(NAV_MAIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def thresholds_menu_keyboard():
    """Keyboard for selecting which threshold to set"""
    keyboard = [
        [InlineKeyboardButton(msg.BTN_THRESHOLD_BUY, callback_data="set_buy_threshold")],
        [InlineKeyboardButton(msg.BTN_THRESHOLD_SELL, callback_data="set_wait_threshold")],
        [InlineKeyboardButton(msg.BTN_THRESHOLD_MOVE, callback_data="set_significant_move_threshold")],
        back_row(NAV_SETTINGS),
    ]
    return InlineKeyboardMarkup(keyboard)

def chart_fallback_keyboard(back_target=NAV_MAIN):
    """Keyboard with back button after viewing a chart."""
    return kb_back(back_target)

def back_to_previous_menu_keyboard(previous_menu_callback_data):
    """Keyboard with a back button to the previous menu."""
    return kb_back(previous_menu_callback_data)

# ================= COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    add_or_update_user(user.id, user.username, user.first_name)
    response = main_menu_text(user.id)
    await update.message.reply_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
    await audit_log(context, user.id, user.username, "/start", "Sent welcome message and main menu")

async def gold_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    # Determine user and send initial processing message
    if query:
        user = query.from_user
        user_msg = f"Callback: {query.data}"
        await query.answer()
        processing_msg = await context.bot.send_message(chat_id=query.message.chat_id, text=msg.PROCESSING)
        try:
            await query.delete_message()
        except Exception as e:
            logger.warning(f"Could not delete gold button message: {e}")
    else:
        user = update.effective_user
        user_msg = update.message.text
        processing_msg = await update.message.reply_text(msg.PROCESSING)

    settings = get_user_settings(user.id)
    try:
        try:
            tala, ounce = fetch_and_parse_gold()
            usd_toman = fetch_and_parse_usd()
            source_note = ""
            logger.info(f"Fetched fresh  Tala={tala}, Ounce={ounce}, USD={usd_toman}")

        except Exception as e:
            logger.warning(f"Real-time data fetch failed: {e}. Fetching from database.")
            source_note = msg.STALE_DATA_NOTE
            conn = sqlite3.connect('gold_bot.db')
            c = conn.cursor()
            c.execute('''SELECT tala_price, ounce_price, usd_price FROM price_history ORDER BY timestamp DESC LIMIT 1''')
            latest_record = c.fetchone()
            conn.close()

            if latest_record:
                tala, ounce, usd_toman = latest_record
                logger.info(f"Using database  Tala={tala}, Ounce={ounce}, USD={usd_toman}")
            else:
                logger.error("No data available in database either.")
                raise RuntimeError(msg.ERROR_NO_DATA)

        fair, var, verdict, emoji, status = analyze_market(
            tala, usd_toman, ounce,
            settings['buy_threshold'],
            settings['wait_threshold']
        )

        bubble_percentage = 0.0
        if fair > 0:
            bubble_percentage = ((var) / fair) * 100

        trend_info = get_price_history_for_analysis_bot(TREND_HOURS)
        trend_str = trend_info.get('trend', 'N/A')
        rsi_str = trend_info.get('rsi', 'N/A')
        volatility_str = trend_info.get('volatility', 'N/A')

        save_price_history(tala, usd_toman, ounce, fair, var)

        response = msg.gold_analysis_message(
            emoji=emoji,
            analysis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            tala=tala,
            fair=fair,
            var=var,
            bubble_percentage=bubble_percentage,
            usd_toman=usd_toman,
            ounce=ounce,
            trend_str=trend_str,
            rsi_str=rsi_str,
            volatility_str=volatility_str,
            verdict=verdict,
            source_note=source_note,
            trend_hours=TREND_HOURS,
        )

        await processing_msg.edit_text(response, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN))

        try:
            await audit_log(context, user.id, user.username, user_msg, f"Gold analysis: {status}, Trend: {trend_str}, Bubble: {bubble_percentage:.2f}% (Source: {'Fresh' if not source_note else 'Database'})")
        except Exception as e:
            logger.error(f"Failed to log gold_analysis for user {user.id}: {e}")

    except Exception:
        logger.exception("Gold analysis failed")
        await processing_msg.edit_text(msg.ERROR_FETCH, reply_markup=kb_back(NAV_MAIN))

def generate_price_chart_by_timeframe(start_time, end_time):
    """Generate price comparison chart for a specific time range."""
    history = get_price_history_by_timeframe(start_time, end_time)
    if len(history) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    tala_prices = [h[1] for h in history]
    fair_prices = [h[2] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, tala_prices, label='Market Price', marker='o', linewidth=2)
    plt.plot(timestamps, fair_prices, label='Fair Price', marker='s', linewidth=2, linestyle='--')

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title(f'Gold Price Comparison ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_usd_price_chart_by_timeframe(start_time, end_time):
    """Generate USD price chart for a specific time range."""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, usd_price
                 FROM price_history
                 WHERE timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC''', (start_time, end_time))
    results = c.fetchall()
    conn.close()

    if len(results) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    usd_prices_toman = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, usd_prices_toman, label='USD Price (Toman)', marker='o', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (Toman)')
    plt.title(f'USD Price in Toman ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def generate_ounce_price_chart_by_timeframe(start_time, end_time):
    """Generate Ounce price chart for a specific time range."""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, ounce_price
                 FROM price_history
                 WHERE timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC''', (start_time, end_time))
    results = c.fetchall()
    conn.close()

    if len(results) < 2:
        return None

    timestamps = [datetime.fromisoformat(h[0]) for h in results]
    ounce_prices_usd = [h[1] for h in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ounce_prices_usd, label='Ounce Price (USD)', marker='s', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title(f'Gold Ounce Price in USD ({start_time} to {end_time})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


async def show_history_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
        user_msg = f"Callback: {query.data}"
        await open_menu_from_callback(update, context, NAV_HISTORY)
    else:
        user = update.effective_user
        user_msg = "Command: /history"
        await update.message.reply_text(
            msg.HISTORY_MENU, parse_mode="Markdown", reply_markup=history_menu_keyboard()
        )

    try:
        await audit_log(context, user.id, user.username, user_msg, "History menu opened")
    except Exception as e:
        logger.error(f"Failed to log show_history_menu for user {user.id}: {e}")


async def show_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Initiates chart timeframe selection conversation."""
    query = update.callback_query
    context.user_data[STORE_PREV_MENU] = NAV_MAIN
    context.user_data['requested_chart_type'] = 'gold'
    await query.answer()
    chat_id = query.message.chat_id
    await delete_message_safe(query.message)
    await context.bot.send_message(
        chat_id, msg.CHART_SELECT_GOLD, parse_mode="Markdown", reply_markup=chart_timeframe_keyboard()
    )
    return ASK_CHART_TIMEFRAME

async def show_usd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Initiates USD chart timeframe selection conversation."""
    query = update.callback_query
    context.user_data[STORE_PREV_MENU] = NAV_MAIN
    context.user_data['requested_chart_type'] = 'usd'
    await query.answer()
    chat_id = query.message.chat_id
    await delete_message_safe(query.message)
    await context.bot.send_message(
        chat_id, msg.CHART_SELECT_USD, parse_mode="Markdown", reply_markup=chart_timeframe_keyboard()
    )
    return ASK_CHART_TIMEFRAME

async def show_ounce_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Initiates Ounce chart timeframe selection conversation."""
    query = update.callback_query
    context.user_data[STORE_PREV_MENU] = NAV_MAIN
    context.user_data['requested_chart_type'] = 'ounce'
    await query.answer()
    chat_id = query.message.chat_id
    await delete_message_safe(query.message)
    await context.bot.send_message(
        chat_id, msg.CHART_SELECT_OUNCE, parse_mode="Markdown", reply_markup=chart_timeframe_keyboard()
    )
    return ASK_CHART_TIMEFRAME



async def show_history_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
        user_msg = f"Callback: {query.data}"
        await query.answer("در حال تولید نمودار تاریخچه...")
    else:
        user = update.effective_user
        user_msg = f"Command: /history with unknown source"

    timeframe = query.data.split('_')[1] if query else None
    if not timeframe:
        error_msg = "❌ خطای زمان‌بندی"
        if query:
            await query.answer(error_msg, show_alert=True)
        else:
            await update.message.reply_text(error_msg)
        return

    try:
        now = datetime.now()
        if timeframe == '24h':
            start_time = (now - timedelta(hours=24)).isoformat()
            caption = "📈 نمودار تاریخچه قیمت (24 ساعت اخیر)"
        elif timeframe == '7d':
            start_time = (now - timedelta(days=7)).isoformat()
            caption = "📊 نمودار تاریخچه قیمت (7 روز اخیر)"
        elif timeframe == '30d':
            start_time = (now - timedelta(days=30)).isoformat()
            caption = "📈 نمودار تاریخچه قیمت (30 روز اخیر)"
        else:
            error_msg = "❌ بازه زمانی نامعتبر"
            if query:
                await query.answer(error_msg, show_alert=True)
            else:
                await update.message.reply_text(error_msg)
            return

        end_time = now.isoformat()

        chart = generate_detailed_history_chart(start_time, end_time)

        if chart is None:
            no_data_msg = f"📊 داده‌های کافی برای نمودار {timeframe} وجود ندارد. لطفاً بعداً تلاش کنید."
            if query:
                await query.edit_message_text(no_data_msg, reply_markup=kb_back(NAV_HISTORY))
            else:
                await update.message.reply_text(no_data_msg, reply_markup=kb_back(NAV_HISTORY))
            return

        if query:
            await delete_message_safe(query.message)
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart,
                caption=caption,
                reply_markup=kb_back(NAV_HISTORY),
            )
        else:
            await update.message.reply_photo(photo=chart, caption=caption)

        try:
            await audit_log(context, user.id, user.username, user_msg, f"History chart ({timeframe}) sent successfully")
        except Exception as e:
            logger.error(f"Failed to log show_history_chart for user {user.id}: {e}")

    except Exception as e:
        logger.exception("History chart generation failed")
        error_msg = f"❌ خطا در تولید نمودار {timeframe}"
        if query:
            await query.answer(error_msg, show_alert=True)
        else:
            await update.message.reply_text(error_msg)

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
        user_msg = f"Settings accessed via callback: {query.data}"
        await open_menu_from_query(query, context, NAV_SETTINGS)
    elif update and update.callback_query:
        user = update.callback_query.from_user
        user_msg = f"Settings accessed via callback: {update.callback_query.data}"
        await open_menu_from_callback(update, context, NAV_SETTINGS)
    elif update and update.message:
        user = update.effective_user
        user_msg = update.message.text
        settings = get_user_settings(user.id)
        await update.message.reply_text(
            msg.settings_message(settings),
            parse_mode="Markdown",
            reply_markup=settings_menu_keyboard(settings['notifications'], settings['notification_flags']),
        )
    else:
        logger.error("settings_menu called without update or query")
        return

    try:
        settings = get_user_settings(user.id)
        await audit_log(context, user.id, user.username, user_msg, f"Settings accessed. Notifications: {settings['notifications']}, Buy Thresh: {settings['buy_threshold']}, Sell Thresh: {settings['wait_threshold']}, Significant Move Thresh: {settings['significant_move_threshold']}")
    except Exception as e:
        logger.error(f"Failed to log settings_menu for user {user.id}: {e}")

async def toggle_notifications(query, user_id, context):
    settings = get_user_settings(user_id)
    new_value = 0 if settings['notifications'] else 1
    update_user_settings(user_id, notifications=new_value)
    await open_menu_from_query(query, context, NAV_SETTINGS, toast=msg.SETTINGS_SAVED)

async def toggle_notification_flag(query, user_id, flag, context_from_callback):
    settings = get_user_settings(user_id)
    current_flags = settings['notification_flags']
    new_flags = current_flags ^ flag
    update_user_settings(user_id, notification_flags=new_flags)
    await open_menu_from_query(query, context_from_callback, NAV_SETTINGS, toast=msg.NOTIF_FLAG_UPDATED)

async def set_thresholds_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the conversation for setting thresholds"""
    await open_menu_from_callback(update, context, NAV_THRESHOLDS)

async def set_threshold_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the callback for selecting buy/wait/significant_move threshold to set"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    if query.data == "set_buy_threshold":
        context.user_data['setting_threshold'] = 'buy'
        prompt = msg.THRESHOLD_BUY_PROMPT
    elif query.data == "set_wait_threshold":
        context.user_data['setting_threshold'] = 'wait'
        prompt = msg.THRESHOLD_SELL_PROMPT
    elif query.data == "set_significant_move_threshold":
        context.user_data['setting_threshold'] = 'significant_move'
        prompt = msg.THRESHOLD_MOVE_PROMPT
    else:
        await query.edit_message_text(msg.ERROR_INTERNAL)
        return ConversationHandler.END
    chat_id = query.message.chat_id
    await delete_message_safe(query.message)
    await context.bot.send_message(
        chat_id, prompt, parse_mode="Markdown", reply_markup=kb_back(NAV_THRESHOLDS)
    )
    return ASK_THRESHOLD_VALUE

async def set_threshold_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the message input for the threshold value"""
    user = update.effective_user
    try:
        value = int(update.message.text.replace(",", ""))
        threshold_type = context.user_data.get('setting_threshold')

        if threshold_type == 'buy':
            update_user_settings(user.id, buy_threshold=value)
            success_msg = msg.threshold_saved('buy', value)
        elif threshold_type == 'wait':
            update_user_settings(user.id, wait_threshold=value)
            success_msg = msg.threshold_saved('wait', value)
        elif threshold_type == 'significant_move':
            update_user_settings(user.id, significant_move_threshold=value)
            success_msg = msg.threshold_saved('significant_move', value)
        else:
            success_msg = msg.ERROR_INTERNAL
            logger.warning(f"User {user.id} tried to set threshold without selecting type first.")

        await update.message.reply_text(success_msg, parse_mode="Markdown", reply_markup=kb_back(NAV_SETTINGS))

        # Audit log
        try:
            await audit_log(context, user.id, user.username, f"Set threshold {threshold_type} to {value:,}", success_msg)
        except Exception as e:
            logger.error(f"Failed to log set_threshold_value for user {user.id}: {e}")

    except ValueError:
        await update.message.reply_text(msg.ERROR_INVALID_NUMBER, reply_markup=kb_back(NAV_THRESHOLDS))
        return ASK_THRESHOLD_VALUE

    except Exception:
        logger.exception("Setting threshold value failed")
        await update.message.reply_text(msg.ERROR_GENERIC, reply_markup=kb_back(NAV_THRESHOLDS))

    context.user_data.pop('setting_threshold', None)
    return ConversationHandler.END

async def about_us(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Handle the /about command and the 'About Us' button."""
    if query:
        user = query.from_user
        user_msg = f"Callback: {query.data}"
        await open_menu_from_callback(update, context, "about_us")
        await audit_log(context, user.id, user.username, user_msg, "About Us section accessed via button")
    else:
        user = update.effective_user
        user_msg = "/about"
        await update.message.reply_text(
            msg.about_message(USD_CHANNEL_USERNAME, GOLD_CHANNEL_USERNAME),
            parse_mode="Markdown",
            reply_markup=kb_back(NAV_MAIN),
        )
        await audit_log(context, user.id, user.username, user_msg, "About Us section accessed via /about command")

async def help_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        await open_menu_from_callback(update, context, "help")
    else:
        await update.message.reply_text(
            msg.help_message(), parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
async def handle_chart_timeframe_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the callback query for timeframe selection."""
    logger.info(f"handle_chart_timeframe_selection called. Query data: {update.callback_query.data}")
    query = update.callback_query
    await query.answer()

    timeframe_data = query.data
    origin_menu = context.user_data.get(STORE_PREV_MENU, NAV_MAIN)
    now_local = datetime.now() 
    local_tz = pytz.timezone('Asia/Tehran')
    now_local_aware = local_tz.localize(now_local)
    now_utc = now_local_aware.astimezone(pytz.utc)
    now_utc_truncated = now_utc.replace(microsecond=0)
    start_time_utc = None
    end_time_utc = now_utc_truncated.isoformat()
    caption_suffix = ""

    if timeframe_data == "tf_7h":
        start_time_utc = (now_utc_truncated - timedelta(hours=7)).isoformat()
        caption_suffix = "(Last 7 Hours)"
    elif timeframe_data == "tf_24h":
        start_time_utc = (now_utc_truncated - timedelta(hours=24)).isoformat()
        caption_suffix = "(Last 24 Hours)"
    elif timeframe_data == "tf_7d":
        start_time_utc = (now_utc_truncated - timedelta(days=7)).isoformat()
        caption_suffix = "(Last 7 Days)"
    elif timeframe_data == "tf_30d":
        start_time_utc = (now_utc_truncated - timedelta(days=30)).isoformat()
        caption_suffix = "(Last 30 Days)"
    elif timeframe_data == "tf_6m":
        start_time_utc = (now_utc_truncated - timedelta(days=30*6)).isoformat()
        caption_suffix = "(Last 6 Months)"
    elif timeframe_data == "cancel_chart":
        return await navigate_back(update, context, NAV_MAIN)
    else:
        await query.edit_message_text(msg.ERROR_INTERNAL, reply_markup=kb_back(origin_menu))
        context.user_data.pop(STORE_PREV_MENU, None)
        context.user_data.pop('requested_chart_type', None)
        return ConversationHandler.END

    chart_type = context.user_data.get('requested_chart_type', 'gold')

    chart = None
    caption = ""
    logger.info(f"Generating chart for type: {chart_type}, timeframe: {timeframe_data}, range (UTC): {start_time_utc} to {end_time_utc}")
    if chart_type == 'gold':
        chart = generate_price_chart_by_timeframe(start_time_utc, end_time_utc)
        caption = msg.chart_caption_gold(caption_suffix)
    elif chart_type == 'usd':
        chart = generate_usd_price_chart_by_timeframe(start_time_utc, end_time_utc)
        caption = msg.chart_caption_usd(caption_suffix)
    elif chart_type == 'ounce':
        chart = generate_ounce_price_chart_by_timeframe(start_time_utc, end_time_utc)
        caption = msg.chart_caption_ounce(caption_suffix)

    if chart:
        logger.info("Chart generated successfully, sending photo.")
        await delete_message_safe(query.message)
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=chart,
            caption=caption,
            reply_markup=kb_back(origin_menu),
        )
    else:
        logger.info("Chart generation failed (likely insufficient data), sending error message.")
        await query.edit_message_text(
            msg.CHART_INSUFFICIENT_DATA,
            reply_markup=kb_back(origin_menu),
        )

    context.user_data.pop(STORE_PREV_MENU, None)
    context.user_data.pop('requested_chart_type', None)
    logger.info("Ending chart conversation.")
    return ConversationHandler.END


async def cancel_chart_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the cancel/back button during chart selection."""
    return await navigate_back(update, context, NAV_MAIN)

# ================= CALLBACK HANDLER =================
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    logger.info(f"button_callback received query  '{query.data}' from user {query.from_user.id}. Conversation might have ended or query not handled by conversation.")

    if query.data.startswith("nav_back:"):
        await navigate_back(update, context)
        return

    if query.data.startswith("admin_") or query.data.startswith("chart_") or query.data.startswith("db_") or query.data.startswith("export_"):
        await audit_log(context, query.from_user.id, query.from_user.username, f"Callback: {query.data}", f"Admin action initiated: {query.data}")
        await admin_callback_handler(update, context)
        return

    await audit_log(context, query.from_user.id, query.from_user.username, f"Callback: {query.data}", f"Button '{query.data}' pressed")

    if query.data == "gold":
        await gold_analysis(update, context, query)
    elif query.data == "history_menu":
        await show_history_menu(update, context, query)
    elif query.data.startswith("history_"):
        await show_history_chart(update, context, query)
    elif query.data == "settings":
        await settings_menu(update, context, query)
    elif query.data == "about_us":
        await about_us(update, context, query)
    elif query.data == "help":
        await help_menu(update, context, query)
    elif query.data == "main_menu":
        await navigate_back(update, context, NAV_MAIN)
    elif query.data == "toggle_notif":
        await toggle_notifications(query, query.from_user.id, context)
    elif query.data.startswith("toggle_notif_"):
        flag_map = {
            "toggle_notif_buy": NOTIF_BUY,
            "toggle_notif_sell": NOTIF_SELL,
            "toggle_notif_move": NOTIF_SIGNIFICANT_MOVE,
            "toggle_notif_summary": NOTIF_SUMMARY,
            "toggle_notif_portfolio": NOTIF_PORTFOLIO,
        }
        flag = flag_map.get(query.data)
        if flag is not None:
            await toggle_notification_flag(query, query.from_user.id, flag, context)
        else:
            logger.warning(f"Unknown toggle flag requested: {query.data}")
    elif query.data == "set_thresholds":
        await set_thresholds_start(update, context)
    elif query.data.startswith("set_") and ("threshold" in query.data):
        await set_threshold_type(update, context)
        return ASK_THRESHOLD_VALUE
    elif query.data == "calc":
        context.user_data['waiting_for_calc'] = True
        await open_menu_from_callback(update, context, "calc")
    else:
        logger.info(f"button_callback: No specific handler found for query data '{query.data}'.")

# ================= PORTFOLIO =================

async def portfolio_show(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    user = query.from_user if query else update.effective_user
    add_or_update_user(user.id, user.username, user.first_name)
    chat_id = query.message.chat_id if query else update.message.chat_id

    if query:
        await delete_message_safe(query.message)

    portfolio = get_user_portfolio(user.id)
    if not portfolio or not user_has_portfolio(user.id):
        await context.bot.send_message(
            chat_id, msg.PORTFOLIO_NOT_SET, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
        return ConversationHandler.END

    try:
        tala_price, _, usd_toman, stale = fetch_current_prices()
    except Exception:
        await context.bot.send_message(
            chat_id, msg.ERROR_FETCH, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN)
        )
        return ConversationHandler.END

    values = calculate_portfolio_values(portfolio, tala_price, usd_toman)
    stale_note = msg.STALE_DATA_NOTE if stale else ""
    text = msg.portfolio_view(
        gold_grams=portfolio['gold_grams'],
        cash_toman=portfolio['cash_toman'],
        cash_usd=portfolio['cash_usd'],
        total_toman=values['total_toman'],
        total_usd=values['total_usd'],
        pnl_toman=values['pnl_toman'],
        pnl_usd=values['pnl_usd'],
        pnl_pct=values['pnl_pct'],
        tala_price=tala_price,
        usd_toman=usd_toman,
        updated_at=portfolio['portfolio_updated_at'],
        stale_note=stale_note,
    )
    await context.bot.send_message(
        chat_id, text, parse_mode="Markdown", reply_markup=portfolio_keyboard()
    )
    return ConversationHandler.END


async def portfolio_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Conversation entry for portfolio button or command."""
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        user = query.from_user
        add_or_update_user(user.id, user.username, user.first_name)
        if user_has_portfolio(user.id):
            return await portfolio_show(update, context, query)
        await delete_message_safe(query.message)
        await context.bot.send_message(
            query.message.chat_id,
            msg.PORTFOLIO_PROMPT_GOLD,
            parse_mode="Markdown",
            reply_markup=portfolio_setup_keyboard(NAV_MAIN),
        )
        return ASK_PORTFOLIO_GOLD

    user = update.effective_user
    add_or_update_user(user.id, user.username, user.first_name)
    if user_has_portfolio(user.id):
        return await portfolio_show(update, context)
    await update.message.reply_text(
        msg.PORTFOLIO_PROMPT_GOLD,
        parse_mode="Markdown",
        reply_markup=portfolio_setup_keyboard(NAV_MAIN),
    )
    return ASK_PORTFOLIO_GOLD


async def portfolio_cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await portfolio_entry(update, context)


async def portfolio_update_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query:
        await query.answer()
        await delete_message_safe(query.message)
        await context.bot.send_message(
            query.message.chat_id,
            msg.PORTFOLIO_PROMPT_GOLD,
            parse_mode="Markdown",
            reply_markup=portfolio_setup_keyboard(NAV_PORTFOLIO),
        )
    elif update.message:
        await update.message.reply_text(
            msg.PORTFOLIO_PROMPT_GOLD,
            parse_mode="Markdown",
            reply_markup=portfolio_setup_keyboard(NAV_PORTFOLIO),
        )
    return ASK_PORTFOLIO_GOLD


def portfolio_setup_back_target(user_id):
    return NAV_PORTFOLIO if user_has_portfolio(user_id) else NAV_MAIN


async def portfolio_gold_grams(update: Update, context: ContextTypes.DEFAULT_TYPE):
    back_target = portfolio_setup_back_target(update.effective_user.id)
    try:
        value = float(update.message.text.replace(",", "").strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(msg.ERROR_NON_NEGATIVE, reply_markup=portfolio_setup_keyboard(back_target))
        return ASK_PORTFOLIO_GOLD
    context.user_data['portfolio_gold'] = value
    await update.message.reply_text(
        msg.PORTFOLIO_PROMPT_TOMAN, parse_mode="Markdown", reply_markup=portfolio_setup_keyboard(back_target)
    )
    return ASK_PORTFOLIO_TOMAN


async def portfolio_cash_toman(update: Update, context: ContextTypes.DEFAULT_TYPE):
    back_target = portfolio_setup_back_target(update.effective_user.id)
    try:
        value = int(update.message.text.replace(",", "").strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(msg.ERROR_NON_NEGATIVE, reply_markup=portfolio_setup_keyboard(back_target))
        return ASK_PORTFOLIO_TOMAN
    context.user_data['portfolio_toman'] = value
    await update.message.reply_text(
        msg.PORTFOLIO_PROMPT_USD, parse_mode="Markdown", reply_markup=portfolio_setup_keyboard(back_target)
    )
    return ASK_PORTFOLIO_USD


async def portfolio_cash_usd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    back_target = portfolio_setup_back_target(user.id)
    try:
        value = float(update.message.text.replace(",", "").strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(msg.ERROR_NON_NEGATIVE, reply_markup=portfolio_setup_keyboard(back_target))
        return ASK_PORTFOLIO_USD

    gold_grams = context.user_data.get('portfolio_gold', 0)
    cash_toman = context.user_data.get('portfolio_toman', 0)
    cash_usd = value

    if gold_grams == 0 and cash_toman == 0 and cash_usd == 0:
        await update.message.reply_text(msg.PORTFOLIO_EMPTY_ERROR, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN))
        context.user_data.pop('portfolio_gold', None)
        context.user_data.pop('portfolio_toman', None)
        return ConversationHandler.END

    try:
        tala_price, _, usd_toman, _ = fetch_current_prices()
    except Exception:
        await update.message.reply_text(msg.ERROR_FETCH, reply_markup=kb_back(back_target))
        return ConversationHandler.END

    save_user_portfolio(user.id, gold_grams, cash_toman, cash_usd, tala_price, usd_toman)
    context.user_data.pop('portfolio_gold', None)
    context.user_data.pop('portfolio_toman', None)

    await update.message.reply_text(msg.PORTFOLIO_SAVED, parse_mode="Markdown", reply_markup=portfolio_keyboard())
    await portfolio_show(update, context)
    return ConversationHandler.END


async def send_portfolio_daily_report(context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.info("Starting portfolio daily report.")
        tala_price, _, usd_toman, _ = fetch_current_prices()
        users = get_users_with_portfolio_notifications()
        date_str = datetime.now(TEHRAN_TZ).strftime('%Y-%m-%d')
        success_count = 0
        failed_count = 0

        for row in users:
            user_id, gold_grams, cash_toman, cash_usd, baseline_toman, baseline_usd, flags = row
            if not (flags & NOTIF_PORTFOLIO):
                continue
            portfolio = {
                'gold_grams': gold_grams or 0,
                'cash_toman': cash_toman or 0,
                'cash_usd': cash_usd or 0,
                'baseline_total_toman': baseline_toman,
                'baseline_total_usd': baseline_usd,
            }
            values = calculate_portfolio_values(portfolio, tala_price, usd_toman)
            report = msg.portfolio_daily_report(
                date_str=date_str,
                gold_grams=portfolio['gold_grams'],
                cash_toman=portfolio['cash_toman'],
                cash_usd=portfolio['cash_usd'],
                total_toman=values['total_toman'],
                total_usd=values['total_usd'],
                pnl_toman=values['pnl_toman'],
                pnl_pct=values['pnl_pct'],
                tala_price=tala_price,
                usd_toman=usd_toman,
            )
            try:
                await context.bot.send_message(chat_id=user_id, text=report, parse_mode="Markdown")
                success_count += 1
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.warning(f"Portfolio daily report failed for user {user_id}: {e}")
                failed_count += 1

        logger.info(f"Portfolio daily report finished. Sent: {success_count}, failed: {failed_count}")
    except Exception:
        logger.exception("Portfolio daily report process failed.")

# ================= CALC CONVERSATION =================
async def calc_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    context.user_data['waiting_for_calc'] = True
    await update.message.reply_text(msg.CALC_PROMPT, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN))
    await audit_log(context, user.id, user.username, "/calc", "Started calc conversation")
    return ASK_AMOUNT

async def calc_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    try:
        amount_toman = int(update.message.text.replace(",", ""))
        if amount_toman <= 0:
            await update.message.reply_text(msg.ERROR_POSITIVE_NUMBER, reply_markup=kb_back(NAV_MAIN))
            return

        calc_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            current_price_per_gram, _ = fetch_and_parse_gold()
            source = "لحظه‌ای"
        except Exception:
            logger.warning("Calc: Real-time data fetch failed, using database.")
            conn = sqlite3.connect('gold_bot.db')
            c = conn.cursor()
            c.execute('SELECT tala_price FROM price_history ORDER BY timestamp DESC LIMIT 1')
            result = c.fetchone()
            conn.close()
            if result:
                current_price_per_gram = result[0]
                source = "آخرین دادهٔ ذخیره شده"
            else:
                raise RuntimeError("هیچ داده‌ای برای محاسبه موجود نیست.")

        grams = amount_toman / current_price_per_gram
        response = msg.calc_result(calc_time_str, amount_toman, current_price_per_gram, grams, source)

        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=kb_back(NAV_MAIN))
        context.user_data.pop('waiting_for_calc', None)

        # Audit log
        try:
            await audit_log(context, user.id, user.username, f"Calc: {amount_toman:,} Toman -> {grams:.4f} Grams (Source: {source})", f"Calculation successful. Source: {source}")
        except Exception as e:
            logger.error(f"Failed to log calc_amount for user {user.id}: {e}")

    except ValueError:
        await update.message.reply_text(msg.ERROR_INVALID_NUMBER, reply_markup=kb_back(NAV_MAIN))
    except Exception:
        logger.exception("Calc failed")
        await update.message.reply_text(msg.ERROR_FETCH, reply_markup=kb_back(NAV_MAIN))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages - check if waiting for calc input or threshold value, otherwise log as unhandled text"""
    user = update.effective_user
    user_text = update.message.text

    if context.user_data.get('waiting_for_calc'):
        return await calc_amount(update, context)
    elif context.user_data.get('setting_threshold'):
        return await set_threshold_value(update, context)
    else:
        await audit_log(context, user.id, user.username, f"Text Message: {user_text}", "Received text message outside of a conversation. Ignored.")

# ================= ADMIN COMMANDS =================
def is_admin(user_id):
    return user_id in ADMIN_IDS

def admin_keyboard():
    """Admin main menu keyboard"""
    keyboard = [
        [InlineKeyboardButton(msg.BTN_ADMIN_STATS, callback_data="admin_stats"),
         InlineKeyboardButton(msg.BTN_ADMIN_USERS, callback_data="admin_users")],
        [InlineKeyboardButton(msg.BTN_ADMIN_PRICES, callback_data="admin_prices"),
         InlineKeyboardButton(msg.BTN_ADMIN_CHARTS, callback_data="admin_charts")],
        [InlineKeyboardButton(msg.BTN_ADMIN_DB, callback_data="admin_db"),
         InlineKeyboardButton(msg.BTN_ADMIN_EXPORT, callback_data="admin_export")],
        [InlineKeyboardButton(msg.BTN_ADMIN_BROADCAST, callback_data="admin_broadcast_menu")],
        [InlineKeyboardButton(msg.BTN_ADMIN_HEALTH, callback_data="admin_health_check")],
        back_row(NAV_MAIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_charts_keyboard():
    """Admin charts menu keyboard"""
    keyboard = [
        [InlineKeyboardButton("📈 نمودار قیمت (24 ساعت)", callback_data="chart_price_24h")],
        [InlineKeyboardButton("📊 نمودار اختلاف (7 روز)", callback_data="chart_diff_7d")],
        [InlineKeyboardButton("👥 نمودار رشد کاربران (30 روز)", callback_data="chart_users_30d")],
        back_row(NAV_ADMIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_db_keyboard():
    """Admin database management keyboard"""
    keyboard = [
        [InlineKeyboardButton("🗑 پاک کردن تاریخچه قدیمی", callback_data="db_clean_old")],
        [InlineKeyboardButton("📊 اطلاعات دیتابیس", callback_data="db_info")],
        back_row(NAV_ADMIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_export_keyboard():
    """Admin export data keyboard"""
    keyboard = [
        [InlineKeyboardButton("👥 خروجی کاربران (CSV)", callback_data="export_users")],
        [InlineKeyboardButton("💰 خروجی قیمت‌ها 7 روز", callback_data="export_prices_7")],
        [InlineKeyboardButton("💰 خروجی قیمت‌ها 30 روز", callback_data="export_prices_30")],
        back_row(NAV_ADMIN),
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_broadcast_menu_keyboard():
    """Admin broadcast menu keyboard"""
    keyboard = [
        [InlineKeyboardButton("📢 ارسال همگانی", callback_data="admin_broadcast_general")],
        [InlineKeyboardButton("🎯 ارسال هدفمند", callback_data="admin_broadcast_targeted")],
        back_row(NAV_ADMIN),
    ]
    return InlineKeyboardMarkup(keyboard)

async def admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Show admin main menu"""
    if query:
        user = query.from_user
        user_msg = f"Callback: {query.data}" 
    else:
        user = update.effective_user
        user_msg = "Command: /admin"
    if not is_admin(user.id):
        if query:
            await query.answer(msg.ERROR_ACCESS_DENIED, show_alert=True)
        else:
            await update.message.reply_text(msg.ERROR_ACCESS_DENIED)
        return

    response = msg.ADMIN_PANEL
    if query:
        chat_id = query.message.chat_id
        user = query.from_user
        await delete_message_safe(query.message)
        await show_menu(context, chat_id, NAV_ADMIN, user)
    else:
        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    try:
        await audit_log(context, user.id, user.username, user_msg, f"Admin panel accessed. Admin: {user.id}")
    except Exception as e:
        logger.error(f"Failed to log admin_menu for user {user.id}: {e}")

async def admin_health_check(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Admin health check command"""
    if query:
        user = query.from_user
    else:
        user = update.effective_user
    if not is_admin(user.id):
        if query:
            await query.answer(msg.ERROR_ACCESS_DENIED, show_alert=True)
        else:
            await update.message.reply_text(msg.ERROR_ACCESS_DENIED)
        return

    health_status = []
    try:
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT 1')
        conn.close()
        health_status.append("✅ دیتابیس: قابل دسترسی")
    except Exception as e:
        health_status.append(f"❌ دیتابیس: خطا - {e}")

    try:
        tala, ounce = fetch_and_parse_gold(max_attempts=3)
        usd_toman = fetch_and_parse_usd(max_attempts=3)
        health_status.append(f"✅ جذب داده: موفق (USD: {usd_toman:.0f}, Gold: {tala}, Ounce: {ounce})")
    except Exception as e:
        health_status.append(f"❌ جذب داده: خطا - {e}")

    try:
        if PRIVATE_CHANNEL_ID:
            await context.bot.send_message(chat_id=PRIVATE_CHANNEL_ID, text="🧪 Health Check Ping")
            health_status.append(f"✅ کانال لاگ: قابل دسترسی ({PRIVATE_CHANNEL_ID})")
        else:
            health_status.append("❌ کانال لاگ: تنظیم نشده (PRIVATE_CHANNEL_ID)")
    except Exception as e:
        health_status.append(f"❌ کانال لاگ: خطا - {e}")

    response = "🔍 **چک سلامت ربات**\n" + "\n".join(health_status)

    if query:
        try:
            await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())
        except telegram.error.BadRequest as e:
            if "Message is not modified" in str(e):
                logger.info("Health check message was not modified, ignoring.")
                await query.answer("Health check run, no changes to display.")
            else:
                raise
    else:
        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    try:
        await audit_log(context, user.id, user.username, "Command: /health" if not query else f"Callback: {query.data}", f"Health check performed. Status: {health_status[0]}")
    except Exception as e:
        logger.error(f"Failed to log admin_health_check for user {user.id}: {e}")

async def test_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test audit logging - admin only"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return
    user = update.effective_user
    user_msg = "Command: /test_audit"
    if not PRIVATE_CHANNEL_ID:
        await update.message.reply_text(
            "❌ **خطا در تنظیمات**\n"
            "PRIVATE_CHANNEL_ID تنظیم نشده است.\n"
            "لطفاً آن را در فایل .env تنظیم کنید."
        )
        return

    test_msg = (
        "🧪 **تست ارسال لاگ**\n"
        f"👤 ادمین: {user.username} ({user.id})\n"
        f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "این یک پیام تست است."
    )
    try:
        await context.bot.send_message(
            chat_id=PRIVATE_CHANNEL_ID,
            text=test_msg,
            parse_mode="Markdown"
        )
        await update.message.reply_text(
            "✅ **تست موفق**\n"
            f"پیام با موفقیت به کانال {PRIVATE_CHANNEL_ID} ارسال شد.\n"
            "لاگ‌ها باید کار کنند."
        )
        await audit_log(context, user.id, user.username, user_msg, "Audit log test successful")
    except Exception as e:
        await update.message.reply_text(
            f"❌ **تست ناموفق**\n"
            f"خطا: `{str(e)}`\n"
            "**راهنمای رفع مشکل:**\n"
            "1. مطمئن شوید PRIVATE_CHANNEL_ID صحیح است\n"
            "2. ربات باید ادمین کانال باشد\n"
            "3. ID کانال باید با - شروع شود (مثلاً -1001234567890)\n"
            "4. برای گرفتن ID کانال، پیامی را forward کنید به @userinfobot",
            parse_mode="Markdown"
        )
        await audit_log(context, user.id, user.username, user_msg, f"Audit log test failed: {e}")

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return
    user = update.effective_user
    user_msg = "Command: /stats"
    user_count = get_user_count()
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users WHERE notifications = 1')
    notif_count = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM price_history')
    history_count = c.fetchone()[0]
    conn.close()

    response = (
        "📊 **آمار کلی ربات**\n"
        f"👥 تعداد کاربران: {user_count}\n"
        f"🔔 اعلان فعال: {notif_count}\n"
        f"📈 رکوردهای قیمت: {history_count}\n"
    )
    await update.message.reply_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    await audit_log(context, user.id, user.username, user_msg, f"Admin stats requested. Users: {user_count}, Active Notifs: {notif_count}, History: {history_count}")

async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin panel callbacks"""
    query = update.callback_query
    user = query.from_user
    user_action = f"Callback: {query.data}" 

    if not is_admin(query.from_user.id):
        await query.answer("❌ شما دسترسی ندارید", show_alert=True)
        await audit_log(context, user.id, user.username, user_action, "Unauthorized admin access attempt")
        return

    await query.answer()

    await audit_log(context, user.id, user.username, user_action, f"Admin action: {query.data}")

    if query.data == "admin_menu":
        await admin_menu(update, context, query)
    elif query.data == "admin_health_check":
        await admin_health_check(update, context, query)
    elif query.data == "admin_stats":
        user_count = get_user_count()
        recent_users = get_recent_users(7)
        active_users = get_active_users(7)
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users WHERE notifications = 1')
        notif_count = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM price_history')
        history_count = c.fetchone()[0]
        conn.close()
        db_size = get_db_size()

        response = (
            "📊 **آمار کلی ربات**\n"
            f"👥 کل کاربران: {user_count}\n"
            f"🆕 کاربران جدید (7 روز): {recent_users}\n"
            f"✅ کاربران فعال: {active_users}\n"
            f"🔔 اعلان فعال: {notif_count}\n"
            f"📈 رکوردهای قیمت: {history_count}\n"
            f"💾 حجم دیتابیس: {db_size:.2f} MB"
        )
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    elif query.data == "admin_users":
        user_count = get_user_count()
        recent_7d = get_recent_users(7)
        recent_30d = get_recent_users(30)
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users WHERE notifications = 1')
        notif_on = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM users WHERE notifications = 0')
        notif_off = c.fetchone()[0]
        conn.close()

        response = (
            "👥 **آمار کاربران**\n"
            f"📊 کل کاربران: {user_count}\n"
            f"🆕 عضو شده 7 روز اخیر: {recent_7d}\n"
            f"🆕 عضو شده 30 روز اخیر: {recent_30d}\n"
            f"🔔 اعلان فعال: {notif_on}\n"
            f"🔕 اعلان غیرفعال: {notif_off}\n"
            f"📊 نرخ فعال‌سازی: {(notif_on/user_count*100) if user_count > 0 else 0:.1f}%"
        )
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    elif query.data == "admin_prices":
        stats = get_price_stats()
        if stats['latest']:
            latest_price, latest_fair, latest_diff, latest_time, latest_source = stats['latest']
            response = (
                "💰 **آمار قیمت‌ها**\n"
                f"**آخرین قیمت (منبع: {latest_source}):**\n"
                f"🏷 بازار: {latest_price:,} تومان\n"
                f"⚖️ منصفانه: {int(latest_fair):,} تومان\n"
                f"📊 اختلاف: {int(latest_diff):,} تومان\n"
                f"⏰ زمان: {latest_time}\n"
            )
            if stats['avg_24h'][0]:
                avg_market, avg_fair, avg_diff = stats['avg_24h']
                response += (
                    f"**میانگین 24 ساعت:**\n"
                    f"🏷 بازار: {int(avg_market):,} تومان\n"
                    f"⚖️ منصفانه: {int(avg_fair):,} تومان\n"
                    f"📊 اختلاف: {int(avg_diff):,} تومان\n"
                )
            if stats['minmax_24h'][0]:
                min_price, max_price = stats['minmax_24h']
                response += (
                    f"**محدوده 24 ساعت:**\n"
                    f"⬇️ کمترین: {min_price:,} تومان\n"
                    f"⬆️ بیشترین: {max_price:,} تومان\n"
                    f"📊 نوسان: {max_price - min_price:,} تومان"
                )
        else:
            response = "💰 **آمار قیمت‌ها**\nداده‌ای موجود نیست."

        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

    elif query.data == "admin_charts":
        chat_id = query.message.chat_id
        user = query.from_user
        await delete_message_safe(query.message)
        await show_menu(context, chat_id, NAV_ADMIN_CHARTS, user)
    elif query.data == "chart_price_24h":
        await query.edit_message_text("⏳ در حال تولید نمودار...")
        chart = generate_price_chart()
        if chart:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart,
                caption="📈 نمودار مقایسه قیمت (24 ساعت اخیر)"
            )
            await query.message.reply_text("نمودار ارسال شد", reply_markup=admin_charts_keyboard())
        else:
            await query.edit_message_text(
                "❌ داده کافی برای نمودار وجود ندارد",
                reply_markup=admin_charts_keyboard()
            )
    elif query.data == "chart_diff_7d":
        await query.edit_message_text("⏳ در حال تولید نمودار...")
        chart = generate_price_difference_chart(7)
        if chart:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart,
                caption="📊 نمودار اختلاف قیمت (7 روز اخیر)"
            )
            await query.message.reply_text("نمودار ارسال شد", reply_markup=admin_charts_keyboard())
        else:
            await query.edit_message_text(
                "❌ داده کافی برای نمودار وجود ندارد",
                reply_markup=admin_charts_keyboard()
            )
    elif query.data == "chart_users_30d":
        await query.edit_message_text("⏳ در حال تولید نمودار...")
        chart = generate_user_growth_chart(30)
        if chart:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart,
                caption="👥 نمودار رشد کاربران (30 روز اخیر)"
            )
            await query.message.reply_text("نمودار ارسال شد", reply_markup=admin_charts_keyboard())
        else:
            await query.edit_message_text(
                "❌ داده کافی برای نمودار وجود ندارد",
                reply_markup=admin_charts_keyboard()
            )
    elif query.data == "admin_db":
        chat_id = query.message.chat_id
        user = query.from_user
        await delete_message_safe(query.message)
        await show_menu(context, chat_id, NAV_ADMIN_DB, user)
    elif query.data == "db_clean_old":
        deleted = clear_old_price_history(30)
        await query.answer(f"✅ {deleted} رکورد پاک شد", show_alert=True)
        await query.answer()
        await admin_callback_handler(update, context)
    elif query.data == "db_info":
        db_size = get_db_size()
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        user_count = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM price_history')
        price_count = c.fetchone()[0]
        c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM price_history')
        date_range = c.fetchone()
        conn.close()

        db_size_escaped = escape_for_markdown_v2(f"{db_size:.2f}")
        db_path_escaped = escape_for_markdown_v2("gold_bot.db")
        user_count_escaped = escape_for_markdown_v2(str(user_count))
        price_count_escaped = escape_for_markdown_v2(str(price_count))

        response = (
            "📊 **اطلاعات دیتابیس**\n"
            f"💾 حجم: {db_size_escaped} MB\n"
            f"📁 مسیر: {db_path_escaped}\n"
            f"**جداول:**\n"
            f"👥 Users: {user_count_escaped} رکورد\n"
            f"💰 Price History: {price_count_escaped} رکورد\n"
        )
        if date_range[0]:
            start_date_escaped = escape_for_markdown_v2(date_range[0]) if date_range[0] else ""
            end_date_escaped = escape_for_markdown_v2(date_range[1]) if date_range[1] else ""
            response += f"📅 بازه زمانی: {start_date_escaped} تا {end_date_escaped}"

        await query.edit_message_text(response, parse_mode="MarkdownV2", reply_markup=admin_db_keyboard())

    elif query.data == "admin_export":
        chat_id = query.message.chat_id
        user = query.from_user
        await delete_message_safe(query.message)
        await show_menu(context, chat_id, NAV_ADMIN_EXPORT, user)
    elif query.data == "export_users":
        await query.answer("در حال آماده‌سازی...")
        csv_data = export_users_to_csv()
        from io import BytesIO
        file = BytesIO(csv_data.encode('utf-8'))
        file.name = f"users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=file,
            filename=file.name,
            caption="📊 خروجی لیست کاربران"
        )
        await query.message.reply_text("✅ فایل ارسال شد", reply_markup=admin_export_keyboard())
    elif query.data.startswith("export_prices_"):
        days = int(query.data.split("_")[-1])
        await query.answer("در حال آماده‌سازی...")
        csv_data = export_price_history_to_csv(days)
        from io import BytesIO
        file = BytesIO(csv_data.encode('utf-8'))
        file.name = f"prices_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=file,
            filename=file.name,
            caption=f"💰 خروجی قیمت‌ها ({days} روز اخیر)"
        )
        await query.message.reply_text("✅ فایل ارسال شد", reply_markup=admin_export_keyboard())
    elif query.data == "admin_broadcast_menu":
        chat_id = query.message.chat_id
        user = query.from_user
        await delete_message_safe(query.message)
        await show_menu(context, chat_id, NAV_ADMIN_BROADCAST, user)
    elif query.data == "admin_broadcast_general":
        await query.edit_message_text(
            "📢 **ارسال پیام همگانی**\n"
            "برای ارسال پیام به همه کاربران، از دستور زیر استفاده کنید:\n"
            "/broadcast",
            reply_markup=InlineKeyboardMarkup([back_row(NAV_ADMIN)])
        )
    elif query.data == "admin_broadcast_targeted":
        example_target_msg = "🎯 **ارسال هدفمند**\n\n"
        example_target_msg += "این ویژگی اکنون فقط یک مثال است.\n"
        example_target_msg += "برای پیاده‌سازی کامل، باید منطق جدیدی در `admin_broadcast_send` اضافه شود.\n"
        example_target_msg += "مثلاً، ارسال به کاربرانی که `buy_threshold` آن‌ها کمتر از 80,000 تومان است.\n\n"
        example_target_msg += "برای این کار، باید یک ورودی جدید برای مقدار آستانه دریافت شود و سپس لیست کاربران مطابق با شرط فیلتر شود.\n\n"
        example_target_msg += "کد فعلی فقط `/broadcast` عمومی را پیاده می‌کند. لطفاً برای اهداف هدفمند، دستور `/broadcast` را اجرا کرده و سپس کد `admin_broadcast_send` را بر اساس نیاز تغییر دهید."

        await query.edit_message_text(
            example_target_msg,
            reply_markup=InlineKeyboardMarkup([back_row(NAV_ADMIN)])
        )

async def admin_broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return ConversationHandler.END
    user = update.effective_user
    user_msg = "Command: /broadcast"
    await update.message.reply_text("📢 پیام خود را برای ارسال به همه کاربران وارد کنید:")
    await audit_log(context, user.id, user.username, user_msg, "Started broadcast conversation")
    return ASK_BROADCAST

async def admin_broadcast_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    all_users = get_all_users_with_notifications()
    users_to_notify = [u[0] for u in all_users] 

    success = 0
    failed = 0
    for user_id in users_to_notify:
        try:
            await context.bot.send_message(chat_id=user_id, text=message)
            success += 1
            await asyncio.sleep(0.05)  
        except Exception as e:
            logger.warning(f"Broadcast failed for user {user_id}: {e}")
            failed += 1

    await update.message.reply_text(
        f"✅ پیام ارسال شد\n"
        f"موفق: {success}\n"
        f"ناموفق: {failed}"
    )
    await audit_log(context, update.effective_user.id, update.effective_user.username, "Broadcast sent", f"Message: {message[:200]}... Success: {success}, Failed: {failed}")
    return ConversationHandler.END

# ================= PRICE MONITORING =================
# Inside the monitor_prices function loop
async def monitor_prices(context: ContextTypes.DEFAULT_TYPE):
    """Background task to monitor prices and send alerts"""
    try:
        tala, ounce = fetch_and_parse_gold()
        usd_toman = fetch_and_parse_usd()

        logger.info(f"Monitor Prices - Fetched Raw Tala: {tala}, Raw USD (Toman): {usd_toman}, Raw Ounce: {ounce}")
        all_users = get_all_users_with_notifications()

        analysis_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for user_tuple in all_users:
            user_id, flags, buy_thresh, wait_thresh, sig_move_thresh = user_tuple

            fair = usd_toman * ounce / 41.5
            var = tala - fair

            logger.debug(f"Monitor Prices - User {user_id}: Calculated Fair: {fair:.2f}, Diff (Var): {var:.2f}")

            verdict, emoji, status = determine_verdict(var, buy_thresh, wait_thresh)
            if flags & NOTIF_BUY and var < 0 and abs(var) > buy_thresh:
                alert_msg = msg.alert_buy(
                    analysis_time_str, verdict, int(var), tala, int(fair)
                )
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=alert_msg,
                        parse_mode="Markdown"
                    )
                    logger.info(f"BUY Alert sent to user {user_id}")
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.warning(f"Alert send failed for user {user_id}: {e}")

            if flags & NOTIF_SELL and var > 0 and var > wait_thresh:
                alert_msg = msg.alert_sell(
                    analysis_time_str, verdict, int(var), tala, int(fair)
                )
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=alert_msg,
                        parse_mode="Markdown"
                    )
                    logger.info(f"SELL Alert sent to user {user_id}")
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.warning(f"Alert send failed for user {user_id}: {e}")

            if flags & NOTIF_SIGNIFICANT_MOVE:
                if abs(var) > sig_move_thresh and var > 0:
                    alert_msg = msg.alert_significant_move(
                        analysis_time_str, int(var), tala, int(fair)
                    )
                    try:
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=alert_msg,
                            parse_mode="Markdown"
                        )
                        logger.info(f"SIGNIFICANT MOVE Alert sent to user {user_id}")
                        await asyncio.sleep(0.05)
                    except Exception as e:
                        logger.warning(f"Alert send failed for user {user_id}: {e}")

    except Exception as e:
        logger.exception("Price monitoring failed")

def determine_verdict(var, buy_thresh, wait_thresh):
    """Determine the verdict, emoji, and status based on var and thresholds."""
    if var < buy_thresh:
        return msg.verdict_alert_buy(), "🟢", "BUY"
    elif var < wait_thresh:
        return msg.verdict_alert_wait(), "🟡", "WAIT"
    else:
        return msg.verdict_alert_sell(), "🔴", "SELL"

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Regular commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("gold", lambda u, c: gold_analysis(u, c)))
    app.add_handler(CommandHandler("history", lambda u, c: show_history_menu(u, c)))
    app.add_handler(CommandHandler("settings", lambda u, c: settings_menu(u, c)))
    app.add_handler(CommandHandler("help", lambda u, c: help_menu(u, c)))
    app.add_handler(CommandHandler("about", lambda u, c: about_us(u, c)))

    portfolio_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("portfolio", portfolio_entry),
            CallbackQueryHandler(portfolio_entry, pattern='^portfolio$'),
            CallbackQueryHandler(portfolio_update_start, pattern='^portfolio_update$'),
        ],
        states={
            ASK_PORTFOLIO_GOLD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, portfolio_gold_grams),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
            ASK_PORTFOLIO_TOMAN: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, portfolio_cash_toman),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
            ASK_PORTFOLIO_USD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, portfolio_cash_usd),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
        },
        fallbacks=[
            CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
        ],
    )
    app.add_handler(portfolio_conv_handler)

    # Admin commands
    app.add_handler(CommandHandler("admin", lambda u, c: admin_menu(u, c)))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("test_audit", test_audit))
    app.add_handler(CommandHandler("health", admin_health_check))

    threshold_conv_handler = ConversationHandler(
    entry_points=[CallbackQueryHandler(set_threshold_type, pattern='^set_(buy|wait|significant_move)_threshold$')], # Update pattern
    states={
        ASK_THRESHOLD_VALUE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, set_threshold_value),
            CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
        ],
        ASK_THRESHOLD_TYPE_SIGNIFICANT_MOVE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, set_threshold_value),
            CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
        ],
    },
    fallbacks=[CallbackQueryHandler(handle_nav_back, pattern='^nav_back:')]
    )
    app.add_handler(threshold_conv_handler)

    # Calc conversation
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("calc", calc_start)],
        states={
            ASK_AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, calc_amount),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
        },
        fallbacks=[CallbackQueryHandler(handle_nav_back, pattern='^nav_back:')],
    ))

    # Threshold setting conversation
    app.add_handler(ConversationHandler(
        entry_points=[CallbackQueryHandler(set_threshold_type, pattern='^set_(buy|wait)_threshold$')],
        states={
            ASK_THRESHOLD_VALUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_threshold_value),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
        },
        fallbacks=[CallbackQueryHandler(handle_nav_back, pattern='^nav_back:')],
    ))

    chart_conv_handler = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(show_chart, pattern='^chart$'),
            CallbackQueryHandler(show_usd_chart, pattern='^usd_chart$'),
            CallbackQueryHandler(show_ounce_chart, pattern='^ounce_chart$'),
        ],
        states={
            ASK_CHART_TIMEFRAME: [
                CallbackQueryHandler(handle_chart_timeframe_selection, pattern='^tf_'),
                CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
            ],
        },
        fallbacks=[
            CallbackQueryHandler(handle_nav_back, pattern='^nav_back:'),
        ]
    )
    logger.info("Adding chart conversation handler.")
    app.add_handler(chart_conv_handler)

    logger.info("Adding general button callback handler.")
    app.add_handler(CallbackQueryHandler(button_callback))


    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    try:
        job_queue = app.job_queue
        if job_queue:
            job_queue.run_repeating(monitor_prices, interval=1800, first=10)
            daily_time = time(hour=9, minute=0, tzinfo=TEHRAN_TZ)
            portfolio_time = time(hour=9, minute=5, tzinfo=TEHRAN_TZ)
            job_queue.run_daily(send_daily_summary, time=daily_time)
            job_queue.run_daily(send_portfolio_daily_report, time=portfolio_time)
            logger.info("Price monitoring and daily jobs enabled")
        else:
            logger.warning("JobQueue not available. Install with: pip install 'python-telegram-bot[job-queue]'")
    except Exception as e:
        logger.warning(f"JobQueue setup failed: {e}")

    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()