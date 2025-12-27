import re
import os
import logging
import requests
import asyncio
import sqlite3
from datetime import datetime
from bs4 import BeautifulSoup
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
from io import BytesIO

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
REQUEST_TIMEOUT = 10

# Default thresholds (in tomans)
DEFAULT_BUY_THRESHOLD = 100_000
DEFAULT_WAIT_THRESHOLD = 500_000

ASK_AMOUNT = 1
ASK_BROADCAST = 2
ASK_DB_ACTION = 3
ASK_EXPORT_DAYS = 4

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    
    # Users table
    c.execute(f'''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        notifications INTEGER DEFAULT 1,
        buy_threshold INTEGER DEFAULT {DEFAULT_BUY_THRESHOLD},
        wait_threshold INTEGER DEFAULT {DEFAULT_WAIT_THRESHOLD},
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Price history table
    c.execute('''CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tala_price INTEGER,
        usd_price REAL,
        ounce_price REAL,
        fair_price REAL,
        difference REAL
    )''')
    
    conn.commit()
    conn.close()

init_db()

# ================= DATABASE HELPERS =================
def add_or_update_user(user_id, username, first_name):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    # Check if user exists
    c.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
    exists = c.fetchone()
    
    if exists:
        # Update only username and first_name, preserve other settings
        c.execute('''UPDATE users SET username = ?, first_name = ? WHERE user_id = ?''',
                  (username, first_name, user_id))
    else:
        # Insert new user with defaults
        c.execute('''INSERT INTO users (user_id, username, first_name, notifications, buy_threshold, wait_threshold)
                     VALUES (?, ?, ?, 1, ?, ?)''',
                  (user_id, username, first_name, DEFAULT_BUY_THRESHOLD, DEFAULT_WAIT_THRESHOLD))
    
    conn.commit()
    conn.close()

def get_user_settings(user_id):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT notifications, buy_threshold, wait_threshold FROM users WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return {'notifications': result[0], 'buy_threshold': result[1], 'wait_threshold': result[2]}
    return {'notifications': 1, 'buy_threshold': DEFAULT_BUY_THRESHOLD, 'wait_threshold': DEFAULT_WAIT_THRESHOLD}

def update_user_settings(user_id, notifications=None, buy_threshold=None, wait_threshold=None):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    if notifications is not None:
        c.execute('UPDATE users SET notifications = ? WHERE user_id = ?', (notifications, user_id))
    if buy_threshold is not None:
        c.execute('UPDATE users SET buy_threshold = ? WHERE user_id = ?', (buy_threshold, user_id))
    if wait_threshold is not None:
        c.execute('UPDATE users SET wait_threshold = ? WHERE user_id = ?', (wait_threshold, user_id))
    conn.commit()
    conn.close()

def save_price_history(tala, usd, ounce, fair, diff):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''INSERT INTO price_history (tala_price, usd_price, ounce_price, fair_price, difference)
                 VALUES (?, ?, ?, ?, ?)''', (tala, usd, ounce, fair, diff))
    conn.commit()
    conn.close()

def get_price_history(limit=24):
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, tala_price, fair_price, difference 
                 FROM price_history ORDER BY timestamp DESC LIMIT ?''', (limit,))
    results = c.fetchall()
    conn.close()
    return results[::-1]  # Reverse to get chronological order

def get_all_users_with_notifications():
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT user_id FROM users WHERE notifications = 1')
    results = [row[0] for row in c.fetchall()]
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
    
    # Latest price
    c.execute('''SELECT tala_price, fair_price, difference, timestamp 
                 FROM price_history ORDER BY timestamp DESC LIMIT 1''')
    latest = c.fetchone()
    
    # Average prices last 24 hours
    c.execute('''SELECT AVG(tala_price), AVG(fair_price), AVG(difference)
                 FROM price_history 
                 WHERE timestamp >= datetime('now', '-1 day')''')
    avg_24h = c.fetchone()
    
    # Min/Max last 24 hours
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
    c.execute('''SELECT timestamp, tala_price, usd_price, ounce_price, fair_price, difference
                 FROM price_history 
                 WHERE timestamp >= datetime('now', '-' || ? || ' days')
                 ORDER BY timestamp DESC''', (days,))
    prices = c.fetchall()
    conn.close()
    
    csv_content = "timestamp,tala_price,usd_price,ounce_price,fair_price,difference\n"
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
    usd = re.search(r"🇺🇸\s*دلار\s*:\s*([\d,]+)\s*ریال", text)
    if not usd:
        return None
    usd_rial = int(usd.group(1).replace(",", ""))
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

def analyze_market(tala, usd_toman, ounce, buy_threshold, wait_threshold):
    fair_price = usd_toman * ounce / 41.5
    var = tala - fair_price
    
    if var < buy_threshold:
        verdict = "✅ **زمان خرید طلاست!**"
        emoji = "🟢"
        status = "BUY"
    elif var < wait_threshold:
        verdict = "⏳ **صبر کنید و بازار را رصد کنید**"
        emoji = "🟡"
        status = "WAIT"
    else:
        verdict = "💰 **زمان فروش طلاست!**"
        emoji = "🔴"
        status = "SELL"
    
    return fair_price, var, verdict, emoji, status

def generate_price_chart():
    """Generate price comparison chart"""
    history = get_price_history(limit=24)
    if len(history) < 2:
        return None
    
    timestamps = [datetime.fromisoformat(h[0]) for h in history]
    tala_prices = [h[1] for h in history]
    fair_prices = [h[2] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, tala_prices, label='قیمت بازار', marker='o', linewidth=2)
    plt.plot(timestamps, fair_prices, label='قیمت منصفانه', marker='s', linewidth=2, linestyle='--')
    
    plt.xlabel('زمان')
    plt.ylabel('قیمت (تومان)')
    plt.title('مقایسه قیمت طلا')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

def generate_user_growth_chart(days=30):
    """Generate user growth chart"""
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
    
    plt.xlabel('تاریخ')
    plt.ylabel('تعداد کاربران')
    plt.title(f'رشد کاربران ({days} روز اخیر)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

def generate_price_difference_chart(days=7):
    """Generate price difference trend chart"""
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, difference
                 FROM price_history 
                 WHERE timestamp >= datetime('now', '-' || ? || ' days')
                 ORDER BY timestamp''', (days,))
    data = c.fetchall()
    conn.close()
    
    if len(data) < 2:
        return None
    
    timestamps = [datetime.fromisoformat(d[0]) for d in data]
    differences = [d[1] for d in data]
    
    # Color code based on thresholds
    colors = []
    for diff in differences:
        if diff < DEFAULT_BUY_THRESHOLD:
            colors.append('#4CAF50')  # Green
        elif diff < DEFAULT_WAIT_THRESHOLD:
            colors.append('#FFC107')  # Yellow
        else:
            colors.append('#F44336')  # Red
    
    plt.figure(figsize=(12, 6))
    plt.scatter(timestamps, differences, c=colors, s=50, alpha=0.6)
    plt.plot(timestamps, differences, linewidth=1, alpha=0.5, color='gray')
    
    # Add threshold lines
    plt.axhline(y=DEFAULT_BUY_THRESHOLD, color='green', linestyle='--', label='آستانه خرید', alpha=0.7)
    plt.axhline(y=DEFAULT_WAIT_THRESHOLD, color='red', linestyle='--', label='آستانه فروش', alpha=0.7)
    
    plt.xlabel('زمان')
    plt.ylabel('اختلاف قیمت (تومان)')
    plt.title(f'روند اختلاف قیمت ({days} روز اخیر)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

# ================= AUDIT LOGGING =================
async def audit_log(context: ContextTypes.DEFAULT_TYPE, user_id, username, user_msg, bot_response):
    """Enhanced audit logging with both user and bot messages"""
    if not PRIVATE_CHANNEL_ID:
        logger.warning("PRIVATE_CHANNEL_ID not set - skipping audit log")
        return
    
    # Ensure username is not None
    username_display = username if username else "No username"
    
    # Truncate very long messages to avoid Telegram limits
    max_msg_length = 3000
    if len(user_msg) > max_msg_length:
        user_msg = user_msg[:max_msg_length] + "... (truncated)"
    if len(bot_response) > max_msg_length:
        bot_response = bot_response[:max_msg_length] + "... (truncated)"
    
    msg = (
        f"📨 **گزارش تعامل**\n\n"
        f"👤 کاربر: {username_display} (`{user_id}`)\n"
        f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"📩 **پیام کاربر:**\n`{user_msg}`\n\n"
        f"🤖 **پاسخ ربات:**\n{bot_response[:1000]}"  # Limit bot response to prevent overflow
    )
    
    try:
        await context.bot.send_message(
            chat_id=PRIVATE_CHANNEL_ID,
            text=msg,
            parse_mode="Markdown"
        )
        logger.info(f"Audit log sent for user {user_id}")
    except Exception as e:
        logger.error(f"Audit send failed for user {user_id}: {e}")
        # Try sending without markdown as fallback
        try:
            simple_msg = (
                f"📨 گزارش تعامل\n\n"
                f"کاربر: {username_display} ({user_id})\n"
                f"زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"پیام کاربر: {user_msg[:500]}\n\n"
                f"پاسخ ربات: {bot_response[:500]}"
            )
            await context.bot.send_message(
                chat_id=PRIVATE_CHANNEL_ID,
                text=simple_msg
            )
            logger.info(f"Audit log sent (fallback) for user {user_id}")
        except Exception as e2:
            logger.error(f"Audit fallback also failed for user {user_id}: {e2}")

# ================= INLINE KEYBOARDS =================
def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("📊 تحلیل بازار", callback_data="gold")],
        [InlineKeyboardButton("💰 محاسبه گرم", callback_data="calc"),
         InlineKeyboardButton("📈 نمودار قیمت", callback_data="chart")],
        [InlineKeyboardButton("⚙️ تنظیمات", callback_data="settings"),
         InlineKeyboardButton("ℹ️ راهنما", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def settings_keyboard(notifications_on):
    notif_text = "🔔 غیرفعال کردن اعلان‌ها" if notifications_on else "🔕 فعال کردن اعلان‌ها"
    keyboard = [
        [InlineKeyboardButton(notif_text, callback_data="toggle_notif")],
        [InlineKeyboardButton("🎚 تنظیم آستانه‌ها", callback_data="set_thresholds")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ================= COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    add_or_update_user(user.id, user.username, user.first_name)
    
    response = (
        "👋 سلام! به ربات تحلیل طلا خوش آمدید\n\n"
        "این ربات قیمت طلا را بر اساس:\n"
        "• دلار آزاد 💵\n"
        "• اونس جهانی 🌍\n\n"
        "📏 **قوانین تصمیم‌گیری:**\n"
        "🟢 اختلاف کمتر از 100 هزار تومان → خرید\n"
        "🟡 اختلاف 100-500 هزار تومان → صبر و رصد\n"
        "🔴 اختلاف بیش از 500 هزار تومان → فروش\n\n"
        "از منوی زیر استفاده کنید:"
    )
    
    await update.message.reply_text(response, reply_markup=main_menu_keyboard())
    await audit_log(context, user.id, user.username, "/start", response)

async def gold_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
        user_msg = "کلیک روی دکمه تحلیل بازار"
        # Show processing message
        await query.edit_message_text("⏳ در حال دریافت اطلاعات...")
    else:
        user = update.effective_user
        user_msg = update.message.text
        # Show processing message
        processing_msg = await update.message.reply_text("⏳ در حال دریافت اطلاعات...")
    
    settings = get_user_settings(user.id)
    
    try:
        # Fetch gold data (will check multiple posts if needed)
        tala, ounce = fetch_and_parse_gold()
        
        # Fetch USD data (will check multiple posts if needed)
        usd_toman = fetch_and_parse_usd()
        
        fair, var, verdict, emoji, status = analyze_market(
            tala, usd_toman, ounce,
            settings['buy_threshold'],
            settings['wait_threshold']
        )
        
        # Save to history
        save_price_history(tala, usd_toman, ounce, fair, var)
        
        response = (
            f"{emoji} **تحلیل بازار طلا**\n\n"
            f"💵 دلار آزاد: {usd_toman:,} تومان\n"
            f"🌍 اونس جهانی: ${ounce}\n"
            f"🏷 قیمت بازار (هر گرم): {tala:,} تومان\n"
            f"📊 قیمت بازار (مثقال): {int(tala * 4.6):,} تومان\n"
            f"⚖️ قیمت منصفانه: {int(fair):,} تومان\n\n"
            f"📉 اختلاف قیمت: {int(var):,} تومان\n\n"
            f"{verdict}\n\n"
            "👤 Bot creator: @b4bak"
        )
        
        if query:
            await query.edit_message_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        else:
            await processing_msg.edit_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        
        # Audit log with proper error handling
        try:
            await audit_log(context, user.id, user.username, user_msg, response)
        except Exception as e:
            logger.error(f"Failed to log gold_analysis for user {user.id}: {e}")
        
    except Exception as e:
        logger.exception("Gold analysis failed")
        error_msg = "❌ خطا در دریافت اطلاعات. لطفاً دوباره تلاش کنید."
        if query:
            await query.edit_message_text(error_msg, reply_markup=main_menu_keyboard())
        else:
            await processing_msg.edit_text(error_msg, reply_markup=main_menu_keyboard())

async def show_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
        await query.answer("در حال تولید نمودار...")
    else:
        user = update.effective_user
    
    try:
        chart = generate_price_chart()
        if chart is None:
            msg = "📊 داده‌های کافی برای نمودار وجود ندارد. لطفاً بعداً تلاش کنید."
            if query:
                await query.edit_message_text(msg)
            else:
                await update.message.reply_text(msg)
            return
        
        caption = "📈 نمودار مقایسه قیمت طلا (24 ساعت اخیر)"
        
        if query:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=chart,
                caption=caption
            )
        else:
            await update.message.reply_photo(photo=chart, caption=caption)
        
        # Audit log with proper error handling
        try:
            await audit_log(context, user.id, user.username, "درخواست نمودار", "نمودار ارسال شد")
        except Exception as e:
            logger.error(f"Failed to log show_chart for user {user.id}: {e}")
        
    except Exception as e:
        logger.exception("Chart generation failed")
        error_msg = "❌ خطا در تولید نمودار"
        if query:
            await query.answer(error_msg, show_alert=True)
        else:
            await update.message.reply_text(error_msg)

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    if query:
        user = query.from_user
    else:
        user = update.effective_user
    
    settings = get_user_settings(user.id)
    
    response = (
        "⚙️ **تنظیمات شما**\n\n"
        f"🔔 اعلان‌ها: {'فعال' if settings['notifications'] else 'غیرفعال'}\n"
        f"🟢 آستانه خرید: {settings['buy_threshold']:,} تومان\n"
        f"🔴 آستانه فروش: {settings['wait_threshold']:,} تومان\n"
    )
    
    if query:
        await query.edit_message_text(
            response,
            parse_mode="Markdown",
            reply_markup=settings_keyboard(settings['notifications'])
        )
    else:
        await update.message.reply_text(
            response,
            parse_mode="Markdown",
            reply_markup=settings_keyboard(settings['notifications'])
        )

async def toggle_notifications(query, user_id):
    settings = get_user_settings(user_id)
    new_value = 0 if settings['notifications'] else 1
    update_user_settings(user_id, notifications=new_value)
    await query.answer("✅ تنظیمات ذخیره شد")
    await settings_menu(None, None, query)

async def help_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    response = (
        "📚 **راهنمای استفاده**\n\n"
        "**دستورات:**\n"
        "/start - شروع و منوی اصلی\n"
        "/gold - تحلیل بازار طلا\n"
        "/chart - نمودار قیمت\n"
        "/settings - تنظیمات\n"
        "/calc - محاسبه گرم\n\n"
        "**ویژگی‌ها:**\n"
        "🔔 دریافت اعلان زمان خرید مناسب\n"
        "📊 تحلیل لحظه‌ای بازار\n"
        "📈 نمودار روند قیمت\n"
        "⚙️ تنظیمات شخصی‌سازی شده\n\n"
        "👤 Bot creator: @b4bak"
    )
    
    if query:
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
    else:
        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())

# ================= CALLBACK HANDLER =================
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    
    # Check if it's an admin callback
    if query.data.startswith("admin_") or query.data.startswith("chart_") or query.data.startswith("db_") or query.data.startswith("export_"):
        await admin_callback_handler(update, context)
        return
    
    await query.answer()
    
    if query.data == "gold":
        await gold_analysis(update, context, query)
    elif query.data == "chart":
        await show_chart(update, context, query)
    elif query.data == "settings":
        await settings_menu(update, context, query)
    elif query.data == "help":
        await help_menu(update, context, query)
    elif query.data == "main_menu":
        await query.edit_message_text(
            "منوی اصلی:",
            reply_markup=main_menu_keyboard()
        )
    elif query.data == "toggle_notif":
        await toggle_notifications(query, query.from_user.id)
    elif query.data == "calc":
        # Store that we're waiting for calc amount from this user
        context.user_data['waiting_for_calc'] = True
        await query.edit_message_text("💰 مبلغ خود را به تومان وارد کنید:")

# ================= CALC CONVERSATION =================
async def calc_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['waiting_for_calc'] = True
    await update.message.reply_text("💰 مبلغ خود را به تومان وارد کنید:")
    return ASK_AMOUNT

async def calc_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    # Show processing message
    processing_msg = await update.message.reply_text("⏳ در حال دریافت اطلاعات...")
    
    try:
        money = int(update.message.text.replace(",", ""))
        
        # Fetch gold and USD data (will check multiple posts if needed)
        tala, ounce = fetch_and_parse_gold()
        usd_toman = fetch_and_parse_usd()
        
        fair_price = usd_toman * ounce / 41.5
        
        response = (
            f"📊 **محاسبه با {money:,} تومان**\n\n"
            f"🏷 بازار: {money / tala:.2f} گرم\n"
            f"⚖️ منصفانه: {money / fair_price:.2f} گرم\n\n"
            "👤 Bot creator: @b4bak"
        )
        
        await processing_msg.edit_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        
        # Audit log with proper error handling
        try:
            await audit_log(context, user.id, user.username, f"محاسبه: {money:,}", response)
        except Exception as e:
            logger.error(f"Failed to log calc_amount for user {user.id}: {e}")
        
    except ValueError:
        await processing_msg.edit_text("❌ عدد معتبر وارد کنید", reply_markup=main_menu_keyboard())
    except Exception as e:
        logger.exception("Calc failed")
        await processing_msg.edit_text("❌ خطا در دریافت اطلاعات. لطفاً دوباره تلاش کنید.", reply_markup=main_menu_keyboard())
    
    # Clear the flag
    context.user_data['waiting_for_calc'] = False
    return ConversationHandler.END

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages - check if waiting for calc input"""
    if context.user_data.get('waiting_for_calc'):
        # Process as calc amount
        await calc_amount(update, context)
    else:
        # Ignore other text messages or provide help
        pass

# ================= ADMIN COMMANDS =================
def is_admin(user_id):
    return user_id in ADMIN_IDS

def admin_keyboard():
    """Admin main menu keyboard"""
    keyboard = [
        [InlineKeyboardButton("📊 آمار کلی", callback_data="admin_stats"),
         InlineKeyboardButton("👥 آمار کاربران", callback_data="admin_users")],
        [InlineKeyboardButton("💰 آمار قیمت‌ها", callback_data="admin_prices"),
         InlineKeyboardButton("📈 نمودارها", callback_data="admin_charts")],
        [InlineKeyboardButton("💾 مدیریت دیتابیس", callback_data="admin_db"),
         InlineKeyboardButton("📤 خروجی داده", callback_data="admin_export")],
        [InlineKeyboardButton("📢 ارسال پیام همگانی", callback_data="admin_broadcast_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_charts_keyboard():
    """Admin charts menu keyboard"""
    keyboard = [
        [InlineKeyboardButton("📈 نمودار قیمت (24 ساعت)", callback_data="chart_price_24h")],
        [InlineKeyboardButton("📊 نمودار اختلاف (7 روز)", callback_data="chart_diff_7d")],
        [InlineKeyboardButton("👥 نمودار رشد کاربران (30 روز)", callback_data="chart_users_30d")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_db_keyboard():
    """Admin database management keyboard"""
    keyboard = [
        [InlineKeyboardButton("🗑 پاک کردن تاریخچه قدیمی", callback_data="db_clean_old")],
        [InlineKeyboardButton("📊 اطلاعات دیتابیس", callback_data="db_info")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def admin_export_keyboard():
    """Admin export data keyboard"""
    keyboard = [
        [InlineKeyboardButton("👥 خروجی کاربران (CSV)", callback_data="export_users")],
        [InlineKeyboardButton("💰 خروجی قیمت‌ها 7 روز", callback_data="export_prices_7")],
        [InlineKeyboardButton("💰 خروجی قیمت‌ها 30 روز", callback_data="export_prices_30")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Show admin main menu"""
    if query:
        user = query.from_user
    else:
        user = update.effective_user
    
    if not is_admin(user.id):
        if query:
            await query.answer("❌ شما دسترسی ندارید", show_alert=True)
        else:
            await update.message.reply_text("❌ شما دسترسی ندارید")
        return
    
    response = (
        "👑 **پنل مدیریت**\n\n"
        "از منوی زیر گزینه مورد نظر را انتخاب کنید:"
    )
    
    if query:
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())
    else:
        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

async def test_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test audit logging - admin only"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return
    
    user = update.effective_user
    
    # Check if PRIVATE_CHANNEL_ID is set
    if not PRIVATE_CHANNEL_ID:
        await update.message.reply_text(
            "❌ **خطا در تنظیمات**\n\n"
            "PRIVATE_CHANNEL_ID تنظیم نشده است.\n"
            "لطفاً آن را در فایل .env تنظیم کنید."
        )
        return
    
    # Try to send a test message
    test_msg = (
        "🧪 **تست ارسال لاگ**\n\n"
        f"👤 ادمین: {user.username} ({user.id})\n"
        f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "این یک پیام تست است."
    )
    
    try:
        await context.bot.send_message(
            chat_id=PRIVATE_CHANNEL_ID,
            text=test_msg,
            parse_mode="Markdown"
        )
        await update.message.reply_text(
            "✅ **تست موفق**\n\n"
            f"پیام با موفقیت به کانال {PRIVATE_CHANNEL_ID} ارسال شد.\n"
            "لاگ‌ها باید کار کنند."
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ **تست ناموفق**\n\n"
            f"خطا: `{str(e)}`\n\n"
            "**راهنمای رفع مشکل:**\n"
            "1. مطمئن شوید PRIVATE_CHANNEL_ID صحیح است\n"
            "2. ربات باید ادمین کانال باشد\n"
            "3. ID کانال باید با - شروع شود (مثلاً -1001234567890)\n"
            "4. برای گرفتن ID کانال، پیامی را forward کنید به @userinfobot",
            parse_mode="Markdown"
        )

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return
    
    user_count = get_user_count()
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users WHERE notifications = 1')
    notif_count = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM price_history')
    history_count = c.fetchone()[0]
    conn.close()
    
    response = (
        "📊 **آمار کلی ربات**\n\n"
        f"👥 تعداد کاربران: {user_count}\n"
        f"🔔 اعلان فعال: {notif_count}\n"
        f"📈 رکوردهای قیمت: {history_count}\n"
    )
    
    await update.message.reply_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())

async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin panel callbacks"""
    query = update.callback_query
    
    if not is_admin(query.from_user.id):
        await query.answer("❌ شما دسترسی ندارید", show_alert=True)
        return
    
    await query.answer()
    
    if query.data == "admin_menu":
        await admin_menu(update, context, query)
    
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
            "📊 **آمار کلی ربات**\n\n"
            f"👥 کل کاربران: {user_count}\n"
            f"🆕 کاربران جدید (7 روز): {recent_users}\n"
            f"✅ کاربران فعال: {active_users}\n"
            f"🔔 اعلان فعال: {notif_count}\n\n"
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
            "👥 **آمار کاربران**\n\n"
            f"📊 کل کاربران: {user_count}\n"
            f"🆕 عضو شده 7 روز اخیر: {recent_7d}\n"
            f"🆕 عضو شده 30 روز اخیر: {recent_30d}\n\n"
            f"🔔 اعلان فعال: {notif_on}\n"
            f"🔕 اعلان غیرفعال: {notif_off}\n"
            f"📊 نرخ فعال‌سازی: {(notif_on/user_count*100) if user_count > 0 else 0:.1f}%"
        )
        
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())
    
    elif query.data == "admin_prices":
        stats = get_price_stats()
        
        if stats['latest']:
            latest_price, latest_fair, latest_diff, latest_time = stats['latest']
            response = (
                "💰 **آمار قیمت‌ها**\n\n"
                f"**آخرین قیمت:**\n"
                f"🏷 بازار: {latest_price:,} تومان\n"
                f"⚖️ منصفانه: {int(latest_fair):,} تومان\n"
                f"📊 اختلاف: {int(latest_diff):,} تومان\n"
                f"⏰ زمان: {latest_time}\n\n"
            )
            
            if stats['avg_24h'][0]:
                avg_market, avg_fair, avg_diff = stats['avg_24h']
                response += (
                    f"**میانگین 24 ساعت:**\n"
                    f"🏷 بازار: {int(avg_market):,} تومان\n"
                    f"⚖️ منصفانه: {int(avg_fair):,} تومان\n"
                    f"📊 اختلاف: {int(avg_diff):,} تومان\n\n"
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
            response = "💰 **آمار قیمت‌ها**\n\nداده‌ای موجود نیست."
        
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_keyboard())
    
    elif query.data == "admin_charts":
        await query.edit_message_text(
            "📈 **نمودارهای تحلیلی**\n\nنمودار مورد نظر را انتخاب کنید:",
            reply_markup=admin_charts_keyboard()
        )
    
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
        db_size = get_db_size()
        conn = sqlite3.connect('gold_bot.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM price_history')
        total_records = c.fetchone()[0]
        c.execute('''SELECT COUNT(*) FROM price_history 
                     WHERE timestamp < datetime('now', '-30 days')''')
        old_records = c.fetchone()[0]
        conn.close()
        
        response = (
            "💾 **مدیریت دیتابیس**\n\n"
            f"📊 حجم فایل: {db_size:.2f} MB\n"
            f"📈 کل رکوردها: {total_records}\n"
            f"🗑 رکوردهای قدیمی‌تر از 30 روز: {old_records}\n\n"
            "عملیات مورد نظر را انتخاب کنید:"
        )
        
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_db_keyboard())
    
    elif query.data == "db_clean_old":
        deleted = clear_old_price_history(30)
        await query.answer(f"✅ {deleted} رکورد پاک شد", show_alert=True)
        # Refresh the db info
        await query.answer()
        await admin_callback_handler(update, context)  # Re-trigger to show updated info
    
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
        
        response = (
            "📊 **اطلاعات دیتابیس**\n\n"
            f"💾 حجم: {db_size:.2f} MB\n"
            f"📁 مسیر: gold_bot.db\n\n"
            f"**جداول:**\n"
            f"👥 Users: {user_count} رکورد\n"
            f"💰 Price History: {price_count} رکورد\n\n"
        )
        
        if date_range[0]:
            response += f"📅 بازه زمانی: {date_range[0]} تا {date_range[1]}"
        
        await query.edit_message_text(response, parse_mode="Markdown", reply_markup=admin_db_keyboard())
    
    elif query.data == "admin_export":
        await query.edit_message_text(
            "📤 **خروجی داده‌ها**\n\nنوع خروجی را انتخاب کنید:",
            reply_markup=admin_export_keyboard()
        )
    
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
        await query.edit_message_text(
            "📢 **ارسال پیام همگانی**\n\n"
            "برای ارسال پیام به همه کاربران، از دستور زیر استفاده کنید:\n"
            "/broadcast",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="admin_menu")]])
        )

async def admin_broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ شما دسترسی ندارید")
        return ConversationHandler.END
    
    await update.message.reply_text("📢 پیام خود را برای ارسال به همه کاربران وارد کنید:")
    return ASK_BROADCAST

async def admin_broadcast_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    
    conn = sqlite3.connect('gold_bot.db')
    c = conn.cursor()
    c.execute('SELECT user_id FROM users')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    
    success = 0
    failed = 0
    
    for user_id in users:
        try:
            await context.bot.send_message(chat_id=user_id, text=message)
            success += 1
            await asyncio.sleep(0.05)  # Rate limiting
        except:
            failed += 1
    
    await update.message.reply_text(
        f"✅ پیام ارسال شد\n"
        f"موفق: {success}\n"
        f"ناموفق: {failed}"
    )
    
    return ConversationHandler.END

# ================= PRICE MONITORING =================
async def monitor_prices(context: ContextTypes.DEFAULT_TYPE):
    """Background task to monitor prices and send alerts"""
    try:
        # Fetch gold and USD data (will check multiple posts if needed)
        tala, ounce = fetch_and_parse_gold()
        usd_toman = fetch_and_parse_usd()
        
        users = get_all_users_with_notifications()
        
        for user_id in users:
            settings = get_user_settings(user_id)
            fair, var, verdict, emoji, status = analyze_market(
                tala, usd_toman, ounce,
                settings['buy_threshold'],
                settings['wait_threshold']
            )
            
            # Send alert only for BUY status
            if status == "BUY":
                alert_msg = (
                    f"🔔 **هشدار خرید!**\n\n"
                    f"{verdict}\n\n"
                    f"📊 اختلاف قیمت: {int(var):,} تومان\n"
                    f"🏷 قیمت بازار: {tala:,} تومان\n\n"
                    "برای جزئیات بیشتر /gold را بزنید"
                )
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=alert_msg,
                        parse_mode="Markdown"
                    )
                    await asyncio.sleep(0.05)
                except:
                    pass
    
    except Exception as e:
        logger.exception("Price monitoring failed")

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Regular commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("gold", lambda u, c: gold_analysis(u, c)))
    app.add_handler(CommandHandler("chart", lambda u, c: show_chart(u, c)))
    app.add_handler(CommandHandler("settings", lambda u, c: settings_menu(u, c)))
    app.add_handler(CommandHandler("help", lambda u, c: help_menu(u, c)))
    
    # Admin commands
    app.add_handler(CommandHandler("admin", lambda u, c: admin_menu(u, c)))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("test_audit", test_audit))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("broadcast", admin_broadcast_start)],
        states={ASK_BROADCAST: [MessageHandler(filters.TEXT & ~filters.COMMAND, admin_broadcast_send)]},
        fallbacks=[]
    ))
    
    # Calc conversation
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("calc", calc_start)],
        states={ASK_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, calc_amount)]},
        fallbacks=[]
    ))
    
    # Callback handlers (must be before text handler)
    app.add_handler(CallbackQueryHandler(button_callback))
    
    # Handle text messages (for inline button calc)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Job queue for price monitoring (every 30 minutes)
    # Optional: install with `pip install "python-telegram-bot[job-queue]"`
    try:
        job_queue = app.job_queue
        if job_queue:
            job_queue.run_repeating(monitor_prices, interval=1800, first=10)
            logger.info("Price monitoring enabled")
        else:
            logger.warning("JobQueue not available. Install with: pip install 'python-telegram-bot[job-queue]'")
    except Exception as e:
        logger.warning(f"JobQueue setup failed: {e}")
    
    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()