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
from dotenv import load_dotenv

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
REQUEST_TIMEOUT = 10

# Default thresholds (in tomans)
DEFAULT_BUY_THRESHOLD = 100_000
DEFAULT_WAIT_THRESHOLD = 500_000

ASK_AMOUNT = 1
ASK_BROADCAST = 2

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
    c.execute('''INSERT OR REPLACE INTO users (user_id, username, first_name)
                 VALUES (?, ?, ?)''', (user_id, username, first_name))
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

# ================= AUDIT LOGGING =================
async def audit_log(context: ContextTypes.DEFAULT_TYPE, user_id, username, user_msg, bot_response):
    """Enhanced audit logging with both user and bot messages"""
    msg = (
        f"📨 **گزارش تعامل**\n\n"
        f"👤 کاربر: {username} ({user_id})\n"
        f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"📩 **پیام کاربر:**\n{user_msg}\n\n"
        f"🤖 **پاسخ ربات:**\n{bot_response}"
    )
    try:
        await context.bot.send_message(
            chat_id=PRIVATE_CHANNEL_ID,
            text=msg,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.warning(f"Audit send failed: {e}")

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
        "• اونس جهانی 🌍\n"
        "\nمحاسبه می کند"
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
            f"🏷 قیمت بازار: {tala:,} تومان\n"
            f"⚖️ قیمت منصفانه: {int(fair):,} تومان\n\n"
            f"📊 اختلاف قیمت: {int(var):,} تومان\n\n"
            f"{verdict}\n\n"
            "👤 Bot creator: @b4bak"
        )
        
        if query:
            await query.edit_message_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        else:
            await processing_msg.edit_text(response, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        
        await audit_log(context, user.id, user.username, user_msg, response)
        
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
        
        await audit_log(context, user.id, user.username, "درخواست نمودار", "نمودار ارسال شد")
        
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
        await query.edit_message_text("💰 مبلغ خود را به تومان وارد کنید:")

# ================= CALC CONVERSATION =================
async def calc_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await audit_log(context, user.id, user.username, f"محاسبه: {money:,}", response)
        
    except ValueError:
        await processing_msg.edit_text("❌ عدد معتبر وارد کنید")
    except Exception as e:
        logger.exception("Calc failed")
        await processing_msg.edit_text("❌ خطا در دریافت اطلاعات. لطفاً دوباره تلاش کنید.")
    
    return ConversationHandler.END

# ================= ADMIN COMMANDS =================
def is_admin(user_id):
    return user_id in ADMIN_IDS

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
        "📊 **آمار ربات**\n\n"
        f"👥 تعداد کاربران: {user_count}\n"
        f"🔔 اعلان فعال: {notif_count}\n"
        f"📈 رکوردهای قیمت: {history_count}\n"
    )
    
    await update.message.reply_text(response, parse_mode="Markdown")

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
    app.add_handler(CommandHandler("stats", admin_stats))
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
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(button_callback))
    
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