"""User-facing message templates for the gold bot (friendly modern Persian)."""

from datetime import datetime


# ================= BUTTON LABELS =================

BTN_ANALYSIS = "📊 تحلیل بازار"
BTN_CALC = "💰 محاسبه گرم"
BTN_CHART_GOLD = "📈 نمودار قیمت"
BTN_CHART_USD = "📈 نمودار دلار"
BTN_CHART_OUNCE = "📈 نمودار اونس"
BTN_HISTORY = "🔍 تاریخچه قیمت"
BTN_SETTINGS = "⚙️ تنظیمات"
BTN_PORTFOLIO = "💼 دارایی من"
BTN_CRYPTO = "🪙 ارزهای دیجیتال"
BTN_ABOUT = "ℹ️ درباره ما"
BTN_HELP = "📚 راهنما"
BTN_BACK = "🔙 بازگشت"
BTN_BACK_MAIN = "🔙 بازگشت به منو اصلی"
BTN_BACK_PREV = "🔙 بازگشت به منو پیشین"
BTN_CANCEL = "❌ لغو"
BTN_UPDATE_PORTFOLIO = "✏️ به‌روزرسانی دارایی"
BTN_BACK_TO_PORTFOLIO = "🔙 بازگشت"
BTN_SET_THRESHOLDS = "🎚 تنظیم آستانه‌ها"

BTN_HISTORY_24H = "📈 ۲۴ ساعت اخیر"
BTN_HISTORY_7D = "📊 ۷ روز اخیر"
BTN_HISTORY_30D = "📈 ۳۰ روز اخیر"

BTN_CRYPTO_BTC = "₿ بیت‌کوین"
BTN_CRYPTO_ETH = "Ξ اتریوم"
BTN_CRYPTO_TRX = "🔴 ترون"
BTN_CRYPTO_USDT = "🟢 تتر"

BTN_THRESHOLD_BUY = "🟢 آستانه خرید"
BTN_THRESHOLD_SELL = "🔴 آستانه فروش"
BTN_THRESHOLD_MOVE = "📊 آستانه حرکت قیمت"

# Admin buttons
BTN_ADMIN_STATS = "📊 آمار کلی"
BTN_ADMIN_USERS = "👥 آمار کاربران"
BTN_ADMIN_PRICES = "💰 آمار قیمت‌ها"
BTN_ADMIN_CHARTS = "📈 نمودارها"
BTN_ADMIN_DB = "💾 مدیریت دیتابیس"
BTN_ADMIN_EXPORT = "📤 خروجی داده"
BTN_ADMIN_BROADCAST = "📢 ارسال پیام همگانی"
BTN_ADMIN_HEALTH = "🔍 چک سلامت"


# ================= WELCOME =================

def welcome_message(buy_threshold: int, wait_threshold: int) -> str:
    return (
        "👋 **سلام! به ربات تحلیل طلا خوش آمدید**\n\n"
        "اینجا قیمت طلای ۱۸ عیار را با ترکیب **دلار آزاد** و **اونس جهانی** "
        "تحلیل می‌کنیم و سیگنال خرید، فروش یا رصد می‌دهیم.\n\n"
        "⚠️ **تذکر:**\n"
        "پیشنهادهای این ربات صرفاً تحلیلی هستند. مسئولیت هر تصمیم مالی با خود شماست.\n\n"
        "📏 **قوانین تصمیم‌گیری (پیش‌فرض):**\n"
        f"🟢 اختلاف کمتر از {buy_threshold:,} تومان → خرید\n"
        f"🟡 اختلاف {buy_threshold:,} تا {wait_threshold:,} تومان → صبر و رصد\n"
        f"🔴 اختلاف بیش از {wait_threshold:,} تومان → فروش\n\n"
        "👇 از منوی زیر شروع کنید:"
    )


MAIN_MENU_HINT = "🏠 **منوی اصلی**\nیکی از گزینه‌ها را انتخاب کنید:"


# ================= ANALYSIS =================

PROCESSING = "⏳ لحظه‌ای صبر کنید، در حال دریافت اطلاعات..."

STALE_DATA_NOTE = (
    "\n\n⚠️ **توجه:** داده‌ها از آخرین رکورد ذخیره‌شده هستند "
    "و ممکن است کمی قدیمی باشند."
)


def gold_analysis_message(
    emoji: str,
    analysis_time: str,
    tala: int,
    fair: float,
    var: float,
    bubble_percentage: float,
    usd_toman: float,
    ounce: float,
    trend_str: str,
    rsi_str,
    volatility_str,
    verdict: str,
    source_note: str = "",
    trend_hours: int = 6,
) -> str:
    return (
        f"{emoji} **تحلیل بازار طلا**{source_note}\n"
        f"🕒 زمان: {analysis_time}\n\n"
        "**💰 قیمت‌ها**\n"
        f"🏷 بازار (هر گرم): {tala:,} تومان\n"
        f"📊 بازار (مثقال): {int(tala * 4.6):,} تومان\n"
        f"⚖️ قیمت منصفانه: {int(fair):,} تومان\n"
        f"📉 اختلاف: {int(var):+,} تومان\n"
        f"🫧 حباب: {bubble_percentage:.2f}%\n"
        f"💵 دلار: {usd_toman:,.0f} تومان\n"
        f"🌍 اونس: ${ounce:,.2f}\n\n"
        f"**📈 تحلیل ({trend_hours} ساعت اخیر)**\n"
        f"• روند: {trend_str}\n"
        f"• RSI: {rsi_str}\n"
        f"• نوسان: {volatility_str}\n\n"
        f"**🎯 سیگنال**\n{verdict}"
    )


def verdict_buy() -> str:
    return "✅ **زمان خرید طلاست!**"


def verdict_wait() -> str:
    return "⏳ **صبر کنید و بازار را رصد کنید**"


def verdict_sell() -> str:
    return "💰 **زمان فروش طلاست!**"


def verdict_alert_buy() -> str:
    return "✅ زمان خرید طلاست!"


def verdict_alert_wait() -> str:
    return "🟡 همچنان صبر کنید"


def verdict_alert_sell() -> str:
    return "🔴 زمان فروش طلاست!"


# ================= CALC =================

CALC_PROMPT = (
    "💰 **محاسبه گرم طلا**\n\n"
    "مبلغی که می‌خواهید به طلا تبدیل کنید را **به تومان** وارد کنید:"
)


def calc_result(
    calc_time: str,
    amount_toman: int,
    price_per_gram: int,
    grams: float,
    source: str,
) -> str:
    return (
        "💰 **نتیجه محاسبه**\n"
        f"🕒 زمان: {calc_time}\n\n"
        f"📥 مبلغ: {amount_toman:,} تومان\n"
        f"🏷 قیمت هر گرم ({source}): {price_per_gram:,} تومان\n"
        f"⚖️ معادل طلا: **{grams:.4f} گرم**"
    )


# ================= CHARTS =================

CHART_SELECT_GOLD = "📈 **بازه زمانی نمودار طلا را انتخاب کنید:**"
CHART_SELECT_USD = "📈 **بازه زمانی نمودار دلار را انتخاب کنید:**"
CHART_SELECT_OUNCE = "📈 **بازه زمانی نمودار اونس را انتخاب کنید:**"
CHART_CANCELLED = "❌ درخواست نمودار لغو شد."
CHART_INSUFFICIENT_DATA = "📊 داده کافی برای این بازه وجود ندارد. لطفاً بعداً دوباره تلاش کنید."


def chart_caption_gold(suffix: str) -> str:
    return f"📈 نمودار قیمت طلا {suffix}"


def chart_caption_usd(suffix: str) -> str:
    return f"📈 نمودار دلار {suffix}"


def chart_caption_ounce(suffix: str) -> str:
    return f"📈 نمودار اونس {suffix}"


def chart_caption_crypto(symbol: str, suffix: str) -> str:
    names = {
        "BTC": "بیت‌کوین",
        "ETH": "اتریوم",
        "TRX": "ترون",
        "USDT": "تتر",
    }
    label = names.get(symbol, symbol)
    return f"📈 نمودار {label} {suffix}"


# ================= CRYPTO =================

CRYPTO_MENU = "🪙 **ارزهای دیجیتال**\nقیمت لحظه‌ای یا نمودار هر ارز را انتخاب کنید:"
CRYPTO_SELECT_CHART = "📈 **بازه زمانی نمودار را انتخاب کنید:**"

CRYPTO_NAMES = {
    "BTC": ("₿", "بیت‌کوین"),
    "ETH": ("Ξ", "اتریوم"),
    "TRX": ("🔴", "ترون"),
    "USDT": ("🟢", "تتر"),
}


def _format_toman_short(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} میلیارد"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} میلیون"
    return f"{value:,.0f}"


def crypto_prices_message(
    prices: dict,
    fetched_at: str,
    stale: bool = False,
    missing: list[str] | None = None,
) -> str:
    lines = ["🪙 **قیمت لحظه‌ای ارزهای دیجیتال**", f"🕒 {fetched_at}"]
    if stale:
        lines.append(STALE_DATA_NOTE.strip())
    lines.append("")

    for symbol in ("BTC", "ETH", "TRX", "USDT"):
        emoji, name = CRYPTO_NAMES.get(symbol, ("", symbol))
        entry = prices.get(symbol)
        if not entry:
            lines.append(f"{emoji} **{name} ({symbol})**")
            lines.append("   ❌ نامشخص")
            lines.append("")
            continue

        usd = entry.get("usd")
        toman = entry.get("toman")
        change = entry.get("change_24h_pct")
        source = entry.get("source", "")

        usd_part = f"💵 ${usd:,.2f}" if usd is not None else "💵 —"
        toman_part = f"💰 {_format_toman_short(toman)} تومان" if toman is not None else "💰 —"
        change_part = ""
        if change is not None:
            arrow = "🟢" if change >= 0 else "🔴"
            change_part = f"  {arrow} {change:+.2f}%"

        source_note = f" _(منبع: {source})_" if source == "ecogold_ir" else ""
        lines.append(f"{emoji} **{name} ({symbol})**{change_part}")
        lines.append(f"   {usd_part}  |  {toman_part}{source_note}")
        lines.append("")

    if missing:
        lines.append(f"⚠️ داده دریافت نشد: {', '.join(missing)}")

    lines.append("📢 منبع: @arz_247")
    return "\n".join(lines)


# ================= HISTORY =================

HISTORY_MENU = "🔍 **بازه زمانی تاریخچه قیمت را انتخاب کنید:**"


# ================= SETTINGS =================

def settings_message(settings: dict) -> str:
    notif_on = settings["notifications"]
    flags = settings["notification_flags"]
    return (
        "⚙️ **تنظیمات شما**\n\n"
        f"🔔 اعلان‌ها: {'فعال ✅' if notif_on else 'غیرفعال ❌'}\n"
        f"🟢 آستانه خرید: {settings['buy_threshold']:,} تومان\n"
        f"🔴 آستانه فروش: {settings['wait_threshold']:,} تومان\n"
        f"📊 آستانه حرکت قیمت: {settings['significant_move_threshold']:,} تومان\n\n"
        "**نوع اعلان‌ها:**\n"
        f"{'🟢' if flags & 1 else '⚪️'} اعلان خرید\n"
        f"{'🔴' if flags & 2 else '⚪️'} اعلان فروش\n"
        f"{'📊' if flags & 4 else '⚪️'} حرکت قیمت\n"
        f"{'📅' if flags & 8 else '⚪️'} خلاصه روزانه بازار\n"
        f"{'💼' if flags & 16 else '⚪️'} گزارش روزانه دارایی"
    )


def settings_notif_toggle_on() -> str:
    return "🔔 غیرفعال کردن اعلان‌ها"


def settings_notif_toggle_off() -> str:
    return "🔕 فعال کردن اعلان‌ها"


THRESHOLDS_MENU = "🎚 **کدام آستانه را می‌خواهید تغییر دهید؟**"
THRESHOLD_BUY_PROMPT = "🟢 **آستانه خرید**\nمقدار جدید را به تومان وارد کنید:"
THRESHOLD_SELL_PROMPT = "🔴 **آستانه فروش**\nمقدار جدید را به تومان وارد کنید:"
THRESHOLD_MOVE_PROMPT = "📊 **آستانه حرکت قیمت**\nمقدار جدید را به تومان وارد کنید:"

SETTINGS_SAVED = "✅ تنظیمات ذخیره شد"
NOTIF_FLAG_UPDATED = "✅ تنظیمات اعلان به‌روزرسانی شد"


def threshold_saved(threshold_type: str, value: int) -> str:
    labels = {
        "buy": "آستانه خرید",
        "wait": "آستانه فروش",
        "significant_move": "آستانه حرکت قیمت",
    }
    label = labels.get(threshold_type, "آستانه")
    return f"✅ {label} به {value:,} تومان تغییر کرد."


# ================= ALERTS =================

def alert_buy(
    analysis_time: str,
    verdict: str,
    var: int,
    tala: int,
    fair: int,
    portfolio_footer: str = "",
) -> str:
    return (
        f"🔔 **هشدار خرید!**\n"
        f"🕒 {analysis_time}\n\n"
        f"{verdict}\n"
        f"📊 اختلاف: {var:,} تومان\n"
        f"🏷 قیمت بازار: {tala:,} تومان\n"
        f"⚖️ قیمت منصفانه: {fair:,} تومان\n\n"
        f"برای جزئیات بیشتر /gold را بزنید"
        f"{portfolio_footer}"
    )


def alert_sell(
    analysis_time: str,
    verdict: str,
    var: int,
    tala: int,
    fair: int,
    portfolio_footer: str = "",
) -> str:
    return (
        f"🔔 **هشدار فروش!**\n"
        f"🕒 {analysis_time}\n\n"
        f"{verdict}\n"
        f"📊 اختلاف: {var:,} تومان\n"
        f"🏷 قیمت بازار: {tala:,} تومان\n"
        f"⚖️ قیمت منصفانه: {fair:,} تومان\n\n"
        f"برای جزئیات بیشتر /gold را بزنید"
        f"{portfolio_footer}"
    )


def alert_significant_move(
    analysis_time: str,
    var: int,
    tala: int,
    fair: int,
    portfolio_footer: str = "",
) -> str:
    return (
        f"🔔 **حرکت مهم قیمت!**\n"
        f"🕒 {analysis_time}\n\n"
        f"📊 اختلاف: {var:,} تومان\n"
        f"🏷 قیمت بازار: {tala:,} تومان\n"
        f"⚖️ قیمت منصفانه: {fair:,} تومان\n\n"
        f"برای جزئیات بیشتر /gold را بزنید"
        f"{portfolio_footer}"
    )


# ================= DAILY SUMMARY (MARKET) =================

def daily_market_summary(
    date_str: str,
    open_price: int,
    close_price: int,
    high_price: int,
    low_price: int,
    avg_price: int,
    total_change: int,
    change_percentage: float,
    bubble_percentage: float,
) -> str:
    return (
        f"📈 **خلاصه روزانه بازار طلا**\n"
        f"📅 {date_str}\n\n"
        f"• باز: {open_price:,} تومان\n"
        f"• بسته: {close_price:,} تومان\n"
        f"• بالاترین: {high_price:,} تومان\n"
        f"• پایین‌ترین: {low_price:,} تومان\n"
        f"• میانگین: {avg_price:,} تومان\n\n"
        f"📊 تغییر روز: {total_change:+,} تومان ({change_percentage:+.2f}%)\n"
        f"🫧 حباب (آخرین قیمت منصفانه): {bubble_percentage:.2f}%"
    )


# ================= PORTFOLIO =================

PORTFOLIO_PROMPT_GOLD = (
    "💼 **ثبت دارایی — مرحله ۱ از ۳**\n\n"
    "مقدار **طلا** خود را به **گرم** وارد کنید.\n"
    "اگر طلا ندارید، `0` بزنید.\n\n"
    "مثال: `25.5`"
)

PORTFOLIO_PROMPT_TOMAN = (
    "💼 **ثبت دارایی — مرحله ۲ از ۳**\n\n"
    "مقدار **نقد تومان** خود را وارد کنید.\n"
    "اگر ندارید، `0` بزنید.\n\n"
    "مثال: `10000000`"
)

PORTFOLIO_PROMPT_USD = (
    "💼 **ثبت دارایی — مرحله ۳ از ۳**\n\n"
    "مقدار **دلار نقد** خود را وارد کنید.\n"
    "اگر ندارید، `0` بزنید.\n\n"
    "مثال: `500`"
)

PORTFOLIO_EMPTY_ERROR = (
    "❌ حداقل یکی از دارایی‌ها باید بیشتر از صفر باشد.\n"
    "لطفاً دوباره `/portfolio` را بزنید."
)

PORTFOLIO_SAVED = (
    "✅ **دارایی‌های شما ثبت شد!**\n\n"
    "از این لحظه، ارزش و سود/زیان نسبت به قیمت‌های فعلی محاسبه می‌شود.\n"
    "برای مشاهده `/portfolio` و برای گزارش روزانه، در تنظیمات «گزارش روزانه دارایی» را فعال کنید."
)


def _pnl_emoji(value: float) -> str:
    if value > 0:
        return "🟢"
    if value < 0:
        return "🔴"
    return "⚪️"


def _format_pnl(value: float, pct: float, currency: str = "تومان") -> str:
    sign = "+" if value >= 0 else ""
    return f"{_pnl_emoji(value)} {sign}{value:,.0f} {currency} ({sign}{pct:.1f}%)"


def portfolio_view(
    gold_grams: float,
    cash_toman: int,
    cash_usd: float,
    total_toman: float,
    total_usd: float,
    pnl_toman: float,
    pnl_usd: float,
    pnl_pct: float,
    tala_price: int,
    usd_toman: float,
    updated_at: str | None,
    stale_note: str = "",
) -> str:
    gold_value_toman = gold_grams * tala_price
    usd_value_toman = cash_usd * usd_toman
    updated_line = f"🕒 ثبت/به‌روزرسانی: {updated_at}\n" if updated_at else ""
    return (
        "💼 **دارایی‌های شما**\n\n"
        f"🥇 طلا: {gold_grams:.2f} گرم — **{gold_value_toman:,.0f}** تومان\n"
        f"💵 نقد (تومان): **{cash_toman:,}** تومان\n"
        f"💲 نقد (دلار): ${cash_usd:,.2f} — **{usd_value_toman:,.0f}** تومان\n\n"
        "**💰 ارزش فعلی**\n"
        f"🇮🇷 {total_toman:,.0f} تومان\n"
        f"🌐 ${total_usd:,.2f}\n\n"
        "**📈 سود/زیان (از زمان ثبت)**\n"
        f"{_format_pnl(pnl_toman, pnl_pct)}\n"
        f"{_format_pnl(pnl_usd, pnl_pct, '$')}\n\n"
        f"🏷 قیمت طلا: {tala_price:,} تومان/گرم\n"
        f"💵 دلار: {usd_toman:,.0f} تومان\n"
        f"{updated_line}"
        f"{stale_note}"
        "\nبرای به‌روزرسانی دارایی‌ها دکمه «به‌روزرسانی» را بزنید."
    )


def portfolio_daily_report(
    date_str: str,
    gold_grams: float,
    cash_toman: int,
    cash_usd: float,
    total_toman: float,
    total_usd: float,
    pnl_toman: float,
    pnl_pct: float,
    tala_price: int,
    usd_toman: float,
) -> str:
    gold_value_toman = gold_grams * tala_price
    usd_value_toman = cash_usd * usd_toman
    return (
        f"📊 **گزارش روزانه دارایی**\n"
        f"📅 {date_str}\n\n"
        f"🥇 طلا: {gold_grams:.2f} گرم — **{gold_value_toman:,.0f}** تومان\n"
        f"💵 نقد (تومان): **{cash_toman:,}** تومان\n"
        f"💲 نقد (دلار): ${cash_usd:,.2f} — **{usd_value_toman:,.0f}** تومان\n\n"
        "**💰 ارزش امروز**\n"
        f"🇮🇷 {total_toman:,.0f} تومان\n"
        f"🌐 ${total_usd:,.2f}\n\n"
        "**📈 سود/زیان (از زمان ثبت)**\n"
        f"{_format_pnl(pnl_toman, pnl_pct)}\n\n"
        f"🏷 قیمت طلا: {tala_price:,} تومان/گرم\n"
        f"💵 دلار: {usd_toman:,.0f} تومان"
    )


PORTFOLIO_NOT_SET = (
    "💼 **هنوز دارایی ثبت نکرده‌اید**\n\n"
    "با `/portfolio` می‌توانید طلا، نقد تومان و دلار خود را ثبت کنید "
    "و هر روز ارزش و سود/زیان را دریافت کنید."
)


# ================= HELP & ABOUT =================

def help_message() -> str:
    return (
        "📚 **راهنمای استفاده**\n\n"
        "**دستورات:**\n"
        "/start — شروع و منوی اصلی\n"
        "/gold — تحلیل بازار طلا\n"
        "/crypto — قیمت ارزهای دیجیتال\n"
        "/portfolio — ثبت و مشاهده دارایی\n"
        "/history — تاریخچه قیمت\n"
        "/settings — تنظیمات\n"
        "/calc — تبدیل تومان به گرم طلا\n"
        "/help — این راهنما\n"
        "/about — درباره ربات\n\n"
        "**ویژگی‌ها:**\n"
        "🔔 اعلان خرید/فروش/حرکت قیمت\n"
        "📊 تحلیل لحظه‌ای بازار\n"
        "🪙 قیمت BTC، ETH، TRX و USDT\n"
        "📈 نمودار روند قیمت (از منو)\n"
        "💼 پیگیری دارایی و گزارش روزانه\n"
        "⚙️ تنظیمات شخصی‌سازی\n\n"
        "👤 سازنده: @b4bak"
    )


def about_message(usd_channel: str, gold_channel: str, crypto_channel: str = "arz_247") -> str:
    return (
        "ℹ️ **درباره ربات**\n\n"
        "این ربات قیمت طلای ۱۸ عیار را با ترکیب دلار آزاد و اونس جهانی تحلیل می‌کند "
        "و قیمت ارزهای دیجیتال منتخب را نیز نمایش می‌دهد.\n\n"
        "**منابع قیمت:**\n"
        f"• دلار آزاد: @{usd_channel}\n"
        f"• طلا، اونس و تتر: @{gold_channel}\n"
        f"• ارزهای دیجیتال: @{crypto_channel}\n\n"
        "**سازنده:** @b4bak"
    )


# ================= ERRORS =================

ERROR_GENERIC = "❌ خطایی رخ داد. لطفاً دوباره تلاش کنید."
ERROR_FETCH = "❌ خطا در دریافت اطلاعات. لطفاً دوباره تلاش کنید."
ERROR_NO_DATA = "❌ داده‌ای برای نمایش وجود ندارد."
ERROR_INVALID_NUMBER = "❌ لطفاً یک عدد معتبر وارد کنید."
ERROR_POSITIVE_NUMBER = "❌ مقدار باید بزرگ‌تر از صفر باشد."
ERROR_NON_NEGATIVE = "❌ مقدار نمی‌تواند منفی باشد."
ERROR_HISTORY_MENU = "❌ خطا در نمایش منوی تاریخچه"
ERROR_INTERNAL = "❌ خطای داخلی"
ERROR_ACCESS_DENIED = "❌ شما دسترسی ندارید"


# ================= ADMIN =================

ADMIN_PANEL = "👑 **پنل مدیریت**\nاز منوی زیر گزینه مورد نظر را انتخاب کنید:"
ADMIN_GENERATING_CHART = "⏳ در حال تولید نمودار..."
ADMIN_CHART_SENT = "✅ نمودار ارسال شد"
ADMIN_NO_CHART_DATA = "❌ داده کافی برای نمودار وجود ندارد"
ADMIN_EXPORT_PREP = "در حال آماده‌سازی..."
ADMIN_FILE_SENT = "✅ فایل ارسال شد"
ADMIN_BROADCAST_MENU = "📢 **ارسال پیام همگانی**\nنوع ارسال را انتخاب کنید:"
ADMIN_BROADCAST_PROMPT = "📢 پیام خود را برای ارسال به همه کاربران وارد کنید:"


def admin_stats_message(
    user_count: int,
    recent_users: int,
    active_users: int,
    notif_count: int,
    history_count: int,
    db_size: float,
) -> str:
    return (
        "📊 **آمار کلی ربات**\n\n"
        f"👥 کل کاربران: {user_count}\n"
        f"🆕 کاربران جدید (۷ روز): {recent_users}\n"
        f"✅ کاربران فعال: {active_users}\n"
        f"🔔 اعلان فعال: {notif_count}\n"
        f"📈 رکوردهای قیمت: {history_count}\n"
        f"💾 حجم دیتابیس: {db_size:.2f} MB"
    )


def admin_users_message(
    user_count: int,
    recent_7d: int,
    recent_30d: int,
    notif_on: int,
    notif_off: int,
) -> str:
    rate = (notif_on / user_count * 100) if user_count > 0 else 0
    return (
        "👥 **آمار کاربران**\n\n"
        f"📊 کل: {user_count}\n"
        f"🆕 ۷ روز اخیر: {recent_7d}\n"
        f"🆕 ۳۰ روز اخیر: {recent_30d}\n"
        f"🔔 اعلان فعال: {notif_on}\n"
        f"🔕 اعلان غیرفعال: {notif_off}\n"
        f"📊 نرخ فعال‌سازی: {rate:.1f}%"
    )


def admin_health_check(status_lines: list[str]) -> str:
    return "🔍 **چک سلامت ربات**\n\n" + "\n".join(status_lines)


def admin_broadcast_result(success: int, failed: int) -> str:
    return f"✅ پیام ارسال شد\nموفق: {success}\nناموفق: {failed}"
