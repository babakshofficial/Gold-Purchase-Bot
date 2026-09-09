"""User-facing message templates for the gold bot (friendly modern Persian)."""

from datetime import datetime


# ================= BUTTON LABELS =================

BTN_ANALYSIS = "📊 تحلیل بازار"
BTN_ADVISE = "🤖 تحلیل هوشمند"
BTN_PREDICT = "🔮 پیش‌بینی قیمت"
BTN_SETGOAL = "🎯 تعیین هدف"
BTN_CALC = "💰 محاسبه گرم"
BTN_CHART_GOLD = "📈 نمودار قیمت طلا"
BTN_CHART_USD = "📈 نمودار دلار"
BTN_CHART_OUNCE = "📈 نمودار اونس"
BTN_CHARTS = "📈 نمودارها"
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
BTN_PORTFOLIO_CONTINUE = "⏭ بدون تغییر — ادامه"
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
        f"🟢 اختلاف ≤ −{buy_threshold:,} تومان → خرید\n"
        f"🟡 −{buy_threshold:,} < اختلاف < {wait_threshold:,} تومان → نگهداری\n"
        f"🔴 اختلاف ≥ {wait_threshold:,} تومان → فروش\n\n"
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

CHARTS_MENU = "📈 **نمودار مورد نظر را انتخاب کنید:**"
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

        usd_part = f"💵 \u200e${usd:,.2f}\u200e" if usd is not None else "💵 —"
        toman_part = f"💰 \u200f{_format_toman_short(toman)} تومان\u200f" if toman is not None else "💰 —"
        change_part = ""
        if change is not None:
            arrow = "🟢" if change >= 0 else "🔴"
            change_part = f"  {arrow} \u200e{change:+.2f}%\u200e"

        source_note = ""
        lines.append(f"\u200f{emoji} **{name} ({symbol})**{change_part}")
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

def _portfolio_current_line(show_current: bool, formatted: str) -> str:
    if not show_current:
        return ""
    return f"📌 **موجودی فعلی:** {formatted}\n\n"


def _format_crypto_amount(amount: float) -> str:
    if amount >= 1:
        text = f"{amount:,.4f}"
    elif amount >= 0.0001:
        text = f"{amount:.6f}"
    else:
        text = f"{amount:.8f}"
    return text.rstrip("0").rstrip(".")


def portfolio_prompt_gold(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{current:.2f} گرم")
    return (
        "💼 **ثبت دارایی — مرحله ۱ از ۷**\n\n"
        f"{current_line}"
        "مقدار **طلا** خود را به **گرم** وارد کنید.\n"
        "اگر طلا ندارید، `0` بزنید.\n\n"
        "مثال: `25.5`"
    )


def portfolio_prompt_toman(current: int = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{current:,} تومان")
    return (
        "💼 **ثبت دارایی — مرحله ۲ از ۷**\n\n"
        f"{current_line}"
        "مقدار **نقد تومان** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `10000000`"
    )


def portfolio_prompt_usd(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"${current:,.2f}")
    return (
        "💼 **ثبت دارایی — مرحله ۳ از ۷**\n\n"
        f"{current_line}"
        "مقدار **دلار نقد** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `500`"
    )


def portfolio_prompt_btc(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{_format_crypto_amount(current)} BTC")
    return (
        "💼 **ثبت دارایی — مرحله ۴ از ۷**\n\n"
        f"{current_line}"
        "مقدار **بیت‌کوین (BTC)** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `0.05`"
    )


def portfolio_prompt_eth(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{_format_crypto_amount(current)} ETH")
    return (
        "💼 **ثبت دارایی — مرحله ۵ از ۷**\n\n"
        f"{current_line}"
        "مقدار **اتریوم (ETH)** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `1.2`"
    )


def portfolio_prompt_trx(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{_format_crypto_amount(current)} TRX")
    return (
        "💼 **ثبت دارایی — مرحله ۶ از ۷**\n\n"
        f"{current_line}"
        "مقدار **ترون (TRX)** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `5000`"
    )


def portfolio_prompt_usdt(current: float = 0, show_current: bool = False) -> str:
    current_line = _portfolio_current_line(show_current, f"{_format_crypto_amount(current)} USDT")
    return (
        "💼 **ثبت دارایی — مرحله ۷ از ۷**\n\n"
        f"{current_line}"
        "مقدار **تتر (USDT)** خود را وارد کنید.\n"
        "اگر ندارید، `0` بزنید.\n\n"
        "مثال: `1000`"
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


def _portfolio_crypto_lines(crypto_prices: dict, **amounts: float) -> str:
    lines = []
    for symbol in ("BTC", "ETH", "TRX", "USDT"):
        amount = amounts.get(f"crypto_{symbol.lower()}") or 0
        emoji, name = CRYPTO_NAMES.get(symbol, ("🪙", symbol))
        price_toman = (crypto_prices.get(symbol) or {}).get("toman") or 0
        value_toman = amount * price_toman
        lines.append(
            f"{emoji} {name}: {_format_crypto_amount(amount)} — **{value_toman:,.0f}** تومان"
        )
    return "\n".join(lines)


def portfolio_view(
    gold_grams: float,
    cash_toman: int,
    cash_usd: float,
    crypto_btc: float,
    crypto_eth: float,
    crypto_trx: float,
    crypto_usdt: float,
    crypto_prices: dict,
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
    crypto_lines = _portfolio_crypto_lines(
        crypto_prices,
        crypto_btc=crypto_btc,
        crypto_eth=crypto_eth,
        crypto_trx=crypto_trx,
        crypto_usdt=crypto_usdt,
    )
    updated_line = f"🕒 ثبت/به‌روزرسانی: {updated_at}\n" if updated_at else ""
    return (
        "💼 **دارایی‌های شما**\n\n"
        f"🥇 طلا: {gold_grams:.2f} گرم — **{gold_value_toman:,.0f}** تومان\n"
        f"💵 نقد (تومان): **{cash_toman:,}** تومان\n"
        f"💲 نقد (دلار): ${cash_usd:,.2f} — **{usd_value_toman:,.0f}** تومان\n"
        f"{crypto_lines}\n\n"
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
    crypto_btc: float,
    crypto_eth: float,
    crypto_trx: float,
    crypto_usdt: float,
    crypto_prices: dict,
    total_toman: float,
    total_usd: float,
    pnl_toman: float,
    pnl_pct: float,
    tala_price: int,
    usd_toman: float,
) -> str:
    gold_value_toman = gold_grams * tala_price
    usd_value_toman = cash_usd * usd_toman
    crypto_lines = _portfolio_crypto_lines(
        crypto_prices,
        crypto_btc=crypto_btc,
        crypto_eth=crypto_eth,
        crypto_trx=crypto_trx,
        crypto_usdt=crypto_usdt,
    )
    return (
        f"📊 **گزارش روزانه دارایی**\n"
        f"📅 {date_str}\n\n"
        f"🥇 طلا: {gold_grams:.2f} گرم — **{gold_value_toman:,.0f}** تومان\n"
        f"💵 نقد (تومان): **{cash_toman:,}** تومان\n"
        f"💲 نقد (دلار): ${cash_usd:,.2f} — **{usd_value_toman:,.0f}** تومان\n"
        f"{crypto_lines}\n\n"
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
    "با `/portfolio` می‌توانید طلا، نقد تومان، دلار و ارزهای دیجیتال "
    "(BTC، ETH، TRX، USDT) خود را ثبت کنید و هر روز ارزش و سود/زیان را دریافت کنید."
)


# ================= HELP & ABOUT =================

def help_message() -> str:
    return (
        "📚 **راهنمای استفاده**\n\n"
        "**دستورات:**\n"
        "/start — شروع و منوی اصلی\n"
        "/gold — تحلیل بازار طلا\n"
        "/predict — پیش‌بینی قیمت (۱/۷/۳۰ روز)\n"
        "/setgoal — تعیین هدف و ریسک‌پذیری\n"
        "/advise — توصیه هوشمند شخصی‌سازی‌شده\n"
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
        "🤖 پیش‌بینی مدل + توصیه LLM فارسی\n"
        "🪙 قیمت BTC، ETH، TRX و USDT\n"
        "📈 نمودار روند قیمت (از منو)\n"
        "💼 پیگیری دارایی (طلا، نقد، BTC/ETH/TRX/USDT) و گزارش روزانه\n"
        "⚙️ تنظیمات شخصی‌سازی\n\n"
        "👤 سازنده: @b4bak"
    )


def about_message(usd_channel: str, gold_channel: str, crypto_channel: str = "arz_247") -> str:
    return (
        "ℹ️ **درباره ربات**\n\n"
        "این ربات قیمت طلای ۱۸ عیار را با ترکیب دلار آزاد و اونس جهانی تحلیل می‌کند، "
        "با مدل یادگیری ماشین قیمت آینده را پیش‌بینی می‌کند و توصیه فارسی ارائه می‌دهد.\n\n"
        "**منابع قیمت:**\n"
        f"• دلار آزاد: @{usd_channel}\n"
        f"• طلا، اونس و تتر: @{gold_channel}\n"
        f"• ارزهای دیجیتال: @{crypto_channel}\n\n"
        "**سازنده:** @b4bak"
    )


# ================= ML / ADVISE =================

PREDICT_MODELS_MISSING = (
    "❌ مدل پیش‌بینی هنوز آموزش ندیده است.\n"
    "ادمین باید `python train_models.py` را اجرا کند."
)

SETGOAL_PROMPT_GOAL = "🎯 **هدف سرمایه‌گذاری خود را انتخاب کنید:**"
SETGOAL_PROMPT_RISK = "⚖️ **میزان ریسک‌پذیری شما:**"
SETGOAL_NEED_GOAL = (
    "🎯 ابتدا هدف و ریسک خود را با /setgoal تنظیم کنید، "
    "سپس دوباره /advise را بزنید."
)

BTN_GOAL_SHORT = "کوتاه‌مدت (۱-۷ روز)"
BTN_GOAL_MEDIUM = "میان‌مدت (۱-۳ ماه)"
BTN_GOAL_LONG = "بلندمدت (۶+ ماه)"
BTN_GOAL_WEDDING = "پس‌انداز عروسی/طلا"
BTN_GOAL_INFLATION = "حفظ ارزش در برابر تورم"
BTN_RISK_CONS = "محافظه‌کار 🛡️"
BTN_RISK_MED = "متوسط ⚖️"
BTN_RISK_AGG = "جسورانه 🚀"


def setgoal_saved(goal_fa: str, risk_fa: str) -> str:
    return (
        "✅ **پروفایل سرمایه‌گذاری ذخیره شد**\n\n"
        f"🎯 هدف: {goal_fa}\n"
        f"⚖️ ریسک: {risk_fa}\n\n"
        "حالا می‌توانید /advise را بزنید."
    )


def predict_message(prediction: dict) -> str:
    return (
        "🔮 **پیش‌بینی قیمت طلای ۱۸ عیار**\n\n"
        f"🏷 قیمت فعلی: {prediction['price_now']:,.0f} تومان\n"
        f"📅 فردا: {prediction['pred_1d']:,.0f} "
        f"({prediction['expected_return_1d']:+.2f}%)\n"
        f"📅 ۷ روز: {prediction['pred_7d']:,.0f} "
        f"({prediction['expected_return_7d']:+.2f}%)\n"
        f"📅 ۳۰ روز: {prediction['pred_30d']:,.0f} "
        f"({prediction['expected_return_30d']:+.2f}%)\n\n"
        f"💵 دلار: {prediction.get('usd_toman', 0):,.0f} تومان\n"
        f"🌍 اونس: ${prediction.get('ounce', 0):,.2f}\n\n"
        "⚠️ پیش‌بینی مدل آماری است و قطعی نیست."
    )


def advise_header(signal_info: dict) -> str:
    signal_fa = {"BUY": "خرید", "SELL": "فروش", "HOLD": "نگهداری"}.get(
        signal_info["signal"], signal_info["signal"]
    )
    return (
        f"{signal_info['emoji']} **سیگنال شخصی: {signal_fa}**\n"
        f"🎯 {signal_info['goal_fa']} | ⚖️ {signal_info['risk_fa']}\n"
        f"📈 بازده مورد انتظار ({signal_info['horizon']}): "
        f"{signal_info['expected_return']:+.2f}%\n"
        f"_{signal_info['reason_fa']}_\n"
    )


def model_metrics_message(metrics: dict) -> str:
    if not metrics:
        return "📊 متریک مدل: هنوز آموزشی ثبت نشده."
    lines = ["📊 **متریک آخرین آموزش مدل**"]
    for hor, m in metrics.items():
        lines.append(
            f"• {hor}: MAE={m.get('mae', 0):,.0f} | "
            f"MAPE={m.get('mape', 0):.1f}% | backend={m.get('backend', '?')}"
        )
    return "\n".join(lines)


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
