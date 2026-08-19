"""Persian RTL matplotlib helpers for chart generation."""

import logging
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger("gold_bot")

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Regular.ttf",
]

try:
    import arabic_reshaper

    _RESHAPER = arabic_reshaper.ArabicReshaper(
        configuration={"language": "Persian", "delete_harakat": True}
    )
    _HAS_BIDI = True
except ImportError:
    _RESHAPER = None
    _HAS_BIDI = False
    logger.warning("arabic-reshaper not installed; Persian chart text may render incorrectly")


def _find_font_path() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("Noto Naskh Arabic", "Noto Sans Arabic", "Noto Kufi Arabic"):
        try:
            found = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if found and Path(found).exists():
                return found
        except Exception:
            continue
    return None


_FONT_PATH = _find_font_path()
FONT_PROP = fm.FontProperties(fname=_FONT_PATH) if _FONT_PATH else None


def setup_matplotlib_persian():
    """Configure matplotlib for Persian RTL text."""
    matplotlib.use("Agg")
    if _FONT_PATH:
        fm.fontManager.addfont(_FONT_PATH)
        logger.info(f"Chart Persian font: {FONT_PROP.get_name()} ({_FONT_PATH})")
    else:
        logger.warning("No Persian-capable font found; chart labels may not render correctly")
    plt.rcParams["axes.unicode_minus"] = False


def persian_digits(text: str | int | float) -> str:
    """Convert Western digits to Persian numerals."""
    return str(text).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def fa(text: str) -> str:
    """Shape Persian text for matplotlib (reshape only; bidi reorder breaks Noto rendering)."""
    if not text:
        return text
    if _HAS_BIDI:
        return _RESHAPER.reshape(text)
    return text


def fa_period_title(base: str, count: int | str, unit: str) -> str:
    """Build a Persian chart title like «رشد کاربران ۳۰ روز اخیر»."""
    return fa(f"{base} {persian_digits(count)} {unit}")


# Chart label constants (Persian)
LBL_TIME = "زمان"
LBL_PRICE_TOMAN = "قیمت تومان"
LBL_PRICE_USD = "قیمت دلار"
LBL_PRICE_DIFF = "اختلاف قیمت تومان"
LBL_DATE = "تاریخ"
LBL_USER_COUNT = "تعداد کاربران"
LBL_MARKET_PRICE = "قیمت بازار"
LBL_FAIR_PRICE = "قیمت منصفانه"
LBL_USD_TOMAN = "دلار تومان"
LBL_OUNCE_USD = "اونس طلا دلار"
LBL_BUY_THRESHOLD = "آستانه خرید"
LBL_SELL_THRESHOLD = "آستانه فروش"
LBL_PRICE_HISTORY = "تاریخچه قیمت"
LBL_PRICE_DIFF_HISTORY = "تاریخچه اختلاف قیمت"
LBL_GOLD_COMPARISON = "مقایسه قیمت طلا"
LBL_GOLD_COMPARISON_24H = "مقایسه قیمت طلا ۲۴ ساعت اخیر"
LBL_USD_CHART = "نمودار قیمت دلار"
LBL_OUNCE_CHART = "نمودار قیمت اونس طلا"
LBL_USER_GROWTH = "رشد کاربران"
LBL_PRICE_DIFF_TREND = "روند اختلاف قیمت"
LBL_LAST_HOURS = "۲۴ ساعت اخیر"
LBL_DAYS_AGO = "روز اخیر"

CRYPTO_SYMBOL_LABELS = {
    "BTC": "بیت‌کوین",
    "ETH": "اتریوم",
    "TRX": "ترون",
    "USDT": "تتر",
}
LBL_CRYPTO_TOMAN = "قیمت تومان"


def apply_rtl_xaxis(ax):
    """Invert x-axis so time flows right-to-left (RTL reading order)."""
    ax.invert_xaxis()
    plt.setp(ax.get_xticklabels(), rotation=-45, ha='left')


def persian_legend(ax=None, **kwargs):
    """Legend with Persian font on label text."""
    ax = ax or plt.gca()
    legend = ax.legend(**kwargs)
    if legend and FONT_PROP:
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)
    return legend


def set_persian_title(ax, title: str, **kwargs):
    kwargs.setdefault("loc", "center")
    ax.set_title(fa(title), fontproperties=FONT_PROP, **kwargs)


def set_persian_xlabel(ax, label: str, **kwargs):
    ax.set_xlabel(fa(label), fontproperties=FONT_PROP, **kwargs)


def set_persian_ylabel(ax, label: str, **kwargs):
    kwargs.setdefault("rotation", "vertical")
    ax.set_ylabel(fa(label), fontproperties=FONT_PROP, **kwargs)


def finalize_chart(fig=None) -> BytesIO:
    """Apply layout, save to buffer, and close figure."""
    fig = fig or plt.gcf()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Glyph .* missing from font",
            category=UserWarning,
        )
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf
