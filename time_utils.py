"""Persian Shamsi (Jalali) calendar helpers for user-facing dates."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


def _gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Convert Gregorian Y/M/D to Jalali (jy, jm, jd)."""
    g_d_m = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        355666
        + (365 * gy)
        + ((gy2 + 3) // 4)
        - ((gy2 + 99) // 100)
        + ((gy2 + 399) // 400)
        + gd
        + g_d_m[gm - 1]
    )
    jy = -1595 + (33 * (days // 12053))
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + (days % 31)
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + ((days - 186) % 30)
    return jy, jm, jd


def to_tehran(value: datetime | date | None = None) -> datetime:
    """Normalize to timezone-aware Tehran datetime."""
    if value is None:
        return datetime.now(TEHRAN_TZ)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=TEHRAN_TZ)
    if value.tzinfo is None:
        # Treat naive as UTC (common for DB/ISO storage), then convert.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(TEHRAN_TZ)


def parse_datetime(value: Any) -> datetime | None:
    """Parse datetime / date / ISO string; return None if unusable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:19] if len(text) >= 19 and " " in text else text[:10], fmt)
            except ValueError:
                continue
    return None


def format_shamsi(
    value: Any = None,
    *,
    with_time: bool = True,
    with_seconds: bool = False,
) -> str:
    """
    Format a Gregorian datetime/ISO string as Shamsi in Asia/Tehran.

    Examples: 1405/06/18 16:10   or   1405/06/18
    """
    dt = parse_datetime(value) if value is not None else None
    if value is not None and dt is None:
        return str(value)
    local = to_tehran(dt)
    jy, jm, jd = _gregorian_to_jalali(local.year, local.month, local.day)
    date_part = f"{jy:04d}/{jm:02d}/{jd:02d}"
    if not with_time:
        return date_part
    if with_seconds:
        return f"{date_part} {local.hour:02d}:{local.minute:02d}:{local.second:02d}"
    return f"{date_part} {local.hour:02d}:{local.minute:02d}"


def now_shamsi(*, with_time: bool = True, with_seconds: bool = False) -> str:
    """Current Tehran time as Shamsi string."""
    return format_shamsi(None, with_time=with_time, with_seconds=with_seconds)
