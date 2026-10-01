"""Detect and remove bad rows in price_history (parse errors, wrong USD source, spikes)."""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = "gold_bot.db"

# Rough physical bounds for Iranian 18k gold bot data (Toman / USD).
TALA_MIN = 4_000_000
TALA_MAX = 45_000_000
USD_TOMAN_MIN = 25_000
USD_TOMAN_MAX = 700_000
OUNCE_USD_MIN = 1_200
OUNCE_USD_MAX = 7_000
FAIR_REL_TOL = 0.08
DIFF_ABS_TOL = 80_000
JUMP_FRAC = 0.15
IQR_MULTIPLIER = 3.0
MIN_ROWS_FOR_IQR = 12


@dataclass(frozen=True)
class PriceOutlier:
    row_id: int
    timestamp: str
    tala_price: float
    usd_price: float
    difference: float
    source: str
    reasons: tuple[str, ...]


def _fetch_rows(db_path: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT id, timestamp, tala_price, usd_price, ounce_price,
                   fair_price, difference, COALESCE(source, 'bot')
            FROM price_history
            WHERE tala_price IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """
        )
        return c.fetchall()
    finally:
        conn.close()


def _iqr_bounds(values: list[float], mult: float = IQR_MULTIPLIER) -> tuple[float, float] | None:
    if len(values) < MIN_ROWS_FOR_IQR:
        return None
    sorted_v = sorted(values)
    q1 = statistics.quantiles(sorted_v, n=4)[0]
    q3 = statistics.quantiles(sorted_v, n=4)[2]
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return q1 - mult * iqr, q3 + mult * iqr


def detect_price_outliers(
    db_path: str = DEFAULT_DB,
    *,
    iqr_multiplier: float = IQR_MULTIPLIER,
) -> list[PriceOutlier]:
    rows = _fetch_rows(db_path)
    if not rows:
        return []

    talas: list[float] = []
    usds: list[float] = []
    diffs: list[float] = []
    parsed: list[dict] = []

    for row_id, ts, tala, usd, ounce, fair, diff, source in rows:
        tala_f = float(tala)
        usd_f = float(usd or 0)
        ounce_f = float(ounce or 0)
        fair_f = float(fair or 0)
        diff_f = float(diff or 0)
        parsed.append(
            {
                "id": row_id,
                "ts": str(ts),
                "tala": tala_f,
                "usd": usd_f,
                "ounce": ounce_f,
                "fair": fair_f,
                "diff": diff_f,
                "source": source,
            }
        )
        talas.append(tala_f)
        if usd_f:
            usds.append(usd_f)
        diffs.append(diff_f)

    tala_bounds = _iqr_bounds(talas, iqr_multiplier)
    usd_bounds = _iqr_bounds(usds, iqr_multiplier) if usds else None
    diff_bounds = _iqr_bounds(diffs, iqr_multiplier)

    outliers: list[PriceOutlier] = []
    prev_tala: float | None = None

    for p in parsed:
        reasons: list[str] = []
        tala_f = p["tala"]
        usd_f = p["usd"]
        ounce_f = p["ounce"]
        fair_f = p["fair"]
        diff_f = p["diff"]

        if tala_f < TALA_MIN or tala_f > TALA_MAX:
            reasons.append("tala_out_of_range")
        if usd_f and (usd_f < USD_TOMAN_MIN or usd_f > USD_TOMAN_MAX):
            reasons.append("usd_out_of_range")
        if ounce_f and (ounce_f < OUNCE_USD_MIN or ounce_f > OUNCE_USD_MAX):
            reasons.append("ounce_out_of_range")

        if usd_f > 0 and ounce_f > 0:
            computed_fair = usd_f * ounce_f / 41.5
            if fair_f > 0:
                rel = abs(computed_fair - fair_f) / max(computed_fair, 1.0)
                if rel > FAIR_REL_TOL:
                    reasons.append("fair_mismatch")
            computed_diff = tala_f - computed_fair
            if abs(diff_f - computed_diff) > max(DIFF_ABS_TOL, FAIR_REL_TOL * max(abs(computed_diff), 1.0)):
                reasons.append("difference_mismatch")

        if tala_bounds and (tala_f < tala_bounds[0] or tala_f > tala_bounds[1]):
            reasons.append("tala_iqr")
        if usd_bounds and usd_f and (usd_f < usd_bounds[0] or usd_f > usd_bounds[1]):
            reasons.append("usd_iqr")
        if diff_bounds and (diff_f < diff_bounds[0] or diff_f > diff_bounds[1]):
            reasons.append("difference_iqr")

        if prev_tala is not None and prev_tala > 0:
            jump = abs(tala_f - prev_tala) / prev_tala
            if jump >= JUMP_FRAC:
                reasons.append("tala_jump")

        prev_tala = tala_f

        if reasons:
            outliers.append(
                PriceOutlier(
                    row_id=p["id"],
                    timestamp=p["ts"],
                    tala_price=tala_f,
                    usd_price=usd_f,
                    difference=diff_f,
                    source=p["source"],
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )

    return outliers


REASON_FA = {
    "tala_out_of_range": "قیمت طلا خارج از بازه منطقی",
    "usd_out_of_range": "دلار خارج از بازه (مثلاً ریال/تتر اشتباه)",
    "ounce_out_of_range": "اونس خارج از بازه",
    "fair_mismatch": "قیمت منصفانه با دلار×اونس همخوان نیست",
    "difference_mismatch": "اختلاف ذخیره‌شده با محاسبه فرق دارد",
    "tala_iqr": "قیمت طلا پرت آماری",
    "usd_iqr": "دلار پرت آماری",
    "difference_iqr": "اختلاف پرت آماری",
    "tala_jump": "جهش ناگهانی قیمت طلا",
}


def format_outlier_summary(outliers: list[PriceOutlier], *, max_lines: int = 8) -> str:
    if not outliers:
        return "✅ ردیف پرتی در `price_history` پیدا نشد."

    by_reason: dict[str, int] = {}
    for o in outliers:
        for r in o.reasons:
            by_reason[r] = by_reason.get(r, 0) + 1

    lines = [
        f"⚠️ **{len(outliers)}** ردیف مشکوک در جدول قیمت:",
        "",
        "**دلایل (تعداد):**",
    ]
    for key, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        label = REASON_FA.get(key, key)
        lines.append(f"• {label}: {count}")

    lines.append("")
    lines.append("**نمونه:**")
    for o in outliers[:max_lines]:
        reasons = ", ".join(REASON_FA.get(r, r) for r in o.reasons[:2])
        lines.append(
            f"`id={o.row_id}` {o.timestamp[:16]} — "
            f"طلا {int(o.tala_price):,} | دلار {int(o.usd_price):,} | {reasons}"
        )
    if len(outliers) > max_lines:
        lines.append(f"… و {len(outliers) - max_lines} ردیف دیگر")

    return "\n".join(lines)


def delete_price_outliers(
    db_path: str = DEFAULT_DB,
    row_ids: list[int] | None = None,
    *,
    outliers: list[PriceOutlier] | None = None,
) -> int:
    """Delete rows by id. Pass row_ids or precomputed outliers."""
    if row_ids is None:
        if not outliers:
            outliers = detect_price_outliers(db_path)
        row_ids = [o.row_id for o in outliers]
    if not row_ids:
        return 0

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        placeholders = ",".join("?" * len(row_ids))
        c.execute(f"DELETE FROM price_history WHERE id IN ({placeholders})", row_ids)
        deleted = c.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def summarize_for_cli(db_path: str = DEFAULT_DB) -> int:
    outliers = detect_price_outliers(db_path)
    print(format_outlier_summary(outliers).replace("**", "").replace("`", ""))
    return len(outliers)
