"""OpenRouter LLM advisor for Persian gold market advice."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("gold_bot")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"


def _fallback_advice(
    user_goal: str,
    user_risk: str,
    prediction: dict,
    signal: str,
) -> str:
    ret = prediction.get("expected_return", 0) or 0
    signal_fa = {"BUY": "خرید", "SELL": "فروش", "HOLD": "نگهداری"}.get(signal, signal)
    return (
        "🤖 **تحلیل هوشمند** (آفلاین):\n"
        f"با توجه به هدف «{user_goal}» و ریسک «{user_risk}»، "
        f"بازده مورد انتظار حدود {ret:+.2f}% است.\n"
        f"{'✅' if signal == 'BUY' else '🔴' if signal == 'SELL' else '🟡'} "
        f"توصیه مدل: **{signal_fa}**.\n"
        "⚠️ این تحلیل بر اساس مدل آماری است و توصیه مالی قطعی نیست."
    )


async def get_persian_advice(
    user_goal: str,
    user_risk: str,
    prediction: dict,
    signal: str,
    portfolio: dict | None = None,
) -> str:
    """Call OpenRouter LLM for natural Persian financial advice."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _fallback_advice(user_goal, user_risk, prediction, signal)

    system_prompt = """تو یک مشاور مالی متخصص در بازار طلای ایران هستی.
وظیفه تو ارائه تحلیل و توصیه به زبان فارسی ساده و قابل فهم است.
نکات مهم:
- همیشه تأکید کن که این توصیه مالی قطعی نیست و مسئولیت با کاربر است.
- از اعداد و درصدها به فارسی استفاده کن.
- لحن دوستانه و حرفه‌ای داشته باش.
- پاسخ را در ۳-۵ جمله کوتاه بنویس.
- از ایموجی مناسب استفاده کن."""

    port_line = f"پرتفوی کاربر: {portfolio}" if portfolio else ""
    user_prompt = f"""اطلاعات بازار طلا:
- قیمت فعلی هر گرم طلای ۱۸ عیار: {prediction.get('price_now', 0):,.0f} تومان
- پیش‌بینی قیمت فردا: {prediction.get('pred_1d', 0):,.0f} تومان
- پیش‌بینی قیمت ۷ روز آینده: {prediction.get('pred_7d', 0):,.0f} تومان
- پیش‌بینی قیمت ۳۰ روز آینده: {prediction.get('pred_30d', 0):,.0f} تومان
- نرخ دلار: {prediction.get('usd_toman', 'N/A')} تومان
- انس جهانی: {prediction.get('ounce', 'N/A')} دلار

پروفایل کاربر:
- هدف: {user_goal}
- ریسک‌پذیری: {user_risk}
- سیگنال مدل: {signal}
- بازده مورد انتظار: {prediction.get('expected_return', 0):.2f}%

{port_line}

بر اساس این اطلاعات، یک توصیه کوتاه و عملی به فارسی بده."""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://t.me/gold_bot",
                    "X-Title": "Gold Bot",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return f"🤖 **تحلیل هوشمند:**\n{content}"
    except Exception as e:
        logger.warning("OpenRouter advice failed: %s", e)
        return _fallback_advice(user_goal, user_risk, prediction, signal)
