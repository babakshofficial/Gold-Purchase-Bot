"""Shared OpenRouter chat client for advice and changelog drafts."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("gold_bot")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


def openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_OPENROUTER_MODEL


async def openrouter_chat(
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    title: str,
    timeout: float = 25.0,
) -> str:
    """Return the assistant text, or an empty string when the call cannot be made."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://t.me/gold_bot",
                    "X-Title": title,
                },
                json={
                    "model": openrouter_model(),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:
        logger.warning("OpenRouter call failed (%s): %s", title, exc)
        return ""
