"""Changelog detection, LLM drafting, and state for restart broadcast prompts."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger("gold_bot")

STATE_FILE = Path("changelog_state.json")
PENDING_FILE = Path("changelog_pending.md")
REPO_ROOT = Path(__file__).resolve().parent

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read %s", STATE_FILE)
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pending_notes() -> str:
    if not PENDING_FILE.exists():
        return ""
    try:
        lines = []
        for raw in PENDING_FILE.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            lines.append(raw.rstrip())
        return "\n".join(lines).strip()
    except Exception:
        return ""


def clear_pending_notes() -> None:
    if PENDING_FILE.exists():
        try:
            PENDING_FILE.write_text("", encoding="utf-8")
        except Exception:
            logger.warning("Could not clear %s", PENDING_FILE)


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("git %s failed: %s", args, result.stderr.strip())
            return ""
        return result.stdout.strip()
    except Exception as e:
        logger.warning("git command failed: %s", e)
        return ""


def get_head_sha() -> str:
    return _run_git("rev-parse", "HEAD")


def collect_commit_log(since_sha: str | None = None, limit: int = 15) -> str:
    """Return oneline commit subjects since since_sha, or last `limit` commits."""
    if since_sha:
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{since_sha}^{{commit}}"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=5,
        )
        if check.returncode == 0:
            log = _run_git("log", "--oneline", f"{since_sha}..HEAD")
            if log:
                return log
    return _run_git("log", "--oneline", f"-{limit}")


def has_pending_changes(state: dict | None = None) -> bool:
    """True if there are unbroadcast changes that have not been skipped for this HEAD."""
    state = state if state is not None else load_state()
    head = get_head_sha()
    pending = read_pending_notes()
    last_broadcast = state.get("last_broadcast_sha") or ""
    last_prompted = state.get("last_prompted_sha") or ""

    if head and head == last_prompted:
        return False
    if head and head == last_broadcast and not pending:
        return False
    if pending:
        return True
    if not head:
        return False
    return head != last_broadcast


def build_change_context(state: dict | None = None) -> dict:
    """Gather context for LLM / fallback drafting."""
    state = state if state is not None else load_state()
    head = get_head_sha()
    since = state.get("last_broadcast_sha") or None
    commits = collect_commit_log(since_sha=since)
    pending = read_pending_notes()
    return {
        "head_sha": head,
        "since_sha": since,
        "commits": commits,
        "pending": pending,
    }


def _fallback_changelog(commits: str, pending: str) -> str:
    lines = [" ann به‌روزرسانی ربات طلا:"]
    if pending:
        for raw in pending.splitlines():
            text = raw.strip().lstrip("-•* ").strip()
            if text:
                lines.append(f"• {text}")
    elif commits:
        for raw in commits.splitlines()[:12]:
            # strip short sha
            parts = raw.split(" ", 1)
            subject = parts[1] if len(parts) > 1 else raw
            lines.append(f"• {subject}")
    else:
        lines.append("• بهبودها و رفع اشکال‌های اخیر")
    return "\n".join(lines)


async def draft_changelog_text(commits: str = "", pending: str = "") -> str:
    """Draft a short Persian user-facing changelog via OpenRouter."""
    if not commits and not pending:
        ctx = build_change_context()
        commits = ctx["commits"]
        pending = ctx["pending"]

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _fallback_changelog(commits, pending)

    system_prompt = """تو نویسنده اعلامیه به‌روزرسانی برای کاربران یک ربات تلگرام تحلیل طلا هستی.
متن را به فارسی ساده، دوستانه و کوتاه بنویس (حداکثر ۸ خط).
قوانین:
- فقط برای کاربران نهایی؛ بدون مسیر فایل، نام ماژول، API، SHA یا جزئیات فنی داخلی.
- از بولت‌پوینت و ایموجی مناسب استفاده کن.
- روی قابلیت‌های جدید و بهبود تجربه تمرکز کن.
- هیچ راز یا توکنی ننویس.
- فقط متن changelog را برگردان، بدون مقدمه اضافه."""

    user_prompt = f"""یادداشت‌های Cursor (ترجیحی):
{pending or '(ندارد)'}

کامیت‌های گیت:
{commits or '(ندارد)'}

یک changelog فارسی برای کاربران بنویس."""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://t.me/gold_bot",
                    "X-Title": "Gold Bot Changelog",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.5,
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data["choices"][0]["message"]["content"] or "").strip()
            if content:
                return content
    except Exception as e:
        logger.warning("OpenRouter changelog draft failed: %s", e)

    return _fallback_changelog(commits, pending)


def mark_prompted(head_sha: str, draft: str) -> None:
    state = load_state()
    state["last_prompted_sha"] = head_sha
    state["last_draft"] = draft
    save_state(state)


def mark_broadcast(head_sha: str) -> None:
    state = load_state()
    state["last_broadcast_sha"] = head_sha
    state["last_prompted_sha"] = head_sha
    state["last_draft"] = ""
    save_state(state)
    clear_pending_notes()


def mark_skipped(head_sha: str) -> None:
    state = load_state()
    state["last_prompted_sha"] = head_sha
    save_state(state)
