"""Changelog detection, LLM drafting, and state for restart broadcast prompts."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger("gold_bot")

REPO_ROOT = Path(__file__).resolve().parent
STATE_FILE = REPO_ROOT / "changelog_state.json"
PENDING_FILE = REPO_ROOT / "changelog_pending.md"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


def openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_OPENROUTER_MODEL


PERSIAN_CHANGELOG_SYSTEM = """تو نویسنده اعلامیه به‌روزرسانی برای کاربران یک ربات تلگرام تحلیل طلا هستی.
خروجی باید فقط فارسی باشد (هیچ جمله یا عنوان انگلیسی ننویس).
حداکثر ۸ خط، با بولت‌پوینت (•) و ایموجی ملایم.
قوانین:
- فقط برای کاربران نهایی؛ بدون مسیر فایل، نام ماژول، API، SHA، git یا جزئیات فنی.
- یادداشت‌های فارسی Cursor را مبنا قرار بده؛ کامیت‌های انگلیسی را به زبان ساده فارسی بازگو کن.
- هیچ راز یا توکنی ننویس.
- فقط متن changelog را برگردان، بدون مقدمهٔ جداگانه."""


def looks_mostly_english(text: str) -> bool:
    """Heuristic: Latin-heavy text is probably not user-facing Persian."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return False
    latin = sum(1 for c in letters if ("a" <= c.lower() <= "z"))
    persian = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return latin > persian and latin / len(letters) > 0.35


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
    sha = _run_git("rev-parse", "HEAD")
    if sha:
        return sha
    env_sha = os.getenv("DEPLOY_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    marker = REPO_ROOT / "deploy_sha.txt"
    if marker.exists():
        try:
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    return ""


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


def get_changelog_watermark(state: dict | None = None) -> str:
    """Last HEAD whose changelog was sent or explicitly skipped (do not re-surface)."""
    state = state if state is not None else load_state()
    return (
        state.get("last_changelog_sha")
        or state.get("last_broadcast_sha")
        or state.get("last_skipped_sha")
        or ""
    )


def has_pending_changes(state: dict | None = None) -> bool:
    """True if admins should be prompted to broadcast (new notes, new deploy, or unfinished draft)."""
    state = state if state is not None else load_state()
    head = get_head_sha()
    pending = read_pending_notes()
    watermark = get_changelog_watermark(state)
    last_broadcast = state.get("last_broadcast_sha") or ""
    draft = (state.get("last_draft") or "").strip()

    if pending:
        return True

    if head and last_broadcast and head == last_broadcast:
        return False

    if head and watermark and head == watermark:
        return False

    if head and watermark and head != watermark:
        return True

    if head and not last_broadcast:
        return True

    if draft and not last_broadcast:
        return True

    commits = collect_commit_log(since_sha=watermark or None)
    if commits.strip() and not last_broadcast:
        return True

    return False


def build_change_context(state: dict | None = None) -> dict:
    """Gather context for LLM / fallback drafting."""
    state = state if state is not None else load_state()
    head = get_head_sha()
    since = get_changelog_watermark(state) or None
    commits = collect_commit_log(since_sha=since)
    pending = read_pending_notes()
    return {
        "head_sha": head,
        "since_sha": since,
        "commits": commits,
        "pending": pending,
    }


def _format_pending_bullets(pending: str) -> list[str]:
    lines: list[str] = []
    for raw in pending.splitlines():
        text = raw.strip().lstrip("-•* ").strip()
        if text:
            lines.append(f"• {text}")
    return lines


def _static_persian_changelog(commits: str, pending: str) -> str:
    """Persian-only fallback when LLM is unavailable (never paste English commit subjects)."""
    lines = ["📢 به‌روزرسانی ربات طلا:"]
    bullets = _format_pending_bullets(pending)
    if bullets:
        lines.extend(bullets)
    else:
        commit_lines = [ln for ln in commits.splitlines() if ln.strip()]
        if commit_lines:
            n = len(commit_lines)
            lines.append(f"• نسخه جدید با {n} به‌روزرسانی در تحلیل قیمت و امکانات ربات")
        lines.append("• بهبود پایداری، دقت قیمت‌ها و تجربه کاربری")
        lines.append("• رفع اشکالات گزارش‌شده در نسخه قبل")
    return "\n".join(lines)


async def _call_openrouter_changelog(user_prompt: str, *, max_tokens: int = 500) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ""
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
                    "model": openrouter_model(),
                    "messages": [
                        {"role": "system", "content": PERSIAN_CHANGELOG_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.4,
                },
                timeout=25.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        logger.warning("OpenRouter changelog call failed: %s", e)
        return ""


async def _persian_changelog_fallback(commits: str, pending: str) -> str:
    if pending.strip():
        bullets = _format_pending_bullets(pending)
        if bullets and not any(looks_mostly_english(b) for b in bullets):
            return "📢 به‌روزرسانی ربات طلا:\n" + "\n".join(bullets)

    user_prompt = f"""یادداشت‌های تیم (اولویت بالا — همان‌ها را فارسی روان بنویس):
{pending or '(ندارد)'}

خلاصه تغییرات فنی (فقط برای فهم — در خروجی انگلیسی نیاور):
{commits or '(ندارد)'}

یک changelog کاملاً فارسی برای کاربران تلگرام بنویس."""
    translated = await _call_openrouter_changelog(user_prompt)
    if translated and not looks_mostly_english(translated):
        return translated
    return _static_persian_changelog(commits, pending)


def _fallback_changelog(commits: str, pending: str) -> str:
    """Sync fallback (Persian only). Prefer async _persian_changelog_fallback when possible."""
    return _static_persian_changelog(commits, pending)


async def draft_changelog_text(commits: str = "", pending: str = "") -> str:
    """Draft a short Persian user-facing changelog via OpenRouter."""
    if not commits and not pending:
        ctx = build_change_context()
        commits = ctx["commits"]
        pending = ctx["pending"]

    user_prompt = f"""یادداشت‌های Cursor (ترجیحی — همان محتوا به فارسی روان):
{pending or '(ندارد)'}

کامیت‌های گیت (برای فهم؛ در خروجی انگلیسی ننویس):
{commits or '(ندارد)'}

یک changelog کاملاً فارسی برای کاربران بنویس."""

    content = await _call_openrouter_changelog(user_prompt)
    if content and not looks_mostly_english(content):
        return content

    if content and looks_mostly_english(content):
        logger.info("Changelog LLM returned English; using Persian fallback path")

    return await _persian_changelog_fallback(commits, pending)


def mark_prompted(head_sha: str, draft: str) -> None:
    state = load_state()
    state["last_prompted_sha"] = head_sha
    state["last_draft"] = draft
    save_state(state)


def mark_broadcast(head_sha: str) -> None:
    state = load_state()
    state["last_broadcast_sha"] = head_sha
    state["last_changelog_sha"] = head_sha
    state["last_prompted_sha"] = head_sha
    state["last_draft"] = ""
    save_state(state)
    clear_pending_notes()


def mark_skipped(head_sha: str) -> None:
    state = load_state()
    state["last_prompted_sha"] = head_sha
    state["last_changelog_sha"] = head_sha
    state["last_skipped_sha"] = head_sha
    state["last_draft"] = ""
    save_state(state)
    clear_pending_notes()
