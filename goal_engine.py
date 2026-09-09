"""Goal/risk engine: map predictions to personalized BUY/HOLD/SELL."""

from __future__ import annotations

from typing import Any

GOALS = ("short", "medium", "long", "wedding", "inflation")
RISKS = ("conservative", "medium", "aggressive")

GOAL_LABELS_FA = {
    "short": "کوتاه‌مدت (۱-۷ روز)",
    "medium": "میان‌مدت (۱-۳ ماه)",
    "long": "بلندمدت (۶+ ماه)",
    "wedding": "پس‌انداز عروسی/طلا",
    "inflation": "حفظ ارزش در برابر تورم",
}

RISK_LABELS_FA = {
    "conservative": "محافظه‌کار",
    "medium": "متوسط",
    "aggressive": "جسورانه",
}

# Absolute % return thresholds on the chosen horizon
RISK_THRESHOLDS = {
    "conservative": 1.5,
    "medium": 1.0,
    "aggressive": 0.5,
}


def _pick_expected_return(prediction: dict, goal: str) -> tuple[float, str]:
    """Return (expected_return_pct, horizon_key)."""
    r1 = float(prediction.get("expected_return_1d") or 0)
    r7 = float(prediction.get("expected_return_7d") or 0)
    r30 = float(prediction.get("expected_return_30d") or 0)

    if goal == "short":
        # Blend near-term views
        return (0.4 * r1 + 0.6 * r7), "7d"
    if goal in ("medium",):
        return (0.4 * r7 + 0.6 * r30), "30d"
    if goal in ("long", "wedding", "inflation"):
        return r30, "30d"
    return r7, "7d"


def compute_signal(
    prediction: dict,
    goal: str | None = None,
    risk: str | None = None,
) -> dict[str, Any]:
    goal = goal if goal in GOALS else "medium"
    risk = risk if risk in RISKS else "medium"
    thr = RISK_THRESHOLDS[risk]
    expected, horizon = _pick_expected_return(prediction, goal)

    # Wedding / inflation: prefer HOLD unless move is strong (1.5x threshold)
    effective_thr = thr * 1.5 if goal in ("wedding", "inflation") else thr

    if expected >= effective_thr:
        signal = "BUY"
        reason_fa = (
            f"با افق {horizon} بازده مورد انتظار حدود {expected:+.2f}% است "
            f"(آستانه {effective_thr:.1f}% برای ریسک {RISK_LABELS_FA[risk]})."
        )
    elif expected <= -effective_thr:
        signal = "SELL"
        reason_fa = (
            f"با افق {horizon} بازده مورد انتظار حدود {expected:+.2f}% است "
            f"و از آستانه −{effective_thr:.1f}% پایین‌تر آمده."
        )
    else:
        signal = "HOLD"
        reason_fa = (
            f"بازه مورد انتظار ({expected:+.2f}% در افق {horizon}) "
            f"داخل باند نگهداری (±{effective_thr:.1f}%) است."
        )

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
    return {
        "signal": signal,
        "emoji": emoji,
        "horizon": horizon,
        "expected_return": expected,
        "threshold": effective_thr,
        "goal": goal,
        "risk": risk,
        "goal_fa": GOAL_LABELS_FA[goal],
        "risk_fa": RISK_LABELS_FA[risk],
        "reason_fa": reason_fa,
    }
