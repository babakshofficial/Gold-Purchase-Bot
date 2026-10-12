"""Market-versus-fair verdicts shared by the bot and tests."""

from __future__ import annotations

import messages as msg
from pricing import fair_gold_price


def determine_verdict(var, buy_thresh, wait_thresh):
    """Verdict copy, emoji, and status from the market−fair gap."""
    if var <= -buy_thresh:
        return msg.verdict_alert_buy(), "🟢", "BUY"
    if var >= wait_thresh:
        return msg.verdict_alert_sell(), "🔴", "SELL"
    return msg.verdict_alert_wait(), "🟡", "HOLD"


def analyze_market(tala, usd_toman, ounce, buy_threshold, wait_threshold):
    """Signal from market−fair difference vs absolute thresholds.

    BUY  when var <= -buy_threshold
    SELL when var >= +wait_threshold (sell threshold)
    HOLD when -buy_threshold < var < wait_threshold
    """
    fair_price = fair_gold_price(usd_toman, ounce)
    var = tala - fair_price
    if var <= -buy_threshold:
        verdict = msg.verdict_buy()
        emoji = "🟢"
        status = "BUY"
    elif var >= wait_threshold:
        verdict = msg.verdict_sell()
        emoji = "🔴"
        status = "SELL"
    else:
        verdict = msg.verdict_wait()
        emoji = "🟡"
        status = "HOLD"
    return fair_price, var, verdict, emoji, status
