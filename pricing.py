"""Shared text normalization and the 18k gold fair-price formula."""

from __future__ import annotations

# Troy ounce (31.1035 g) / 0.750 fineness ≈ grams of 18k gold per ounce.
GRAMS_PER_OUNCE_18K = 41.5


def normalize(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(persian[i], str(i))
        text = text.replace(arabic[i], str(i))
    return text.replace("٬", ",").replace("،", ",")


def fair_gold_price(usd_toman: float, ounce_usd: float) -> float:
    """Toman per gram of 18k gold implied by USD/Toman and the ounce price."""
    return float(usd_toman) * float(ounce_usd) / GRAMS_PER_OUNCE_18K
