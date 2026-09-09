#!/usr/bin/env python3
"""Periodic model training service for gold price predictors.

Runs independently of the bot (same idea as crawler.py).
Default: retrain every 7 days; override with TRAIN_INTERVAL_HOURS.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from predictor import train_and_save

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_service")

DB_PATH = os.getenv("TRAIN_DB_PATH", "gold_bot.db")
MODELS_DIR = os.getenv("TRAIN_MODELS_DIR", "models")
# Default weekly (168 hours). Example daily: TRAIN_INTERVAL_HOURS=24
TRAIN_INTERVAL_HOURS = float(os.getenv("TRAIN_INTERVAL_HOURS", "168"))
# Run a training pass immediately on startup (1/true/yes)
TRAIN_ON_START = os.getenv("TRAIN_ON_START", "1").strip().lower() in {"1", "true", "yes"}


def run_training() -> bool:
    """Train models once. Returns True on success."""
    db = Path(DB_PATH)
    if not db.exists():
        logger.error("Database not found: %s — skipping train cycle", db.resolve())
        return False

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training: db=%s → models=%s", db, models_dir)
    try:
        metrics = train_and_save(db_path=str(db), models_dir=models_dir, triggered_by="train_service")
        for horizon, m in metrics.items():
            logger.info(
                "Horizon %s | backend=%s | n_train=%s | n_test=%s | MAE=%s | MAPE=%s",
                horizon,
                m.get("backend"),
                m.get("n_train"),
                m.get("n_test"),
                f"{m.get('mae', 0):,.0f}" if "mae" in m else "n/a",
                f"{m.get('mape', 0):.2f}%" if "mape" in m else "n/a",
            )
        logger.info("Training cycle finished successfully")
        return True
    except Exception:
        logger.exception("Training cycle failed")
        return False


def main():
    interval_sec = max(60.0, TRAIN_INTERVAL_HOURS * 3600.0)
    logger.info(
        "Train service started | interval=%.1fh (%.0fs) | on_start=%s | db=%s | models=%s",
        TRAIN_INTERVAL_HOURS,
        interval_sec,
        TRAIN_ON_START,
        DB_PATH,
        MODELS_DIR,
    )

    if TRAIN_ON_START:
        run_training()

    while True:
        logger.info("Sleeping %.1f hours until next training cycle…", TRAIN_INTERVAL_HOURS)
        time.sleep(interval_sec)
        run_training()


if __name__ == "__main__":
    main()
