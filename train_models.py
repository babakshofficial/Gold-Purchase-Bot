#!/usr/bin/env python3
"""Train XGBoost gold price models from gold_bot.db into models/."""

import logging
import sys

from predictor import train_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_models")


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "gold_bot.db"
    models_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    logger.info("Training from %s → %s", db, models_dir)
    metrics = train_and_save(db_path=db, models_dir=models_dir, triggered_by="train_models_cli")
    for horizon, m in metrics.items():
        logger.info("%s: %s", horizon, m)
    logger.info("Done.")


if __name__ == "__main__":
    main()
