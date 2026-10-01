#!/usr/bin/env python3
"""Scan or delete outlier rows in gold_bot.db price_history."""

from __future__ import annotations

import argparse
import sys

from price_outliers import DEFAULT_DB, delete_price_outliers, detect_price_outliers, summarize_for_cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Find/remove bad price_history rows")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete detected outliers (default: report only)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation when using --delete",
    )
    args = parser.parse_args()

    count = summarize_for_cli(args.db)
    if count == 0:
        return 0

    if not args.delete:
        print("\nDry run. Re-run with --delete to remove these rows.")
        return 0

    if not args.yes:
        answer = input(f"Delete {count} row(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return 1

    deleted = delete_price_outliers(args.db, outliers=detect_price_outliers(args.db))
    print(f"Deleted {deleted} row(s). Consider: python train_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
