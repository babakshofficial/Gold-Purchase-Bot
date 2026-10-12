"""Single SQLite entry point for the gold bot database."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

CONNECT_TIMEOUT_SEC = 30.0


def resolve_db_path(path: str | None = None) -> str:
    """Prefer an explicit path, then TRAIN_DB_PATH, then the local default."""
    if path:
        return path
    return os.getenv("TRAIN_DB_PATH", "gold_bot.db")


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(resolve_db_path(path), timeout=CONNECT_TIMEOUT_SEC)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def connection(path: str | None = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
