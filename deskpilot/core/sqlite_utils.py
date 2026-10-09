from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_sqlite(path: str | Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    """创建适合桌面端多线程读写的 SQLite 连接。"""
    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection
